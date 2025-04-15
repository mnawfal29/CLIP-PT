import os
import warnings
import gc
import torch
import wandb

from transformers import CLIPProcessor

from utils import seed_everything, parse_args
from src.datasets import CIFAR100FSCIL, CUB200FSCIL, MiniImageNetFSCIL
from src.strategies import CLIPPT
from src.strategies.loss import CrossDispersionLoss
from avalanche.benchmarks import benchmark_from_datasets

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

warnings.filterwarnings("ignore")

datasets = dict(cub200=dict(dataset=CUB200FSCIL, train_epochs_base_class=6, train_mb_size_base_class=4),
                cifar100=dict(dataset=CIFAR100FSCIL, train_mb_size_base_class=32, train_epochs_base_class=8),
                miniimagenet=dict(dataset=MiniImageNetFSCIL, train_mb_size_base_class=32, train_epochs_base_class=8))

args = parse_args()
print(vars(args))

n_runs = args.n_runs
dataset_name = args.dataset
few_shot_examples = [5]

assert n_runs > 0, "Number of runs must be greater than 0."
# NOTE: Seeds used in the paper for 5 runs are: seeds = [42, 13, 50, 24, 69]

img_preprocess = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16").feature_extractor

cfg = vars(args)
cfg["description"] = "new architecture with prompt tuning"
if args.D_s + args.D_g != 12:
    args.D_g = 12 - args.D_s

if __name__ == "__main__":
    wandb.init(project="clip-pt-new", save_code=True, settings=wandb.Settings(code_dir="."), config=cfg)
    wandb.define_metric("Top1_Acc_Stream/eval_phase/test_stream", summary="mean")
    wandb.define_metric("Loss_Stream/eval_phase/test_stream", summary="mean")
    wandb.define_metric("StreamForgetting/eval_phase/test_stream", summary="mean")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    exp_name = f"model_{dataset_name}_L_g_{args.L_g}_L_s_{args.L_s}_D_g_{args.D_g}_D_s_{args.D_s}_txt_beta_{args.txt_beta}_TRM_{args.text_replace_method}_VRM_{args.vision_replace_method}"

    Dataset = datasets[dataset_name]["dataset"]
    Dataset = Dataset(transform=img_preprocess)
    train_mb_size_base_class = datasets[dataset_name]["train_mb_size_base_class"]
    train_epochs_base_class = datasets[dataset_name]["train_epochs_base_class"]

    seed_everything(args.seed)

    strategy = CLIPPT(
        L_g=args.L_g,
        L_s=args.L_s,
        D_g=args.D_g,
        D_s=args.D_s,
        text_replace_method=args.text_replace_method,
        vision_replace_method=args.vision_replace_method,
        regularization_method="balance",
        train_mb_size_base_class=train_mb_size_base_class,
        train_epochs_base_class=train_epochs_base_class,
        num_classes_per_exp=Dataset.num_classes_per_exp,
        classes_per_exp=Dataset.classes_per_exp,
        text_label_mapping=Dataset.text_label_mapping,
        lr=0.00325,
        txt_beta=args.txt_beta,
        use_scheduler=True,
        eval_mb_size=64,
        device=device
    )

    wandb.watch(strategy.model, criterion=CrossDispersionLoss(args.txt_beta), log_freq=100)
    print(strategy.model)
    trainable_params = sum(p.numel() for p in strategy.model.parameters() if p.requires_grad)
    trainable_size = sum(p.numel() * p.element_size() for p in strategy.model.parameters() if p.requires_grad)
    print(f"Trainable Parameters: {trainable_params}")
    print(f"Trainable parameter size: {trainable_size / (1024 ** 2):.2f} MB")
    wandb.log({"Trainable Parameters": trainable_params, "Trainable parameter size": trainable_size / (1024 ** 2)})

    experiences = benchmark_from_datasets(train = Dataset.train_stream, test = Dataset.eval_stream)

    for experience_id, experience in enumerate(experiences.train_stream):
        strategy.train(experience)
        strategy.eval(experiences.test_stream)

        # Filter parameters that require gradients
        trainable_weights = {
            name: param for name, param in strategy.model.state_dict().items()
            if strategy.model.get_parameter(name).requires_grad
        }

        # Save model checkpoint
        model_path = f"checkpoints/{exp_name}_exp_{experience_id}.pt"
        os.makedirs("checkpoints", exist_ok=True)
        torch.save(trainable_weights, model_path)
        
        # Log checkpoint as a W&B Artifact
        artifact = wandb.Artifact(name=f"{exp_name}_exp_{experience_id}_ckpt", type="model")
        artifact.add_file(model_path)
        wandb.log_artifact(artifact)

    torch.cuda.empty_cache()
    gc.collect()

    del strategy
    del experiences
