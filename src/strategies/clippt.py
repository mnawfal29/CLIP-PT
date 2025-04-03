from typing import List

import torch
from torch import nn
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts
from transformers import CLIPProcessor

from avalanche.training.templates import SupervisedTemplate
from avalanche.evaluation.metrics import accuracy_metrics, loss_metrics, forgetting_metrics
from avalanche.logging import InteractiveLogger, WandBLogger
from avalanche.training.plugins import EvaluationPlugin
from src.strategies.loss import CrossDispersionLoss

from src.models import CLIPForPromptTuning  # Assuming this is your custom model


class CLIPPT(SupervisedTemplate):
    def __init__(
        self,
        L_g: int,
        L_s: int,
        D_g: int,
        D_s: int,
        text_replace_method: str,
        vision_replace_method: str,
        num_classes_per_exp: List[int],
        classes_per_exp: List[List[int]],
        text_label_mapping: dict,
        regularization_method: str = 'balance',
        manual_prompt: str = '[].',
        lr: float = 0.00325,
        txt_beta: float = 0,
        train_mb_size_base_class: int = 4,
        train_epochs_base_class: int = 3,
        use_scheduler: bool = True,
        train_mb_size: int = 4,
        eval_mb_size: int = 4,
        train_epochs: int = 5,
        device='cpu',
    ):
        # Initialize the model
        model = CLIPForPromptTuning(
            L_g=L_g,
            L_s=L_s,
            D_g=D_g,
            D_s=D_s,
            text_replace_method=text_replace_method,
            vision_replace_method=vision_replace_method,
        )

        self.regularization_method = regularization_method
        self.lr = lr
        self.use_scheduler = use_scheduler
        self.train_mb_size_base_class = train_mb_size_base_class
        self.train_epochs_base_class = train_epochs_base_class
        self.n_classes_per_exp = num_classes_per_exp
        self.classes_per_exp = classes_per_exp
        self.text_label_mapping = text_label_mapping

        # Set up evaluation plugin
        eval_plugin = EvaluationPlugin(
            accuracy_metrics(epoch=True, experience=True, stream=True),
            loss_metrics(epoch=True, experience=True, stream=True),
            forgetting_metrics(experience=True, stream=True),
            loggers=[InteractiveLogger(), WandBLogger()],
        )

        # Initialize the SupervisedTemplate
        super().__init__(
            model=model,
            optimizer=None,  # Optimizer will be set in `make_optimizer`
            criterion=CrossDispersionLoss(txt_beta),
            train_mb_size=train_mb_size,
            train_epochs=train_epochs,
            eval_mb_size=eval_mb_size,
            device=device,
            evaluator=eval_plugin,
        )

        self.actual_text_labels = None
        self.text_tokens = None
        self.attn_mask = None
        self.manual_prompt = manual_prompt
        self.prompt_labels = []
        self.text_preprocess = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16").tokenizer
    
    @property
    def mb_x(self):
        return self.mbatch[0].to(self.device)

    @property
    def mb_y(self):
        return self.mbatch[1].long().to(self.device)

    def forward(self):
        logits, self.text_out = self.model(self.mb_x, self.text_tokens, self.attn_mask)
        return logits

    def criterion(self):
        loss = self._criterion(self.mb_output, self.mb_y, self.text_out)
        return loss

    def make_optimizer(self, reset_optimizer_state=True):
        self.optimizer = torch.optim.SGD(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.lr,
            momentum=0.9,
            weight_decay=1e-5,
        )
        if self.use_scheduler:
            self.scheduler = CosineAnnealingWarmupRestarts(
                self.optimizer,
                first_cycle_steps=200,
                cycle_mult=1.0,
                max_lr=0.1,
                min_lr=0.001,
                warmup_steps=50,
                gamma=1.0,
            )
        super().make_optimizer(reset_optimizer_state=reset_optimizer_state)

    def model_adaptation(self, model=None):
        if model is None:
            model = self.model
        if self.is_training and self.clock.train_exp_counter > 0:
            model = super().model_adaptation(model)
        return model.to(self.device)

    def _after_forward(self):
        old_nclasses = sum(self.n_classes_per_exp[:self.clock.train_exp_counter])
        self.mb_output[:, :old_nclasses] = -9999
        super()._after_forward()

    def _after_training_epoch(self):
        if self.use_scheduler:
            self.scheduler.step()
        super()._after_training_epoch()

    def _before_training_exp(self):
        if self.clock.train_exp_counter == 0:
            self.train_mb_size = self.train_mb_size_base_class
            self.train_epochs = self.train_epochs_base_class
        else:
            self.train_mb_size = 4
            self.train_epochs = 5

        if self.regularization_method == 'freeze' and self.clock.train_exp_counter > 0:
            for param in self.model.parameters():
                param.requires_grad = False

        super()._before_training_exp()
        self.actual_text_labels = [self.text_label_mapping[i] for i in self.classes_per_exp[self.clock.train_exp_counter]]
        self.prompt_labels += [self.manual_prompt.replace('[]', i) for i in self.actual_text_labels]
        out_text_tokens = self.text_preprocess(self.prompt_labels, padding=True, return_tensors="pt")
        self.text_tokens = out_text_tokens["input_ids"].to(self.device)
        self.attn_mask = out_text_tokens["attention_mask"].to(self.device)

    def _before_update(self):
        if self.clock.train_exp_counter > 0 and self.regularization_method == 'balance':
            reg_lambda = self.n_classes_per_exp[self.clock.train_exp_counter] / sum(
                self.n_classes_per_exp[:self.clock.train_exp_counter])
            for param in self.model.parameters():
                if param.grad is not None:
                    param.grad *= reg_lambda

    def eval(self, experience_stream):
        """
        Custom evaluation logic:
        - After training on the nth experience, evaluate on the first n experiences.
        """
        current_experience = self.clock.train_exp_counter
        eval_stream = experience_stream[:current_experience]  # Evaluate on first n experiences
        return super().eval(eval_stream)
