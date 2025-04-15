import os
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from avalanche.benchmarks.utils import AvalancheDataset


class PKLDataset(Dataset):
    def __init__(self, data, transform):
        super().__init__()
        self.x = data['images']
        self.y = data['labels']
        self.transform = transform

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        img = Image.open(self.x[idx]).convert('RGB')
        x = self.transform(img, return_tensors="pt").pixel_values[0]
        y = self.y[idx]

        return x, y

class CUB200FSCIL:
    def __init__(self, transform, root="./splits/cub200"): 
        n_exps = 10
        self.num_classes_per_exp = [100] + [10]*n_exps
        self.classes_per_exp = [[i for i in range(100)]] + list(np.split(np.arange(100,200), n_exps))
        
        self.text_label_mapping = {v:k for k,v in pd.read_pickle(os.path.join(root, 'class_to_id.pkl')).items()}

        self.train_stream = [AvalancheDataset(PKLDataset(pd.read_pickle(os.path.join(root, "base_class.pkl"))['train'], transform))]
        self.eval_stream = [AvalancheDataset(PKLDataset(pd.read_pickle(os.path.join(root, "base_class.pkl"))['test'], transform))]
        for i in range(n_exps):
            data = pd.read_pickle(os.path.join(root, f"exp{i}.pkl"))
            train_data = data["train"]
            test_data = data["test"]
            self.train_stream.append(AvalancheDataset(PKLDataset(train_data, transform)))
            self.eval_stream.append(AvalancheDataset(PKLDataset(test_data, transform)))
