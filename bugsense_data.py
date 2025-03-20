import os
import torch
from torch.utils.data import Dataset
import numpy as np
import re
from PIL import Image
import torchvision.transforms as transforms
import random


CLASSES = ['ecoli', 'efaecalis/kpneumoniae', 'ssaprophyticus/ehormaechei', 
           'paeruginosa/pmirabilis', 'saureus', 'sterile']

class BugSenseData(Dataset):
    def __init__(self, root_dir, partition="train", sequencelength=80, split_ratio=(0.7, 0.15,0.15), seed=None):
        """
        Args:
            root_dir (str): Root directory where the dataset is stored.
            partition (str): One of 'train', 'valid', 'eval'. Specifies the data partition to use.
            sequencelength (int): The number of time steps (images) to use for each sample.
            split_ratio (tuple): Proportion of data for the train, valid, eval splits.
            seed (int): Random seed for reproducibility.
        """
        assert partition in ["train", "valid", "eval"], "Invalid partition. Must be 'train', 'valid', or 'eval'."
        
        self.root_dir = root_dir
        self.partition = partition
        self.sequencelength = sequencelength
        self.samples = []
        self.labels = []
        self.categories = {
            'E.C': 'ecoli',
            'E.F': 'efaecalis/kpneumoniae',
            'K.P': 'efaecalis/kpneumoniae',
            'E.H': 'ssaprophyticus/ehormaechei',
            'S.S': 'ssaprophyticus/ehormaechei',
            'P.A': 'paeruginosa/pmirabilis',
            'P.M': 'paeruginosa/pmirabilis',
            'S.A': 'saureus',
            'Ste': 'sterile'
        }
        
        if not os.path.isdir(root_dir):
            raise FileNotFoundError(f"Root directory '{root_dir}' not found.")
        
        # Get all sample paths
        self.series_paths = [
            os.path.join(root_dir, folder)
            for folder in sorted(os.listdir(root_dir))
            if os.path.isdir(os.path.join(root_dir, folder)) and not folder.startswith('.')
        ]

        # Generate labels
        for folder in self.series_paths:
            category = str(os.path.basename(folder)).split("_")[0]
            if category in self.categories.keys():
                self.samples.append(folder)
                self.labels.append(CLASSES.index(self.categories[category]))


        # Set random seed
        if seed is not None:
            random.seed(seed)
        else:
            random.seed()

        
        # Calculate split sizes
        dataset_size = len(self.samples)
        indices = list(range(dataset_size))
        random.shuffle(indices)

        train_size = int(dataset_size * split_ratio[0])
        valid_size = int(dataset_size * split_ratio[1])
        
        # Split indices
        train_indices = indices[:train_size]
        valid_indices = indices[train_size:train_size + valid_size]
        eval_indices = indices[train_size + valid_size:]

        random.shuffle(train_indices)
        random.shuffle(valid_indices)
        random.shuffle(eval_indices)


        # Assign samples based on partition
        if self.partition == 'train':
            self.indices = train_indices
        elif self.partition == 'valid':
            self.indices = valid_indices
        else:  # eval
            self.indices = eval_indices
            
        # Select appropriate samples and labels
        self.samples = [self.samples[i] for i in self.indices]
        self.labels = [self.labels[i] for i in self.indices]

        self.transform = transforms.Compose([
            transforms.CenterCrop((342, 36)),
            transforms.Resize((190, 20)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.504, 0.511, 0.485], std=[0.081, 0.071, 0.065])
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_path = self.samples[idx]
        label = self.labels[idx]

        img_names = [img_name for img_name in os.listdir(sample_path) if img_name.endswith('.png')]
        img_names.sort(key=lambda x: float(re.search(r'time([0-9\.]+)[._]', x).group(1)))

        images = []
        for img_name in img_names[:self.sequencelength]:
                img_path = os.path.join(sample_path, img_name)
                img = Image.open(img_path).convert('RGB')
                if self.transform:
                    img = self.transform(img)
                images.append(img)

        images = torch.stack(images)
        images = images.permute(1, 0, 2, 3)
        y = torch.tensor([label], dtype=torch.long)
        
        return images, y

    def get_labels(self):
        """Returns all labels in the dataset."""
        return self.labels
    