import os
import torch
from torch.utils.data import Dataset
import numpy as np
import re
from PIL import Image
import random

CLASSES = ['ecoli', 'efaecalis/kpneumoniae', 'ssaprophyticus/ehormaechei', 
           'paeruginosa/pmirabilis', 'saureus', 'sterile']

class BugSenseData(Dataset):
    def __init__(self, root_dir, partition="train", sequencelength=80, split_ratio=(0.7, 0.15, 0.15), seed=42):
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
        
        # Ensure the root directory exists
        if not os.path.isdir(root_dir):
            raise FileNotFoundError(f"Root directory '{root_dir}' not found.")
        
        # Get all sample paths (subdirectories representing image series)
        self.series_paths = [
            os.path.join(root_dir, folder)
            for folder in sorted(os.listdir(root_dir))
            if os.path.isdir(os.path.join(root_dir, folder)) and not folder.startswith('.')
        ]

        # Generate labels based on the folder name
        for folder in self.series_paths:
            category = str(os.path.basename(folder)).split("_")[0]
            if category in self.categories.keys():
                self.samples.append(folder)
                self.labels.append(CLASSES.index(self.categories[category]))

        # Set random seed for reproducibility
        random.seed(seed)
        
        # Shuffle the samples and split into train, valid, eval partitions
        dataset_size = len(self.samples)
        indices = list(range(dataset_size))
        random.shuffle(indices)

        train_size = 160
        valid_size = 16
        eval_size = 16
      

        # Split indices for each partition
        train_indices = indices[:train_size]
        valid_indices = indices[train_size:train_size + valid_size]
        eval_indices = indices[train_size + valid_size:]

        # Assign the appropriate samples to the partition
        if self.partition == 'train':
            self.samples = [self.samples[i] for i in train_indices]
            self.labels = [self.labels[i] for i in train_indices]
        elif self.partition == 'valid':
            self.samples = [self.samples[i] for i in valid_indices]
            self.labels = [self.labels[i] for i in valid_indices]
        else:  # eval
            self.samples = [self.samples[i] for i in eval_indices]
            self.labels = [self.labels[i] for i in eval_indices]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_path = self.samples[idx]
        label = self.labels[idx]

        # Get all image filenames in the folder
        img_names = [img_name for img_name in os.listdir(sample_path) if img_name.endswith('.png')]

        # Sort images by the number between 'time' and '_' or '.'
        img_names.sort(key=lambda x: float(re.search(r'time([0-9\.]+)[._]', x).group(1)))

        # Extract image means (features) for up to `sequencelength` images
        images = []
        for img_name in img_names[:self.sequencelength]:
            img_path = os.path.join(sample_path, img_name)
            img = Image.open(img_path).convert('HSV')
            img = np.array(img)
            img_means = img.mean(axis=(0, 1))  # Compute mean across height and width
            images.append(img_means)

        # Handle sequence length adjustments
        t = len(images)
        if t < self.sequencelength:
            # Pad with zeros if fewer than `sequencelength` images
            padding = [np.zeros(3)] * (self.sequencelength - t)  # Zero vector for HSV
            images.extend(padding)
        elif t > self.sequencelength:
            # Trim to `sequencelength` images
            images = images[:self.sequencelength]

        images = np.array(images)

        # Convert to tensor
        X = torch.tensor(images, dtype=torch.float32)  # Shape: [sequencelength, features]
        y = torch.tensor([label], dtype=torch.long)  
        return X, y
