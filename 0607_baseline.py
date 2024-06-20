import torch
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn.model_selection import train_test_split
import numpy as np
import medmnist
from medmnist import INFO
import torchvision.transforms as transforms
from sklearn.metrics import accuracy_score
import logging

# Set up logging
logging.basicConfig(filename='training_log.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s',filemode='a')

PathMNIST_MEAN = [0.73765225, 0.53090023, 0.70307171]
PathMNIST_STD = [0.12319908, 0.17607205, 0.12394462]

# Define the model structure
class ResNet18Baseline(nn.Module):
    def __init__(self, out_dim, num_classes):
        super(ResNet18Baseline, self).__init__()
        self.encoder = self.get_resnet('resnet18')
        self.projector = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, out_dim)
        )
        self.classifier = nn.Linear(out_dim, num_classes)

    def get_resnet(self, base_model):
        model = models.__dict__[base_model](weights=None)  # Load pretrained weights (adjusted)
        model = nn.Sequential(*list(model.children())[:-1])
        return model

    def forward(self, x):
        h = self.encoder(x).squeeze()
        z = self.projector(h)
        logits = self.classifier(z)
        return logits

# Data preprocessing and loading
data_flag = 'pathmnist'
download = True

info = INFO[data_flag]
DataClass = getattr(medmnist, info['python_class'])

data_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=PathMNIST_MEAN, std=PathMNIST_STD)
])

train_dataset = DataClass(split='train', transform=data_transform, download=download)
val_dataset = DataClass(split='val', transform=data_transform, download=download)
test_dataset = DataClass(split='test', transform=data_transform, download=download)

# Extract different proportions of training data

def get_sub_indices(dataset, proportion=0.01):
    class_indices = {}
    
    for i, data in enumerate(dataset): 
        class_idx = int(data[1][0])
        if class_idx not in class_indices:
            class_indices[class_idx] = []
        class_indices[class_idx].append(i)
    
    # List to store subset indices
    subset_indices = []
        
    # Choose proportionate indices from each class
    for indices in class_indices.values():
        selected_indices = np.random.choice(
            indices, 
            size=int(np.ceil(len(indices) * proportion)),
            replace=False
        ).tolist()  # Ensure indices are in a list format
        subset_indices.extend(selected_indices)

    return subset_indices

def get_data_loader(dataset, proportion=0.01, batch_size=256):
    # Get subset indices using the provided function
    split_indices = get_sub_indices(dataset, proportion)
    
    # Create a sampler using the subset indices
    sampler = SubsetRandomSampler(split_indices)
    
    # Create a DataLoader using the sampler
    data_loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    
    return data_loader



# def get_data_loader(dataset, split_size, batch_size=256):
#     num_samples = len(dataset)
#     indices = list(range(num_samples))
#     np.random.shuffle(indices)
#     split_indices = indices[:int(num_samples * split_size)]
#     sampler = SubsetRandomSampler(split_indices)
#     data_loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)
#     return data_loader

# Training function
def train_baseline(model, train_loader, criterion, optimizer, num_epochs=50):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for data, target in train_loader:
            data, target = data.cuda(), target.squeeze().long().cuda()
            optimizer.zero_grad()
            logits = model(data)
            loss = criterion(logits, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        logging.info(f'Epoch {epoch}, Loss: {avg_loss:.4f}')

# Validation function
def validate(model, val_loader):
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.cuda(), target.squeeze().long().cuda()
            logits = model(data)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    accuracy = accuracy_score(all_targets, all_preds)
    return accuracy

# Define the validation data loader
test_loader = DataLoader(dataset=test_dataset, batch_size=256, shuffle=False)
val_loader = DataLoader(dataset=val_dataset, batch_size=256, shuffle=False)

# Instantiate the model, define the loss function and optimizer
out_dim = 128
num_classes = 9
baseline_model = ResNet18Baseline(out_dim=out_dim, num_classes=num_classes).cuda()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(baseline_model.parameters(), lr=3e-4, weight_decay=1e-6)

# Train and validate the model with different data proportions

data_splits = [0.01, 0.1]
# data_splits = [1.0]
for split in data_splits:
    np.random.seed(42)
    train_loader = get_data_loader(train_dataset, split)
    logging.info(f'Training with {int(split*100)}% of the data:')
    train_baseline(baseline_model, val_loader, criterion, optimizer)
    accuracy = validate(baseline_model, test_loader)
    logging.info(f'Validation Accuracy: {accuracy:.4f}')
