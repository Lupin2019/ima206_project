import os
import motti
motti.append_parent_dir(__file__)
thisfile = os.path.basename(__file__).split(".")[0]
o_d = motti.o_d()

from medmnist import PathMNIST
import torchvision.transforms as transforms
from constant import PathMNIST_MEAN, PathMNIST_STD
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
from dataset import pathmnist_collate_fn
from utils.confusion_matrix import pp_matrix_from_data

from dataclasses import dataclass
from typing import Any
import numpy as np

@dataclass
class ClassificationOutput:
    prob: torch.Tensor
    pred_id: torch.Tensor
    y_id: torch.Tensor
    cm: Any

@torch.no_grad()
def get_embeddings(model, dataloader):
    feats = []
    ys = []
    for batch in tqdm(dataloader):
        X, y = batch
        X = X.to(model.device)
        feat = model(X)
        ys.append(y.cpu().detach().numpy())
        feats.append(feat.cpu().detach().numpy())
    feats = np.concatenate(feats, axis=0)
    ys = np.concatenate(ys, axis=0)
    return feats, ys

@torch.no_grad()
def get_embeddings_ft(model, dataloader):
    feats = []
    ys = []
    for batch in tqdm(dataloader):
        X, y = batch
        X = X.to(model.device)
        feat = model._feature_extractor(X)
        ys.append(y.cpu().detach().numpy())
        feats.append(feat.cpu().detach().numpy())
    feats = np.concatenate(feats, axis=0)
    ys = np.concatenate(ys, axis=0)
    return feats, ys
    
def get_test_loader(root="../../data/medmnist2d"):
    test_transforms = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=PathMNIST_MEAN, std=PathMNIST_STD)
    ])

    test_dataset = PathMNIST(split="test", root=root, transform=test_transforms, size=64)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=128, num_workers=8, collate_fn=pathmnist_collate_fn)
    return test_loader

def get_val_loader(root="../../data/medmnist2d"):
    test_transforms = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=PathMNIST_MEAN, std=PathMNIST_STD)
    ])

    test_dataset = PathMNIST(split="val", root=root, transform=test_transforms, size=64)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=128, num_workers=8, collate_fn=pathmnist_collate_fn)
    return test_loader


@torch.no_grad()
def inference_loop(model, dataloader):
    probs = []
    ys = []
    for batch_idx, batch in tqdm(enumerate(dataloader)):
        X, y = batch
        X = X.to(model.device)
        out = model(X)
        
        probs.append(out.cpu())
        ys.append(y.cpu())
    
    prob = torch.cat(probs, dim=0)
    pred_id = torch.cat(probs, dim=0).argmax(dim=1)
    y_id = torch.cat(ys, dim=0)
    return prob, pred_id, y_id


        