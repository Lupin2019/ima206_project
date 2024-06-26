import os
import motti
motti.append_parent_dir(__file__)
thisfile = os.path.basename(__file__).split(".")[0]
o_d = motti.o_d()


# lightning
import lightning as L
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from torch.utils.data import DataLoader
from lightning.pytorch.loggers import WandbLogger

import torch
import numpy as np
import itertools

# self define
from args import opts
os.makedirs(opts.log_dir, exist_ok=True)
os.makedirs(opts.ckpt_dir, exist_ok=True)

from utils.confusion_matrix import LogConfusionMatrix

# dataset
from medmnist import PathMNIST
from dataset import pathmnist_collate_fn
from augmentation import  FinetuneTransform
from torch.utils.data import SubsetRandomSampler
from utils.medmnist_subset import get_subset_indices

# model
from model.barlow_twins import BarlowTwinsPretain, BarlowTwinsForImageClassification

# utils
from utils.cross_correlation import LogCrossCorrMatrix

from functools import partial
from typing import Sequence, Tuple, Union

train_transform = FinetuneTransform(img_size=opts.img_size)
val_transform = FinetuneTransform(img_size=opts.img_size)

train_dataset = PathMNIST(
    split="train", download=False, 
    transform=train_transform,
    root="../../data/medmnist2d/",
    size=64
)

val_dataset = PathMNIST(
    split="val", download=False, 
    transform=val_transform,
    root="../../data/medmnist2d/",
    size=64
)

np.random.seed(42) # don't forget this
subset_indices = get_subset_indices(dataset=train_dataset,  proportion=opts.proportion)
subset_indices = list(itertools.chain(*subset_indices.values())) # inplace


train_loader = DataLoader(
    train_dataset, batch_size=opts.batch_size, 
    num_workers=opts.num_workers, drop_last=True,
    collate_fn=pathmnist_collate_fn,
    sampler=SubsetRandomSampler(indices=subset_indices),
    pin_memory=True,
)

val_loader = DataLoader(
    val_dataset, batch_size=opts.batch_size, 
    shuffle=False, num_workers=opts.num_workers, 
    drop_last=False,
    collate_fn=pathmnist_collate_fn,
    pin_memory=True,
)

if opts.reuse and opts.ckpt is not None and opts.ckpt != "" and os.path.exists(opts.ckpt):
    model = BarlowTwinsForImageClassification.load_from_checkpoint(
        checkpoint_path=opts.ckpt,
        lr=opts.lr,
        num_classes=len(train_dataset.info["label"]),
        criterion=torch.nn.CrossEntropyLoss(),
        frozen=opts.frozen,
        warmup_steps=opts.warmup_epochs * len(train_loader),
        train_steps=opts.max_epochs * len(train_loader),
        from_epoch=opts.from_epoch,
    ) 
elif opts.reuse is False and opts.ckpt is not None and opts.ckpt != "" and os.path.exists(opts.ckpt):
    barlow_model = BarlowTwinsPretain.load_from_checkpoint(opts.ckpt)
    model = BarlowTwinsForImageClassification(
        pretrained_model=barlow_model,
        lr=opts.lr,
        num_classes=len(train_dataset.info["label"]),
        criterion=torch.nn.CrossEntropyLoss(),
        frozen=opts.frozen,
        warmup_steps=opts.warmup_epochs * len(train_loader),
        train_steps=opts.max_epochs * len(train_loader),
        from_epoch=opts.from_epoch,
    )
else:
    raise FileNotFoundError("Checkpoint not found !")


checkpoint_callback = ModelCheckpoint(
    save_top_k=1, save_last=True,
    dirpath=os.path.join(opts.ckpt_dir, o_d),
    monitor="val_loss", mode="min"
)

checkpoint_callback2 = ModelCheckpoint(
    save_top_k=-1, save_last=True,
    dirpath=os.path.join(opts.ckpt_dir, o_d, "every_n_epoch"),
    every_n_epochs=opts.max_epochs // 10,
)


wandblogger = WandbLogger(
    name=f"{o_d}_{thisfile}_{opts.ps}", 
    save_dir=opts.log_dir, 
    project="barlow_twins",
)

trainer = L.Trainer(
    max_epochs=opts.max_epochs,
    accelerator="gpu",
    devices=opts.device_num,
    fast_dev_run=opts.fast,
    logger=wandblogger,
    accumulate_grad_batches=opts.accumulate_grad_batches,
    log_every_n_steps=opts.log_step,
    # callbacks=[checkpoint_callback, LogConfusionMatrix()],
    callbacks=[checkpoint_callback, checkpoint_callback2],
)

trainer.fit(
    model=model,
    train_dataloaders=train_loader,
    val_dataloaders=val_loader,
)
