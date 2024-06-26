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

import numpy as np
import itertools
import torch

from transformers import ViTImageProcessor
from constant import PathMNIST_MEAN, PathMNIST_STD

# self define
from args import opts
os.makedirs(opts.log_dir, exist_ok=True)
os.makedirs(opts.ckpt_dir, exist_ok=True)

# dataset
from medmnist import PathMNIST
from dataset import pathmnist_collate_fn
from augmentation import BarlowTwinPretainTransform
from torch.utils.data import SubsetRandomSampler
from utils.medmnist_subset import get_subset_indices

# model
from model.mae import LitViTMAEForPreTraining

# utils
from utils.cross_correlation import LogCrossCorrMatrix


train_dataset = PathMNIST(
    split="train", download=False, 
    transform=None,
    root="../../data/medmnist2d/",
    size=64,
)

val_dataset = PathMNIST(
    split="val", download=False, 
    transform=None,
    root="../../data/medmnist2d/",
    size=64,
)


processor = ViTImageProcessor(
    do_normalize=True,
    do_rescale=True,
    do_resize=True,
    image_mean=PathMNIST_MEAN,
    image_std=PathMNIST_STD,
    size=64,
)

def _pathmnist_collate_fn(batch):
    X = [x[0] for x in batch]
    y = [x[1] for x in batch]
    y = torch.utils.data.default_collate(y)
    X = processor(X, return_tensors="pt").pixel_values
    return X, y.squeeze(-1)

np.random.seed(42) # don't forget this
subset_indices = get_subset_indices(dataset=train_dataset,  proportion=opts.proportion)
subset_indices = list(itertools.chain(*subset_indices.values())) # inplace

train_loader = DataLoader(
    train_dataset, batch_size=opts.batch_size, 
    num_workers=opts.num_workers, drop_last=True,
    collate_fn=_pathmnist_collate_fn,
    sampler=SubsetRandomSampler(indices=subset_indices)
)

val_loader = DataLoader(
    val_dataset, batch_size=opts.batch_size, 
    shuffle=False, num_workers=opts.num_workers, 
    drop_last=False,
    collate_fn=_pathmnist_collate_fn,
)

# z_dim = 1024

# if opts.ckpt is not None and opts.ckpt != "" and os.path.exists(opts.ckpt):
#     barlow_model = BarlowTwinsPretain.load_from_checkpoint(
#         opts.ckpt,
#         lr = opts.lr,
#         z_dim=z_dim,
#         warmup_steps=opts.warmup_epochs * len(train_loader), 
#         train_steps=opts.max_epochs * len(train_loader),
#     )
# else:
#     barlow_model = BarlowTwinsPretain(
#         lr = opts.lr,
#         z_dim=z_dim,
#         warmup_steps=opts.warmup_epochs * len(train_loader), 
#         train_steps=opts.max_epochs * len(train_loader),
#     )


model = LitViTMAEForPreTraining(
    lr=opts.lr,
    warmup_steps=opts.warmup_epochs * len(train_loader), 
    train_steps=opts.max_epochs * len(train_loader),
)

checkpoint_callback = ModelCheckpoint(
    save_top_k=1, save_last=True,
    dirpath=os.path.join(opts.ckpt_dir, o_d),
    monitor="val_loss", mode="min"
)

wandblogger = WandbLogger(
    name=f"{o_d}_{thisfile}_{opts.ps}", 
    save_dir=opts.log_dir, 
    project=opts.project,
)

trainer = L.Trainer(
    max_epochs=opts.max_epochs,
    accelerator="gpu",
    devices=opts.device_num,
    fast_dev_run=opts.fast,
    logger=wandblogger,
    accumulate_grad_batches=opts.accumulate_grad_batches,
    log_every_n_steps=opts.log_step,
    callbacks=[checkpoint_callback],
)

trainer.fit(
    model=model,
    train_dataloaders=train_loader,
    val_dataloaders=val_loader,
)
