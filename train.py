"""
To run: from repo directory (2024-winter-cmap)
> python -m train configs.<config> [--experiment_name <name>] [--aug_type <aug>]
"""

import argparse
import datetime
import importlib.util
import os
import shutil
from pathlib import Path
from typing import Any, DefaultDict, Tuple

# import albumentations as A
import kornia.augmentation as K

# import numpy as np
# from albumentations.pytorch import ToTensorV2
import torch
from kornia.augmentation.container import AugmentationSequential
from segmentation_models_pytorch.losses import JaccardLoss
from torch.nn.modules import Module
from torch.optim import AdamW, Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchgeo.datasets import NAIP, BoundingBox, stack_samples
from torchgeo.samplers import GridGeoSampler, RandomBatchGeoSampler
from torchmetrics import Metric
from torchmetrics.classification import MulticlassJaccardIndex

from data.kc import KaneCounty

# project imports
from utils.model import SegmentationModel
from utils.plot import plot_from_tensors

# import config and experiment name from runtime args
parser = argparse.ArgumentParser(
    description="Train a segmentation model to predict stormwater storage "
    + "and green infrastructure."
)
parser.add_argument("config", type=str, help="Path to the configuration file")
parser.add_argument(
    "--experiment_name",
    type=str,
    help="Name of experiment",
    default=datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
)

# Current potential aug_type args: "all", "default", "plasma", "gauss"
parser.add_argument(
    "--aug_type",
    type=str,
    help="Type of augmentation",
    default="default",
)
args = parser.parse_args()
spec = importlib.util.spec_from_file_location(args.config)
config = importlib.import_module(args.config)

# if no experiment name provided, set to timestamp
exp_name = args.experiment_name
if exp_name is None:
    exp_name = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
aug_type = args.aug_type
if aug_type is None:
    aug_type = "default"

# set output path and exit run if path already exists
out_root = os.path.join(config.OUTPUT_ROOT, exp_name)
os.makedirs(out_root, exist_ok=False)

# create directory for output images
train_images_root = os.path.join(out_root, "train-images")
test_image_root = os.path.join(out_root, "test-images")
os.mkdir(train_images_root)
os.mkdir(test_image_root)

# open tensorboard writer
writer = SummaryWriter(out_root)

# copy training script and config to output directory
shutil.copy(Path(__file__).resolve(), out_root)
shutil.copy(Path(config.__file__).resolve(), out_root)

# build dataset
naip = NAIP(config.KC_IMAGE_ROOT)
kc = KaneCounty(config.KC_MASK_ROOT)
dataset = naip & kc

# train/test split
roi = dataset.bounds
midx = roi.minx + (roi.maxx - roi.minx) / 2
midy = roi.miny + (roi.maxy - roi.miny) / 2

# New train/test 80-20 split
eightyx = roi.minx + (roi.maxx - roi.minx) * 8 / 10
eightyy = roi.miny + (roi.maxy - roi.miny) * 8 / 10

# random batch sampler for training, grid sampler for testing
# Training sampler only splits on x, using eightyx on both train and test properly below
train_roi = BoundingBox(
    roi.minx, eightyx, roi.miny, roi.maxy, roi.mint, roi.maxt
)
train_sampler = RandomBatchGeoSampler(
    dataset=dataset,
    size=config.PATCH_SIZE,
    batch_size=config.BATCH_SIZE,
    roi=train_roi,
)
test_roi = BoundingBox(
    eightyx, roi.maxx, roi.miny, roi.maxy, roi.mint, roi.maxt
)
test_sampler = GridGeoSampler(
    dataset, size=config.PATCH_SIZE, stride=config.PATCH_SIZE, roi=test_roi
)

# create dataloaders (must use batch_sampler)
train_dataloader = DataLoader(
    dataset,
    batch_sampler=train_sampler,
    collate_fn=stack_samples,
    num_workers=config.NUM_WORKERS,
)
test_dataloader = DataLoader(
    dataset,
    batch_size=config.BATCH_SIZE,
    sampler=test_sampler,
    collate_fn=stack_samples,
    num_workers=config.NUM_WORKERS,
)

# get device for training
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

# create the model
model = SegmentationModel(
    model=config.MODEL,
    backbone=config.BACKBONE,
    num_classes=config.NUM_CLASSES,
    weights=config.WEIGHTS,
).model.to(device)
print(model)

# set the loss function, metrics, and optimizer
loss_fn = JaccardLoss(mode="multiclass", classes=config.NUM_CLASSES)
train_jaccard = MulticlassJaccardIndex(
    num_classes=config.NUM_CLASSES,
    ignore_index=config.IGNORE_INDEX,
    average="micro",
).to(device)
test_jaccard = MulticlassJaccardIndex(
    num_classes=config.NUM_CLASSES,
    ignore_index=config.IGNORE_INDEX,
    average="micro",
).to(device)
optimizer = AdamW(model.parameters(), lr=config.LR)

# Various augmentation definitions
default_aug = AugmentationSequential(
    K.RandomHorizontalFlip(p=0.5),
    K.RandomVerticalFlip(p=0.5),
    K.RandomRotation(degrees=360, align_corners=True),
    data_keys=["image", "mask"],
    keepdim=True,
)
plasma_aug = AugmentationSequential(
    K.RandomHorizontalFlip(p=0.5),
    K.RandomVerticalFlip(p=0.5),
    K.RandomPlasmaShadow(
        roughness=(0.1, 0.7),
        shade_intensity=(-1.0, 0.0),
        shade_quantity=(0.0, 1.0),
        keepdim=True,
    ),
    K.RandomRotation(degrees=360, align_corners=True),
    data_keys=["image", "mask"],
    keepdim=True,
)
gauss_aug = AugmentationSequential(
    K.RandomHorizontalFlip(p=0.5),
    K.RandomVerticalFlip(p=0.5),
    K.RandomGaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0), p=0.25),
    K.RandomRotation(degrees=360, align_corners=True),
    data_keys=["image", "mask"],
    keepdim=True,
)
all_aug = AugmentationSequential(
    K.RandomHorizontalFlip(p=0.5),
    K.RandomVerticalFlip(p=0.5),
    K.RandomPlasmaShadow(
        roughness=(0.1, 0.7),
        shade_intensity=(-1.0, 0.0),
        shade_quantity=(0.0, 1.0),
        keepdim=True,
    ),
    K.RandomGaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0), p=0.25),
    K.RandomRotation(degrees=360, align_corners=True),
    data_keys=["image", "mask"],
    keepdim=True,
)

# Mean and Std should likely be 3 or 4 element long tensors, not single numbers
# Need to decide whether these are preset, or based on our own data.
mean = torch.tensor(0.0)
std = torch.tensor(255.0)
normalize = K.Normalize(mean=mean, std=std)
denormalize = K.Denormalize(mean=mean, std=std)

# Choose the proper augmentation format
if aug_type == "plasma":
    aug = plasma_aug
elif aug_type == "gauss":
    aug = gauss_aug
elif aug_type == "all":
    aug = all_aug
else:
    aug = default_aug


def train_setup(
    sample: DefaultDict[str, Any], epoch: int, batch: int
) -> Tuple[torch.Tensor]:
    # send img and mask to device; convert y to float tensor for augmentation
    X = sample["image"].to(device)
    y = sample["mask"].type(torch.float32).to(device)

    # normalize both img and mask to range of [0, 1] (req'd for augmentations)
    X, y = normalize(X), normalize(y)

    # augment img and mask with same augmentations
    X, y = aug(X, y)

    # denormalize mask to reset to index tensor (req'd for loss func)
    y = denormalize(y).type(torch.int64)

    # remove channel dim from y (req'd for loss func)
    y_squeezed = y[:, 0, :, :].squeeze()

    # plot first batch
    if batch == 0:
        save_dir = os.path.join(train_images_root, f"epoch-{epoch}")
        os.mkdir(save_dir)
        for i in range(config.BATCH_SIZE):
            plot_tensors = {
                "image": sample["image"][i],
                "mask": sample["mask"][i],
                "augmented_image": denormalize(X)[i].cpu(),
                "augmented_mask": y[i].cpu(),
            }
            sample_fname = os.path.join(
                save_dir, f"train_sample-{epoch}.{i}.png"
            )
            plot_from_tensors(plot_tensors, sample_fname, "grid")

    return X, y_squeezed


def train(
    dataloader: DataLoader,
    model: Module,
    loss_fn: Module,
    jaccard: Metric,
    optimizer: Optimizer,
    epoch: int,
):
    num_batches = len(dataloader)
    model.train()
    jaccard.reset()
    train_loss = 0
    for batch, sample in enumerate(dataloader):
        X, y = train_setup(sample, epoch, batch)
        # The following comments provide pseudocode to theoretically filter tiles
        # The problem is that X here is a batch, not a specific image, so it won't work
        # Ideally, we filter before sending the dataset to the dataloader.
        # total_pixels = X.size
        # label_count = torch.sum(X != 0)
        # percentage_cover = (label_count / total_pixels) * 100

        # Filter patches based on weight criteria
        # if percentage_cover <= 1:
        # Skip this sample if weight criteria is not met
        #    continue

        # compute prediction error
        outputs = model(X)
        loss = loss_fn(outputs, y)

        # update jaccard index
        preds = outputs.argmax(dim=1)
        jaccard.update(preds, y)

        # backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        train_loss += loss.item()
        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1)
            print(f"loss: {loss:>7f}  [{current:>5d}/{num_batches:>5d}]")
    train_loss /= num_batches
    final_jaccard = jaccard.compute()

    # Need to rename scalars?
    writer.add_scalar("Loss/train", train_loss, epoch)
    writer.add_scalar("Jaccard/train", final_jaccard, epoch)
    print(f"Jaccard Index: {final_jaccard}")


def test(
    dataloader: DataLoader,
    model: Module,
    loss_fn: Module,
    jaccard: Metric,
    epoch: int,
):
    num_batches = len(dataloader)
    model.eval()
    jaccard.reset()
    test_loss = 0
    with torch.no_grad():
        for batch, sample in enumerate(dataloader):
            X = sample["image"].to(device)
            X = normalize(X)
            y = sample["mask"].to(device)
            y_squeezed = y[:, 0, :, :].squeeze()

            # compute prediction error
            outputs = model(X)
            loss = loss_fn(outputs, y_squeezed)

            # update metric
            preds = outputs.argmax(dim=1)
            jaccard.update(preds, y_squeezed)

            # add test loss to rolling total
            test_loss += loss.item()

            # plot first batch
            if batch == 0:
                save_dir = os.path.join(test_image_root, f"epoch-{epoch}")
                os.mkdir(save_dir)
                for i in range(config.BATCH_SIZE):
                    plot_tensors = {
                        "image": sample["image"][i],
                        "ground_truth": sample["mask"][i],
                        "inference": preds[i].cpu(),
                    }
                    sample_fname = os.path.join(
                        save_dir, f"test_sample-{epoch}.{i}.png"
                    )
                    plot_from_tensors(plot_tensors, sample_fname, "row")
    test_loss /= num_batches
    final_jaccard = jaccard.compute()
    writer.add_scalar("Loss/test", test_loss, epoch)
    writer.add_scalar("Jaccard/test", final_jaccard, epoch)
    print(
        f"Test Error: \n Jaccard index: {final_jaccard:>7f}, "
        + f"Avg loss: {test_loss:>7f} \n"
    )

    # Now returns test_loss such that it can be compared against previous losses
    return test_loss


# How much the loss needs to drop to reset a plateau
threshold = 0.01

# How many epochs loss needs to plateau before terminating
patience = 5

# Beginning loss
best_loss = None

# How long it's been plateauing
plateau_count = 0

for t in range(config.EPOCHS):
    print(f"Epoch {t + 1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, train_jaccard, optimizer, t + 1)
    test_loss = test(test_dataloader, model, loss_fn, test_jaccard, t + 1)

    # Checks for plateau
    if best_loss is None:
        best_loss = test_loss
    elif test_loss < best_loss - threshold:
        best_loss = test_loss
        plateau_count = 0
    # Potentially add another if clause to plateau check
    # such that if test_loss jumps up again, plateau resets?
    else:
        plateau_count += 1
        if plateau_count >= patience:
            break
print("Done!")
writer.close()


torch.save(model.state_dict(), os.path.join(out_root, "model.pth"))
print(f"Saved PyTorch Model State to {out_root}")
