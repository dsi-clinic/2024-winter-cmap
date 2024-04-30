"""
To run: from repo directory (2024-winter-cmap)
> python train.py configs.<config> [--experiment_name <name>]
    [--aug_type <aug>] [--split <split>]
"""

import argparse
import datetime
import importlib.util
import logging
import os
import shutil
import random
import sys
from pathlib import Path
from statistics import mean, stdev
from typing import Any, DefaultDict, Tuple

import kornia.augmentation as K
import torch
import wandb
import yaml
from kornia.augmentation.container import AugmentationSequential
from torch.nn.modules import Module
from torch.optim import AdamW, Optimizer
from torch.utils.data import DataLoader

from torch.utils.tensorboard import SummaryWriter
from torchgeo.datasets import NAIP, random_bbox_assignment
from torchmetrics import Metric
from torchmetrics.classification import MulticlassJaccardIndex

from data.kcv import KaneCounty
from data.dem import KaneDEM
from utils.model import SegmentationModel
from utils.plot import plot_from_tensors
from utils.sampler import (
    BalancedGridGeoSampler,
    BalancedRandomBatchGeoSampler,
    collate_samples,
)

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
aug_types = "'all', 'default', 'plasma', 'gauss'"

# adding arguments
parser.add_argument(
    "--aug_type",
    type=str,
    help="Type of augmentation; potential inputs are: {aug_types}",
    default="default",
)

parser.add_argument(
    "--split",
    type=str,
    help="Ratio of split; enter the size of the train split as an int out of 100",
    default="80",
)


parser.add_argument(
    "--tune",
    type=str,
    help="Whether to apply hyperparameter tuning with wandb; enter True or False",
    default=False,
)

parser.add_argument(
    "--tune_key",
    type=str,
    help="Please enter the API key for wandb tuning",
    default="",
)

parser.add_argument(
    "--num_trials",
    type=str,
    help="Please enter the number of trial for each train",
    default="1",
)

args = parser.parse_args()
spec = importlib.util.spec_from_file_location(args.config)
config = importlib.import_module(args.config)
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)


def arg_parsing():
    # if no experiment name provided, set to timestamp
    exp_name = args.experiment_name
    if exp_name is None:
        exp_name = f'{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}'
    aug_type = args.aug_type
    if aug_type is None:
        aug_type = "default"
    split = float(int(args.split) / 100)
    if split is None:
        split = 0.80

    # tuning with wandb
    wandb_tune = args.tune
    num_trials = int(args.num_trials)

    if wandb_tune:
        print("wandb tuning")
        wandb.login(key=args.tune_key)

    return exp_name, aug_type, split, wandb_tune, num_trials


def writer_prep(exp_name, trial_num):
    # set output path and exit run if path already exists
    exp_trial_name = f"{exp_name}_trial{trial_num}"
    out_root = os.path.join(config.OUTPUT_ROOT, exp_trial_name)
    if wandb_tune:
        os.makedirs(out_root, exist_ok=True)
    else:
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

    # Set up logging
    log_filename = os.path.join(out_root, "training_log.txt")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler(sys.stdout),
        ],
    )
    return train_images_root, test_image_root, out_root, writer


def initialize_dataset():
    naip = NAIP(config.KC_IMAGE_ROOT)

    shape_path = os.path.join(config.KC_SHAPE_ROOT, config.KC_SHAPE_FILENAME)
    kc = KaneCounty(
        shape_path,
        config.KC_LAYER,
        config.KC_LABEL_COL,
        config.KC_LABELS,
        config.CONTEXT_SIZE,
        naip.crs,
        naip.res,
    )
    if config.KC_DEM_ROOT is not None:
        dem = KaneDEM(config.KC_DEM_ROOT)
        naip = naip & dem
        print("naip and dem loaded")

    return naip, kc


def build_dataset(naip, kc, split):
    """
    Randomly split and load data to be the test and train sets

    Input:
        split: the percentage of splitting (entered from args)
    """
    # split the dataset
    train_portion, test_portion = random_bbox_assignment(naip, [split, 1 - split])
    train_dataset = train_portion & kc
    test_dataset = test_portion & kc

    train_sampler = BalancedRandomBatchGeoSampler(
        train_dataset, size=config.PATCH_SIZE, batch_size=config.BATCH_SIZE
    )
    test_sampler = BalancedGridGeoSampler(
        dataset=test_dataset, size=config.PATCH_SIZE, stride=config.PATCH_SIZE
    )

    # create dataloaders (must use batch_sampler)
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_sampler=train_sampler,
        collate_fn=lambda samples: collate_samples(samples, size=config.PATCH_SIZE),
        num_workers=config.NUM_WORKERS,
    )
    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=config.BATCH_SIZE,
        sampler=test_sampler,
        collate_fn=lambda samples: collate_samples(samples, size=config.PATCH_SIZE),
        num_workers=config.NUM_WORKERS,
    )
    logging.info(f"Using {device} device")
    return train_dataloader, test_dataloader


def create_model():
    """
    Setting up training model, loss function and measuring metrics
    """
    # create the model
    model = SegmentationModel(
        model=config.MODEL,
        backbone=config.BACKBONE,
        num_classes=config.NUM_CLASSES,
        weights=config.WEIGHTS,
    ).model.to(device)
    logging.info(model)

    # set the loss function, metrics, and optimizer
    loss_fn_class = getattr(
        importlib.import_module("segmentation_models_pytorch.losses"),
        config.LOSS_FUNCTION,
    )
    # Initialize the loss function with the required parameters
    loss_fn = loss_fn_class(mode="multiclass")

    # IoU metric
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

    return model, loss_fn, train_jaccard, test_jaccard, optimizer


# Define spatial augmentations that apply to both image and mask
spatial_aug = AugmentationSequential(
    K.RandomHorizontalFlip(p=0.5),
    K.RandomVerticalFlip(p=0.5),
    K.RandomRotation(degrees=360, align_corners=True),
    data_keys=["image", "mask"],
    keepdim=True,
)


# Define exclusive image augmentations for color-related effects
def aug_color(bright, contrast):
    """Generate color-related augmentations that apply only to the image."""
    return AugmentationSequential(
        K.ColorJitter(brightness=bright, contrast=contrast),
        data_keys=["image"],
        keepdim=True,
    )


# Add spatial augmentations to the specific color and blur augmentations
def get_aug(aug_type):
    """Select augmentation based on type, with appropriate application to
    image and mask.
    """
    if aug_type == "plasma":
        return spatial_aug + AugmentationSequential(
            K.RandomPlasmaShadow(
                roughness=(0.1, 0.7),
                shade_intensity=(-1.0, 0.0),
                shade_quantity=(0.0, 1.0),
                keepdim=True,
            ),
            data_keys=["image"],
            keepdim=True,
        )
    elif aug_type == "gauss":
        return spatial_aug + AugmentationSequential(
            K.RandomGaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0), p=0.25),
            data_keys=["image"],
            keepdim=True,
        )
    elif aug_type == "all":
        return spatial_aug + AugmentationSequential(
            K.RandomPlasmaShadow(
                roughness=(0.1, 0.7),
                shade_intensity=(-1.0, 0.0),
                shade_quantity=(0.0, 1.0),
                keepdim=True,
            ),
            K.RandomGaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0), p=0.25),
            data_keys=["image"],
            keepdim=True,
        )
    elif aug_type == "color":
        return spatial_aug + aug_color(
            config.COLOR_BRIGHT, config.COLOR_CONTRST
        )
    elif aug_type == "blur":
        return spatial_aug + AugmentationSequential(
            K.RandomBoxBlur(keepdim=True),
            data_keys=["image"],
            keepdim=True,
        )
    else:
        return spatial_aug


def copy_first_entry(a_list: list) -> list:
    """
    Copies the first entry in a list and appends it to the end.

    Parameters
    ----------
    a_list : list
        The list to modify

    Returns
    -------
    list
        The modified list
    """
    first_entry = a_list[0]

    # Append the copy to the end
    a_list.append(first_entry)

    return a_list


def normalize_func(model):
    mean = config.DATASET_MEAN
    std = config.DATASET_STD
    # add copies of first entry to DATASET_MEAN and DATASET_STD
    # to match data in_channels
    if len(mean) != model.in_channels:
        for _ in range(model.in_channels - len(mean)):
            mean = copy_first_entry(mean)
            std = copy_first_entry(std)

    scale_mean = torch.tensor(0.0)
    scale_std = torch.tensor(255.0)
    normalize = K.Normalize(mean=mean, std=std)
    scale = K.Normalize(mean=scale_mean, std=scale_std)
    return normalize, scale


def add_extra_channel(
    image_tensor: torch.Tensor, source_channel: int = 0
) -> torch.Tensor:
    """
    Adds an additional channel to an image by copying an existing channel.

    Parameters
    ----------
    image_tensor : torch.Tensor
        A tensor containing image data. Expected shape is
        (batch, channels, h, w)

    source_channel : int
        The index of the channel to be copied

    Returns
    -------
    torch.Tensor
        A modified tensor with added channels
    """
    # Select the source channel to duplicate
    original_channel = image_tensor[:, source_channel : source_channel + 1, :, :]

    # Generate copy of selected channel
    extra_channel = original_channel.clone()

    # Concatenate the extra channel to the original image along the second
    # dimension (channel dimension)
    augmented_tensor = torch.cat((image_tensor, extra_channel), dim=1)

    return augmented_tensor


def train_setup(
    sample: DefaultDict[str, Any],
    epoch: int,
    batch: int,
    aug_type: str,
    train_images_root,
    model,
    config=config,
) -> Tuple[torch.Tensor]:
    """
    Sets up for the training step by sending images and masks to device,
    applying augmentations, and saving training sample images.

    Parameters
    ----------
    sample : DefaultDict[str, Any]
        A dataloader sample containing image, mask, and bbox data

    epoch : int
        The current epoch

    batch : int
        The current batch

    Returns
    -------
    Tuple[torch.Tensor]
        Augmented image and mask tensors to be used in the train step
    """

    samp_image = sample["image"]
    samp_mask = sample["mask"]
    normalize, scale = normalize_func(model)
    # add extra channel(s) to the images and masks
    if samp_image.size(1) != model.in_channels:
        for _ in range(model.in_channels - samp_image.size(1)):
            samp_image = add_extra_channel(samp_image)

    # send img and mask to device; convert y to float tensor for augmentation
    X = samp_image.to(device)
    y = samp_mask.type(torch.float32).to(device)

    # scale both img and mask to range of [0, 1] (req'd for augmentations)
    X = scale(X)

    # augment img and mask with same augmentations
    aug = get_aug(aug_type)

    X_aug, y_aug = aug(X, y)

    # denormalize mask to reset to index tensor (req'd for loss func)
    y = y_aug.type(torch.int64)

    # remove channel dim from y (req'd for loss func)
    y_squeezed = y[:, :, :].squeeze()

    # plot first batch
    if batch == 0:
        save_dir = os.path.join(
            train_images_root,
            f"-{(config.COLOR_BRIGHT), config.COLOR_CONTRST}-epoch-{epoch}",
        )
        try:
            os.mkdir(save_dir)

        except FileExistsError:
            # Directory already exists, remove it recursively
            shutil.rmtree(save_dir)
            os.mkdir(save_dir)

        for i in range(config.BATCH_SIZE):
            plot_tensors = {
                "image": X[i].cpu(),
                "mask": samp_mask[i],
                "augmented_image": X_aug[i].cpu(),
                "augmented_mask": y[i].cpu(),
            }
            sample_fname = os.path.join(save_dir, f"train_sample-{epoch}.{i}.png")
            plot_from_tensors(
                plot_tensors,
                sample_fname,
                "grid",
                kc.colors,
                kc.labels_inverse,
                sample["bbox"][i],
            )

    return (
        normalize(X_aug),
        y_squeezed,
    )


def train_epoch(
    dataloader: DataLoader,
    model: Module,
    loss_fn: Module,
    jaccard: Metric,
    optimizer: Optimizer,
    epoch: int,
    train_images_root,
    writer,
) -> None:
    """
    Executes a training step for the model

    Parameters
    ----------
    dataloder : DataLoader
        Dataloader for the training data

    model : Module
        A PyTorch model

    loss_fn : Module
        A PyTorch loss function

    jaccard : Metric
        The metric to be used for evaluation, specifically the Jaccard Index

    optimizer : Optimizer
        The optimizer to be used for training

    epoch : int
        The current epoch
    """
    num_batches = len(dataloader)
    model.train()
    jaccard.reset()
    train_loss = 0
    for batch, sample in enumerate(dataloader):
        X, y = train_setup(sample, epoch, batch, aug_type, train_images_root, model)

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
            logging.info(f"loss: {loss:>7f}  [{current:>5d}/{num_batches:>5d}]")
    train_loss /= num_batches
    final_jaccard = jaccard.compute()

    # Need to rename scalars?
    writer.add_scalar("loss/train", train_loss, epoch)
    writer.add_scalar("IoU/train", final_jaccard, epoch)
    logging.info(f"Train Jaccard index: {final_jaccard}:.4f")
    return final_jaccard


def test(
    dataloader: DataLoader,
    model: Module,
    loss_fn: Module,
    jaccard: Metric,
    epoch: int,
    plateau_count: int,
    test_image_root,
    writer,
) -> float:
    """
    Executes a testing step for the model and saves sample output images

    Parameters
    ----------
    dataloder : DataLoader
        Dataloader for the testing data

    model : Module
        A PyTorch model

    loss_fn : Module
        A PyTorch loss function

    jaccard : Metric
        The metric to be used for evaluation, specifically the Jaccard Index

    epoch : int
        The current epoch

    plateau_count : int
        The number of epochs the loss has been plateauing

    Returns
    -------
    float
        The test loss for the epoch
    """
    num_batches = len(dataloader)
    model.eval()
    jaccard.reset()
    test_loss = 0
    with torch.no_grad():
        for batch, sample in enumerate(dataloader):
            samp_image = sample["image"]
            samp_mask = sample["mask"]
            # add an extra channel to the images and masks
            if samp_image.size(1) != model.in_channels:
                for _i in range(model.in_channels - samp_image.size(1)):
                    samp_image = add_extra_channel(samp_image)
                    samp_mask = add_extra_channel(samp_mask)
            X = samp_image.to(device)
            normalize, scale = normalize_func(model)
            X_scaled = scale(X)
            X = normalize(X_scaled)
            y = samp_mask.to(device)
            y_squeezed = y[:, :, :].squeeze()

            # compute prediction error
            outputs = model(X)
            loss = loss_fn(outputs, y_squeezed)

            # update metric
            preds = outputs.argmax(dim=1)
            jaccard.update(preds, y_squeezed)

            # add test loss to rolling total
            test_loss += loss.item()

            # plot first batch
            if batch == 0 or (plateau_count == config.PATIENCE - 1 and batch < 10):
                save_dir = os.path.join(test_image_root, f"epoch-{epoch}")
                if not os.path.exists(save_dir):
                    os.mkdir(save_dir)
                for i in range(config.BATCH_SIZE):
                    plot_tensors = {
                        "image": X_scaled[i].cpu(),
                        "ground_truth": samp_mask[i],
                        "prediction": preds[i].cpu(),
                    }
                    sample_fname = os.path.join(
                        save_dir, f"test_sample-{epoch}.{batch}.{i}.png"
                    )
                    plot_from_tensors(
                        plot_tensors,
                        sample_fname,
                        "row",
                        kc.colors,
                        kc.labels_inverse,
                        sample["bbox"][i],
                    )
    test_loss /= num_batches
    final_jaccard = jaccard.compute()
    writer.add_scalar("loss/test", test_loss, epoch)
    writer.add_scalar("IoU/test", final_jaccard, epoch)
    logging.info(
        f"\nTest error: \n Jaccard index: {final_jaccard:>4f}, "
        + f"Test avg loss: {test_loss:>4f} \n"
    )

    # Now returns test_loss such that it can be compared against previous losses
    return test_loss, final_jaccard


def train(
    writer,
    train_dataloader,
    model,
    test_jaccard,
    out_root,
    loss_fn,
    train_jaccard,
    optimizer,
    test_dataloader,
    test_image_root,
    train_images_root,
    config=config,
):
    # How much the loss needs to drop to reset a plateau
    threshold = config.THRESHOLD

    # How many epochs loss needs to plateau before terminating
    patience = config.PATIENCE

    # Beginning loss
    best_loss = None

    # How long it's been plateauing
    plateau_count = 0

    for t in range(config.EPOCHS):
        if t == 0:
            test_loss, t_jaccard = test(
                test_dataloader,
                model,
                loss_fn,
                test_jaccard,
                t + 1,
                plateau_count,
                test_image_root,
                writer,
            )
            print(f"untrained loss {test_loss:.3f}, jaccard {t_jaccard:.3f}")

        logging.info(f"Epoch {t + 1}\n-------------------------------")
        epoch_jaccard = train_epoch(
            train_dataloader,
            model,
            loss_fn,
            train_jaccard,
            optimizer,
            t + 1,
            train_images_root,
            writer,
        )

        test_loss, t_jaccard = test(
            test_dataloader,
            model,
            loss_fn,
            test_jaccard,
            t + 1,
            plateau_count,
            test_image_root,
            writer,
        )
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
                logging.info(
                    f"Loss Plateau: {t} epochs, reached patience of {patience}"
                )
                break

    print("Done!")
    writer.close()

    torch.save(model.state_dict(), os.path.join(out_root, "model.pth"))
    logging.info(f"Saved PyTorch Model State to {out_root}")

    return epoch_jaccard, t_jaccard


def run_trials():
    """
    Running training for multiple trials
    """

    if wandb_tune:
        run = wandb.init(project="cmap_train")
        vars(args).update(run.config)
        print("wandb taken over config")

    train_ious = []
    test_ious = []
    for num in range(num_trials):
        train_images_root, test_image_root, out_root, writer = writer_prep(
            exp_name, num
        )
        # randomly splitting the data at every trial
        train_dataloader, test_dataloader = build_dataset(naip, kc, split)
        model, loss_fn, train_jaccard, test_jaccard, optimizer = create_model()
        logging.info(f"Trial {num + 1}\n====================================")
        train_iou, test_iou = train(
            writer,
            train_dataloader,
            model,
            test_jaccard,
            out_root,
            loss_fn,
            train_jaccard,
            optimizer,
            test_dataloader,
            test_image_root,
            train_images_root,
        )

        train_ious.append(float(train_iou))
        test_ious.append(float(test_iou))

    test_average = mean(test_ious)
    train_average = mean(train_ious)
    test_std = stdev(test_ious)
    train_std = stdev(train_ious)

    print(
        f"Training: average: {train_average:.3f}, standard deviation: {train_std:.3f}"
    )
    print(f"Test: mean: {test_average:.3f}, standard deviation:{test_std:.3f}")

    if wandb_tune:
        run.log({"average_test_jaccard_index": test_average})
        wandb.finish()


# executing
exp_name, aug_type, split, wandb_tune, num_trials = arg_parsing()
naip, kc = initialize_dataset()

if wandb_tune:
    with open("configs/sweep_config.yml", "r") as file:
        sweep_config = yaml.safe_load(file)
    sweep_id = wandb.sweep(sweep_config, project="cmap_train")
    wandb.agent(sweep_id, run_trials, count=10)

else:
    run_trials()
