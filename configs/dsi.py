import os

# data paths
DATA_ROOT = "/net/projects/cmap/data"
KC_SHAPE_ROOT = os.path.join(DATA_ROOT, "kane-county-data")
KC_IMAGE_ROOT = os.path.join(DATA_ROOT, "KC-images")
KC_MASK_ROOT = os.path.join(DATA_ROOT, "KC-masks/top-structures-masks")
OUTPUT_ROOT = "/net/projects/cmap/model-outputs"

# model selection
MODEL = "unet"
BACKBONE = "resnet50"
WEIGHTS = "ResNet50_Weights.LANDSAT_TM_TOA_MOCO"

# model hyperparams
DATASET_MEAN = [
    0.3281668683529412,
    0.4208941459215686,
    0.4187784871764706,
    0.5470313711372549,
	0.3281668683529412,
	0.3281668683529412,
	0.3281668683529412
]
DATASET_STD = [
    0.030595504117647058,
    0.02581302749019608,
    0.025523325960784313,
    0.03643713776470588,
	0.030595504117647058,
	0.030595504117647058,
	0.030595504117647058
]
BATCH_SIZE = 16
PATCH_SIZE = 512
NUM_CLASSES = 5  # predicting 4 classes + background
LR = 1e-3
NUM_WORKERS = 8
EPOCHS = 6
IGNORE_INDEX = 0  # index in images to ignore for jaccard index
LOSS_FUNCTION = "JaccardLoss"  # JaccardLoss, DiceLoss, TverskyLoss, LovaszLoss
