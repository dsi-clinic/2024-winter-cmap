#!/usr/bin/env python
"""train_sam.py — DDP fine-tuning of SAM mask decoder on Kane County via TorchGeo"""

import argparse
import csv
import logging
import os
import secrets  # for cryptographically secure random numbers
import sys
import time  # For timing
import random # For seeding (though not used for splitting in this version)
from collections import defaultdict
from pathlib import Path

import einops  # for tensor manipulation
import matplotlib # Keep this separate from pyplot if you set backend first
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Configure Matplotlib for headless environments first
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# === TORCHGEO IMPORTS START ===
from torchgeo.datasets import NAIP, stack_samples, BoundingBox # random_bbox_assignment, GeoDataset removed as not used for splitting here
from torchgeo.samplers import RandomGeoSampler, Units
# === TORCHGEO IMPORTS END ===

# === TORCHMETRICS IMPORT START ===
try:
    import torchmetrics
except ImportError:
    print("ERROR: torchmetrics not found. Please install it: pip install torchmetrics", flush=True)
    sys.exit(1)
# === TORCHMETRICS IMPORT END ===


# --- Path Setup for Local Module Imports & SAM ---
_initial_logger = logging.getLogger("train_sam_setup")
_initial_logger.setLevel(logging.INFO)
_initial_stream_handler = logging.StreamHandler(sys.stdout)
_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
_initial_stream_handler.setFormatter(_formatter)
_initial_logger.addHandler(_initial_stream_handler)
_initial_logger.propagate = False

_initial_logger.info("Setting up sys.path for local modules and SAM library...")
repo_root_for_project = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root_for_project))
_initial_logger.info(f"Added project root to sys.path: {repo_root_for_project}")

sam_source_dir_name = "segment_anything_source_code"
sam_source_path = repo_root_for_project / sam_source_dir_name
if sam_source_path.is_dir():
    sys.path.append(str(sam_source_path))
    _initial_logger.info(f"Appended SAM source path to sys.path: {sam_source_path}")
else:
    _initial_logger.warning(f"SAM source directory NOT FOUND at: {sam_source_path}. SAM imports might fail.")

try:
    from data.kc import KaneCounty
    from prompted_kc import PromptedKaneCounty
    from segment_anything.build_sam import sam_model_registry
    from segment_anything.predictor import SamPredictor
    _initial_logger.info("Successfully imported local and SAM modules.")
except ImportError as e_import:
    _initial_logger.error(f"Failed to import necessary modules after path setup: {e_import}")
    sys.exit(1)
# --- End Path Setup ---

# Constants
PLOT_EVERY_N_BATCHES = 100
UPDATE_TENSORBOARD_EVERY_N_STEPS = 10
RGBA_CHANNELS = 4
SIGMOID_THRESHOLD = 0.5


def parse_args():
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description="DDP fine-tune SAM on KC with TorchGeo")
    p.add_argument("--checkpoint", required=True, help="Path to SAM checkpoint .pth")
    p.add_argument("--output-dir", required=True, help="Directory to save outputs")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--accum-steps", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--naip-root", required=True, help="Root directory of NAIP imagery")
    p.add_argument("--shape-path", required=True, help="Path to KC GDB directory/zip")
    p.add_argument("--layer-name", default="Basins", help="Layer name or index in GDB")
    p.add_argument("--chip-size", type=int, default=512, help="Patch size in pixels")
    # --split-rate and --seed removed as we are not splitting for now
    p.add_argument("--local_rank", type=int, default=int(os.environ.get("LOCAL_RANK", -1)))
    return p.parse_args()

# Helper function for IoU using torchmetrics
def calculate_binary_iou(preds: torch.Tensor, target: torch.Tensor) -> float:
    """Calculates binary IoU using torchmetrics, ensuring tensors are on CPU for calculation."""
    preds_cpu = preds.cpu()
    target_cpu = target.cpu()
    try:
        iou_val = torchmetrics.functional.jaccard_index(preds_cpu, target_cpu.bool(), task="binary")
        return iou_val.item()
    except Exception: 
        if preds_cpu.sum() == 0 and target_cpu.sum() == 0:
            return 1.0
        else:
            return 0.0

# main() function modified to remove train/val split and eval_epoch_fn call
def main():
    initial_log_messages = [] 
    initial_log_messages.append("main() started.")
    args = parse_args()
    initial_log_messages.append(f"Arguments parsed: {args}")

    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", str(10000 + secrets.randbelow(10000)))
    
    is_ddp_env = "LOCAL_RANK" in os.environ and os.environ.get("WORLD_SIZE", "1") != "0" 
    if is_ddp_env:
        args.local_rank = int(os.environ["LOCAL_RANK"])
        if not dist.is_initialized():
            dist.init_process_group("nccl", init_method="env://")
        torch.cuda.set_device(args.local_rank)
        rank = dist.get_rank()
        world = dist.get_world_size()
        is_ddp = True 
        initial_log_messages.append(f"DDP Initialized. Rank: {rank}, World Size: {world}.")
    else:
        rank = 0
        world = 1
        args.local_rank = 0 
        is_ddp = False
        initial_log_messages.append("Running in non-DDP mode. Assuming rank 0, world 1.")
    
    device = torch.device(f"cuda:{args.local_rank}" if torch.cuda.is_available() else "cpu")
    initial_log_messages.append(f"Using device: {device}")

    output_dir = Path(args.output_dir).resolve()
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
    if world > 1: dist.barrier()

    model_output_path = output_dir / "fine_tuned_sam.pth"
    csv_train_output_path = output_dir / "per_class_ious_train.csv" # Only train CSV now
    log_file_path = output_dir / f"training_rank{rank}.log"
    debug_images_dir = output_dir / "debug_images"
    tensorboard_log_dir = output_dir / "tensorboard_logs" / f"sam_kc_exp_{output_dir.name}_rank{rank}"

    log_level_str = os.environ.get("LOGLEVEL", "INFO").upper()
    log_level = getattr(logging, log_level_str, logging.INFO)
    
    for handler in logging.root.handlers[:]: logging.root.removeHandler(handler)
    if _initial_logger.hasHandlers(): _initial_logger.removeHandler(_initial_stream_handler)

    handlers_list = [logging.FileHandler(log_file_path, mode='a')]
    if rank == 0:
        debug_images_dir.mkdir(parents=True, exist_ok=True)
        tensorboard_log_dir.mkdir(parents=True, exist_ok=True)
        handlers_list.append(logging.StreamHandler(sys.stdout))
    
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s RANK %(world_rank)s - %(levelname)s - [%(name)s.%(funcName)s:%(lineno)d] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers_list)
    
    old_factory = logging.getLogRecordFactory()
    def record_factory(*args_factory, **kwargs_factory):
        record = old_factory(*args_factory, **kwargs_factory)
        record.world_rank = rank; return record
    logging.setLogRecordFactory(record_factory)
    
    logger = logging.getLogger(__name__)
    for msg in initial_log_messages: logger.debug(msg)
    logger.info(f"Full logging configured. Output directory: {output_dir}")
    logger.info(f"Script arguments: {args}")
    
    libraries_to_quiet = ["rasterio", "fiona", "matplotlib", "PIL", "torchgeo"] 
    for lib_name in libraries_to_quiet: logging.getLogger(lib_name).setLevel(logging.WARNING)

    writer = SummaryWriter(log_dir=str(tensorboard_log_dir)) if rank == 0 else None
    logger.debug(f"SummaryWriter setup. Debug images dir: {debug_images_dir}")

    logger.info(f"Loading SAM model from checkpoint: {args.checkpoint}...")
    sam_model = sam_model_registry["vit_h"](checkpoint=str(args.checkpoint))
    sam_model.to(device)
    logger.info("SAM model loaded to device.")
    logger.debug("Freezing SAM image encoder and prompt encoder parameters...")
    for p in sam_model.image_encoder.parameters(): p.requires_grad = False
    for p in sam_model.prompt_encoder.parameters(): p.requires_grad = False
    sam_model.image_encoder.eval(); sam_model.prompt_encoder.eval() 
    logger.debug("SAM parameters frozen and core encoders set to eval mode.")
    
    model_to_access = sam_model
    if is_ddp:
        logger.debug("Wrapping SAM model with DDP...")
        sam_model_ddp = DDP(sam_model, device_ids=[args.local_rank], find_unused_parameters=False)
        logger.debug("SAM model wrapped with DDP.")
        model_to_access = sam_model_ddp.module
    else:
        sam_model_ddp = sam_model 

    logger.info(f"Initializing NAIP dataset from naip-root: {args.naip_root}...")
    naip_ds = NAIP(args.naip_root) # Full NAIP dataset
    logger.info(f"NAIP dataset initialized. CRS: {naip_ds.crs}, Resolution: {naip_ds.res}")

    logger.debug("Preparing idmap for KaneCounty labels...")
    idmap = {name: id_val for id_val, name in KaneCounty.all_labels.items()}
    logger.debug(f"idmap created (string keys to int values): {idmap}")

    logger.info("Initializing PromptedKaneCounty dataset ...") # Simplified message
    layer_identifier_for_kc = args.layer_name
    try: layer_identifier_for_kc = int(args.layer_name)
    except ValueError: pass
        
    label_ds_configs = (layer_identifier_for_kc, idmap, args.chip_size, naip_ds.crs, naip_ds.res)
    
    start_time_label_ds = time.time()
    label_dataset = PromptedKaneCounty(args.shape_path, label_ds_configs) # Full label dataset
    end_time_label_ds = time.time()
    elapsed_time_label_ds = end_time_label_ds - start_time_label_ds
    logger.info(f"Full PromptedKaneCounty dataset initialization FINISHED. Took {elapsed_time_label_ds:.2f} seconds.")

    logger.debug("Combining NAIP and label datasets for training...")
    train_dataset = naip_ds & label_dataset # Use the full combined dataset for training
    train_dataset_len = len(train_dataset)
    logger.info(f"Training dataset (full combined) length: {train_dataset_len}")

    if train_dataset_len == 0:
        logger.error("Training dataset is empty! Check data inputs and intersection logic."); sys.exit(1)
    
    logger.debug("Initializing RandomGeoSampler for training...")
    train_sampler_length_per_rank = max(1, train_dataset_len // world) 
    train_sampler = RandomGeoSampler(train_dataset, size=args.chip_size, length=train_sampler_length_per_rank, units=Units.PIXELS)
    logger.debug("Training RandomGeoSampler initialized.")

    logger.debug(f"Initializing Training DataLoader with batch_size={args.batch_size}, num_workers={args.num_workers}...")
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler, num_workers=args.num_workers, collate_fn=stack_samples, pin_memory=True)
    logger.info(f"Training DataLoader initialized. Effective length per rank: {len(train_dataloader)}")
    
    logger.debug("Setting up optimizer and GradScaler...")
    optimizer = torch.optim.AdamW(model_to_access.mask_decoder.parameters(), lr=args.lr)
    scaler = torch.cuda.amp.GradScaler()
    logger.debug("Optimizer and GradScaler setup complete.")

    final_train_ious_accumulator = defaultdict(list)
    logger.debug("Initialized final_train_ious_accumulator.")

    total_optimizer_steps_across_epochs = 0

    logger.info("=== SETUP COMPLETE === Starting training loop... ===")
    for epoch in range(args.epochs):
        # train_epoch_fn is now integrated back into this loop
        logger.info(f"Epoch {epoch+1}/{args.epochs} - Starting Training...")
        if hasattr(train_sampler, 'set_epoch') and is_ddp:
            train_sampler.set_epoch(epoch)
        
        model_to_access.mask_decoder.train()
        if hasattr(model_to_access, 'image_encoder'): model_to_access.image_encoder.eval()
        if hasattr(model_to_access, 'prompt_encoder'): model_to_access.prompt_encoder.eval()

        epoch_loss_sum_scaled = 0.0
        epoch_iou_sum_train = 0.0
        epoch_items_processed_train = 0
        epoch_optimizer_steps = 0
        
        pbar_train = None
        if rank == 0:
            pbar_train = tqdm(total=len(train_dataloader), desc=f"Epoch {epoch+1} [Train]", unit="batch", leave=False)

        for batch_idx, batch in enumerate(train_dataloader):
            imgs = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)
            points = batch["point"]
            current_batch_size_actual = imgs.size(0)

            micro_batch_loss_accumulator = 0.0
            
            with torch.cuda.amp.autocast(enabled=device.type == 'cuda'):
                for b_item_idx in range(current_batch_size_actual):
                    img_tensor_cpu = imgs[b_item_idx].cpu()
                    if img_tensor_cpu.shape[0] == RGBA_CHANNELS:
                        rgb_tensor_cpu = img_tensor_cpu[:3, :, :]
                    else:
                        rgb_tensor_cpu = img_tensor_cpu
                    img_np_rgb = einops.rearrange(rgb_tensor_cpu, "c h w -> h w c").numpy().astype(np.uint8)

                    predictor = SamPredictor(model_to_access)
                    predictor.set_image(img_np_rgb)
                    img_embeddings = predictor.get_image_embedding().to(device)
                    
                    current_point_np = points[b_item_idx]
                    pt_tensor = torch.from_numpy(current_point_np).to(device).unsqueeze(0).unsqueeze(0).float()
                    lbl_tensor = torch.ones((1, 1), device=device)
                    sparse_prompt_embeddings, dense_prompt_embeddings = model_to_access.prompt_encoder(
                        (pt_tensor, lbl_tensor), None, None)
                    
                    low_res_masks, _ = model_to_access.mask_decoder(
                        image_embeddings=img_embeddings,
                        image_pe=model_to_access.prompt_encoder.get_dense_pe().to(device),
                        sparse_prompt_embeddings=sparse_prompt_embeddings,
                        dense_prompt_embeddings=dense_prompt_embeddings,
                        multimask_output=False)
                    
                    target_mask_shape = masks[b_item_idx].shape[-2:]
                    upscaled_masks = torch.nn.functional.interpolate(
                        low_res_masks, size=target_mask_shape, mode="bilinear", align_corners=False)
                    
                    current_gt_mask_item = masks[b_item_idx]
                    if current_gt_mask_item.ndim == 3 and current_gt_mask_item.shape[0] == 1:
                        current_gt_mask_item = current_gt_mask_item.squeeze(0)
                    gt_binary_for_loss = (current_gt_mask_item > 0).float().unsqueeze(0).unsqueeze(0)

                    loss = torch.nn.functional.binary_cross_entropy_with_logits(
                        upscaled_masks, gt_binary_for_loss)
                    micro_batch_loss_accumulator += loss

                    pred_binary_mask_tensor = (torch.sigmoid(upscaled_masks) > SIGMOID_THRESHOLD).int().squeeze()
                    gt_mask_for_iou_multiclass = current_gt_mask_item.int()
                    
                    prompt_y, prompt_x = current_point_np[1], current_point_np[0]
                    cls_at_prompt = int(gt_mask_for_iou_multiclass[prompt_y, prompt_x].item())
                    
                    if cls_at_prompt > 0:
                        gt_binary_for_iou_cls = (gt_mask_for_iou_multiclass == cls_at_prompt).int()
                        iou = calculate_binary_iou(pred_binary_mask_tensor, gt_binary_for_iou_cls)
                        final_train_ious_accumulator[cls_at_prompt].append(iou)
                        epoch_iou_sum_train += iou
                        epoch_items_processed_train += 1

                    if rank == 0 and b_item_idx == 0 and batch_idx % PLOT_EVERY_N_BATCHES == 0:
                        logger.debug(f"Generating TRAIN debug plot for Epoch {epoch+1}, Batch {batch_idx+1}")
                        # ... (plotting code as before) ...
                        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
                        axes[0].imshow(img_np_rgb); axes[0].set_title(f"Input (Ep{epoch+1} Bt{batch_idx+1})"); axes[0].axis("off")
                        true_mask_display = gt_mask_for_iou_multiclass.cpu().numpy()
                        vmax_val = max(KaneCounty.all_labels.keys()) if KaneCounty.all_labels else 15
                        axes[1].imshow(true_mask_display, cmap="viridis", vmin=0, vmax=vmax_val); axes[1].set_title("True Mask"); axes[1].axis("off")
                        pred_mask_display = pred_binary_mask_tensor.cpu().numpy()
                        axes[2].imshow(pred_mask_display, cmap="gray"); axes[2].set_title("Pred Mask"); axes[2].axis("off")
                        plt.tight_layout()
                        plot_filename = debug_images_dir / f"train_epoch{epoch+1}_batch{batch_idx+1}.png"
                        try: plt.savefig(plot_filename); logger.debug(f"Saved train plot: {plot_filename}")
                        except Exception as e: logger.error(f"Error saving train plot {plot_filename}: {e}")
                        plt.close(fig)

            loss_to_scale = micro_batch_loss_accumulator / args.accum_steps
            scaler.scale(loss_to_scale).backward()
            epoch_loss_sum_scaled += loss_to_scale.item()

            if (batch_idx + 1) % args.accum_steps == 0:
                scaler.step(optimizer); scaler.update(); optimizer.zero_grad()
                epoch_optimizer_steps += 1
                total_optimizer_steps_across_epochs +=1

                if rank == 0:
                    avg_loss_disp = epoch_loss_sum_scaled / epoch_optimizer_steps if epoch_optimizer_steps > 0 else 0.0
                    avg_iou_disp = epoch_iou_sum_train / epoch_items_processed_train if epoch_items_processed_train > 0 else 0.0
                    if pbar_train: pbar_train.set_postfix(loss=f"{avg_loss_disp:.4f}", iou=f"{avg_iou_disp:.4f}", refresh=False)
                    
                    if writer and total_optimizer_steps_across_epochs % UPDATE_TENSORBOARD_EVERY_N_STEPS == 0:
                        writer.add_scalar("stepwise/train_loss", avg_loss_disp, total_optimizer_steps_across_epochs)
                        writer.add_scalar("stepwise/train_IoU", avg_iou_disp, total_optimizer_steps_across_epochs)
            
            if rank == 0 and pbar_train: pbar_train.update(1)

        if rank == 0 and pbar_train: pbar_train.close()
        
        avg_epoch_train_loss = epoch_loss_sum_scaled / epoch_optimizer_steps if epoch_optimizer_steps > 0 else float('nan')
        avg_epoch_train_iou = epoch_iou_sum_train / epoch_items_processed_train if epoch_items_processed_train > 0 else 0.0
        
        if rank == 0:
            logger.info(f"*** Epoch {epoch+1} TRAIN Summary: Avg Loss {avg_epoch_train_loss:.4f}, Avg IoU {avg_epoch_train_iou:.4f} over {epoch_items_processed_train} items ***")
            if writer:
                writer.add_scalar("EpochSummary/train_loss", avg_epoch_train_loss, epoch)
                writer.add_scalar("EpochSummary/train_IoU", avg_epoch_train_iou, epoch)
        
        # --- Placeholder Comment for Evaluation Loop ---
        if rank == 0:
             logger.info(f"--- End of Epoch {epoch+1} ---")
             logger.info(">>> TODO: Implement evaluation loop here using a validation dataset. <<<")
             logger.info(">>> This script currently only trains and reports metrics on the training data. <<<")
        
        if world > 1: dist.barrier()
        logger.info(f"Epoch {epoch+1}/{args.epochs} - Cycle Finished.")


    logger.info("Training loop finished. Finalizing...")
    if rank == 0:
        final_model_path = output_dir / f"final_fine_tuned_sam_epochs{args.epochs}.pth"
        logger.info(f"Rank 0 saving final model to {final_model_path}...")
        torch.save(model_to_access.state_dict(), final_model_path)
        
        logger.info(f"Saving training per-class IoUs to {csv_train_output_path}...")
        with csv_train_output_path.open("w", newline="") as f:
            csv_w = csv.writer(f)
            csv_w.writerow(["class_id", "mean_iou", "std_iou", "count"])
            for cls_id_val, ious_list in final_train_ious_accumulator.items():
                if ious_list: csv_w.writerow([cls_id_val, np.mean(ious_list), np.std(ious_list), len(ious_list)])
                else: csv_w.writerow([cls_id_val, 0.0, 0.0, 0])
        
        if writer: writer.close()
        logger.info("Rank 0 final model and training CSV saved. Writer closed.")

    if world > 1: dist.barrier()
    if rank == 0: logger.info("All operations complete. Destroying DDP process group (if applicable)...")
    if is_ddp and dist.is_initialized():
        dist.destroy_process_group()
        if rank == 0: logger.info("DDP process group destroyed.")
    
    if rank == 0: logger.info(f"Script {Path(__file__).name} finished successfully.")


if __name__ == "__main__":
    print(f"INFO: Script {Path(__file__).name} execution started (__name__ == '__main__'). RANK: {os.environ.get('LOCAL_RANK', 'N/A')}", flush=True)
    main()