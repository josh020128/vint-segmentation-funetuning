# FILE: vint_train/training/train_eval_loop_seg.py

"""
Training and evaluation loop for the late-fusion SegmentationViNT.
"""
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
import wandb
from typing import Dict
import os
import numpy as np
import shutil

# --- Main project imports ---
from vint_train.training.seg_losses import SegmentationNavigationLoss
from vint_train.training.seg_metrics import evaluate_segmentation_vint
from vint_train.visualizing.visualize_segmentation import visualize_segmentation_predictions

# --- Helper Functions ---

def get_model_module(model: nn.Module) -> nn.Module:
    """Get the actual model module (handles DataParallel wrapper)."""
    return model.module if hasattr(model, 'module') else model

def save_checkpoint(model, optimizer, scheduler, epoch, project_folder, is_best=False):
    """Saves latest and best model checkpoints."""
    if not os.path.exists(project_folder):
        os.makedirs(project_folder)
    model_state = get_model_module(model).state_dict()
    checkpoint = {
        'epoch': epoch, 'model_state_dict': model_state,
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
    }
    latest_path = os.path.join(project_folder, 'latest.pth')
    torch.save(checkpoint, latest_path)
    if is_best:
        best_path = os.path.join(project_folder, 'best.pth')
        shutil.copyfile(latest_path, best_path)
        print(f"Epoch {epoch}: New best model saved!")

def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Helper to convert a tensor to a numpy array."""
    return tensor.detach().cpu().numpy()

def visualize_batch(model, batch, device, epoch, stage, project_folder, use_wandb=True):
    """Generates and logs a visualization for a batch of data."""
    model.eval() # Set to eval mode for consistent predictions
    with torch.no_grad():
        obs_images = batch['obs_images'][:8].to(device)
        goal_images = batch['goal_images'][:8].to(device)
        
        outputs = model(obs_images, goal_images)
        
        visualize_segmentation_predictions(
            batch_obs_images=to_numpy(obs_images),
            batch_goal_images=to_numpy(goal_images),
            batch_seg_preds=to_numpy(outputs['obs_seg_logits']),
            batch_seg_labels=to_numpy(batch.get('obs_seg_mask', torch.zeros_like(outputs['obs_seg_logits']))[:8]),
            batch_pred_waypoints=to_numpy(outputs['action_pred']),
            batch_label_waypoints=to_numpy(batch['actions'][:8]),
            batch_goals=to_numpy(batch.get('goal_pos', torch.zeros(8, 2))[:8]),
            save_folder=project_folder,
            epoch=epoch,
            eval_type=f"stage_{stage}",
            use_wandb=use_wandb,
        )
    model.train() # Set back to train mode

# --- Main Training Loop ---

def train_eval_loop_segmentation(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    test_dataloaders: Dict[str, torch.utils.data.DataLoader],
    epochs: int,
    device: torch.device,
    project_folder: str,
    config: Dict,
    start_stage: int = 1,
):
    """3-stage FuSe training for the late-fusion SegmentationViNT."""
    total_epochs_trained = 0
    model_module = get_model_module(model)
    
    lr_config = config["lr_schedule"]
    stage1_epochs = config.get("stage1_epochs", 30)
    stage2_epochs = config.get("stage2_epochs", 40)
    
    loss_fn = SegmentationNavigationLoss(use_uncertainty_weighting=True)
    scaler = GradScaler() if device.type == 'cuda' else None
    
    best_val_score = float('inf')

    # --- STAGE 1: Train new layers only ---
    if start_stage <= 1:
        print(f"\n{'='*60}\nSTAGE 1: Training New Modules (ViNT Frozen)\n{'='*60}")
        trainable_params = list(model_module.seg_model.parameters()) + \
                           list(model_module.seg_feature_extractor.parameters()) + \
                           list(model_module.action_predictor.parameters()) + \
                           list(model_module.dist_predictor.parameters())
        stage1_optimizer = torch.optim.AdamW(trainable_params, lr=lr_config["stage1"])
        stage1_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(stage1_optimizer, T_max=stage1_epochs)
        
        for epoch in range(stage1_epochs):
            print(f"\n--- Stage 1, Epoch {epoch+1}/{stage1_epochs} ---")
            train_epoch(model, dataloader, stage1_optimizer, loss_fn, device, scaler, config, total_epochs_trained)
            stage1_scheduler.step()
            if (epoch + 1) % 5 == 0:
                evaluate_and_log(model, test_dataloaders, device, total_epochs_trained, config, stage=1)
                visualize_batch(model, next(iter(dataloader)), device, total_epochs_trained, 1, project_folder)
            total_epochs_trained += 1
        save_checkpoint(model, stage1_optimizer, stage1_scheduler, total_epochs_trained, project_folder)

    # --- STAGE 2: Unfreeze and train with different LRs ---
    if start_stage <= 2:
        print(f"\n{'='*60}\nSTAGE 2: Unfreezing ViNT (Discriminative LRs)\n{'='*60}")
        model_module.unfreeze_vint()
        
        param_groups = [
            {'params': list(model_module.seg_model.parameters()) + 
                       list(model_module.seg_feature_extractor.parameters()) + 
                       list(model_module.action_predictor.parameters()) + 
                       list(model_module.dist_predictor.parameters()), 
             'lr': lr_config["stage2_new_modules"]},
            {'params': model_module.vint_model.parameters(), 'lr': lr_config["stage2_vint_backbone"]}
        ]
        stage2_optimizer = torch.optim.AdamW(param_groups)
        stage2_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(stage2_optimizer, T_max=stage2_epochs)
        
        for epoch in range(stage2_epochs):
            print(f"\n--- Stage 2, Epoch {epoch+1}/{stage2_epochs} ---")
            train_epoch(model, dataloader, stage2_optimizer, loss_fn, device, scaler, config, total_epochs_trained)
            stage2_scheduler.step()
            if (epoch + 1) % 5 == 0:
                evaluate_and_log(model, test_dataloaders, device, total_epochs_trained, config, stage=2)
                visualize_batch(model, next(iter(dataloader)), device, total_epochs_trained, 2, project_folder)
            total_epochs_trained += 1
        save_checkpoint(model, stage2_optimizer, stage2_scheduler, total_epochs_trained, project_folder)

    # --- STAGE 3: Fine-tune everything ---
    remaining_epochs = epochs - total_epochs_trained
    if start_stage <= 3 and remaining_epochs > 0:
        print(f"\n{'='*60}\nSTAGE 3: Full Fine-tuning\n{'='*60}")
        stage3_optimizer = torch.optim.AdamW(model.parameters(), lr=lr_config["stage3"])
        stage3_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(stage3_optimizer, mode='min', factor=0.5, patience=5, verbose=True)
        best_val_score = float('inf')
        
        for epoch in range(remaining_epochs):
            print(f"\n--- Stage 3, Epoch {epoch+1}/{remaining_epochs} ---")
            train_epoch(model, dataloader, stage3_optimizer, loss_fn, device, scaler, config, total_epochs_trained)
            
            val_metrics = evaluate_and_log(model, test_dataloaders, device, total_epochs_trained, config, stage=3)
            val_loader = next(iter(test_dataloaders.values()))
            visualize_batch(model, next(iter(val_loader)), device, total_epochs_trained, 3, project_folder)
            
            val_score = val_metrics.get('val_score', float('inf'))
            stage3_scheduler.step(val_score)
            
            is_best = val_score < best_val_score
            if is_best: best_val_score = val_score
            
            save_checkpoint(model, stage3_optimizer, stage3_scheduler, total_epochs_trained, project_folder, is_best=is_best)
            total_epochs_trained += 1

    print(f"\n{'='*60}\n Training Complete! Best validation score: {best_val_score:.4f}\n{'='*60}")

# <<< FIXED: Replaced placeholder with the full, robust training function >>>
def train_epoch(model, dataloader, optimizer, loss_fn, device, scaler, config, epoch):
    """Helper function to train for one epoch with enhanced stability checks."""
    model.train()
    grad_accum = config.get("gradient_accumulation_steps", 1)
    log_freq = config.get("print_log_freq", 100)
    use_wandb = config.get("use_wandb", True)

    for batch_idx, batch in enumerate(dataloader):
        # --- Data loading and moving to device ---
        obs_images = batch['obs_images'].to(device)
        goal_images = batch['goal_images'].to(device)
        true_actions = batch['actions'].to(device)
        true_distance = batch['distance'].to(device)
        true_seg = batch.get('obs_seg_mask', None)
        if true_seg is not None: true_seg = true_seg.to(device)
        action_mask = batch.get('action_mask', None)
        if action_mask is not None: action_mask = action_mask.to(device)
        
        if batch_idx == 0:
            if true_seg is not None and true_seg.sum() > 0:
                print("✅ Using pre-labeled segmentation mask for training.")
            else:
                print("⚠️ Warning: No pre-labeled segmentation mask found. Using fallback/heuristic.")


        # --- Forward pass with mixed precision ---
        with autocast(enabled=(scaler is not None)):
            outputs = model(obs_images, goal_images)
            losses = loss_fn(
                pred_actions=outputs['action_pred'], true_actions=true_actions,
                pred_dist=outputs['dist_pred'].squeeze(), true_dist=true_distance.float(),
                pred_seg=outputs['obs_seg_logits'], true_seg=true_seg,
                action_mask=action_mask,
            )
        
        loss = losses['total_loss']
        
        # SAFETY CHECK 1: Skip batch if loss is NaN or infinite
        if not torch.isfinite(loss):
            print(f" Warning: Skipping batch {batch_idx} due to invalid loss: {loss.item()}")
            optimizer.zero_grad()
            continue

        # --- Backward pass ---
        loss_scaled = loss / grad_accum
        if scaler:
            scaler.scale(loss_scaled).backward()
        else:
            loss_scaled.backward()

        # --- Optimizer step with gradient accumulation ---
        if (batch_idx + 1) % grad_accum == 0:
            if scaler:
                # <<< FIXED: This is the correct and safe way to use GradScaler >>>
                # Unscale the gradients before clipping
                scaler.unscale_(optimizer)
                # Clip the now-unscaled gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                # scaler.step() will check for NaNs/infs and skip the update if they are present
                scaler.step(optimizer)
                # Update the scale for the next iteration
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            
            optimizer.zero_grad(set_to_none=True)
            
            # Logging
            if batch_idx % log_freq == 0 and use_wandb:
                wandb.log({'train/total_loss': loss.item(), 'epoch': epoch})
        # Logging
        if batch_idx % log_freq == 0:
            # <<< EDITED: Moved print statement here and added detailed losses >>>
            print(f"  Batch [{batch_idx:4d}/{len(dataloader)}] "
                  f"Loss: {losses['total_loss'].item():.4f} "
                  f"(Act: {losses['action_loss'].item():.3f}, "
                  f"Dist: {losses['dist_loss'].item():.3f}, "
                  f"Seg: {losses['seg_loss'].item():.3f})")
            if use_wandb:
                wandb.log({
                    'train/total_loss': losses['total_loss'].item(),
                    'train/action_loss': losses['action_loss'].item(),
                    'train/dist_loss': losses['dist_loss'].item(),
                    'train/seg_loss': losses['seg_loss'].item(),
                    'epoch': epoch
                })

def evaluate_and_log(model, test_dataloaders, device, epoch, config, stage):
    """Helper function to evaluate and log metrics."""
    model_module = get_model_module(model)
    val_loader = next(iter(test_dataloaders.values()))
    
    print(f"\n--- Running evaluation for epoch {epoch} (Stage {stage}) ---")
    val_metrics = evaluate_segmentation_vint(
        model, val_loader, device, 
        num_seg_classes=model_module.num_seg_classes
    )
    
    print(f"Validation Metrics: mIoU={val_metrics.get('seg/mIoU', 0):.3f}, fFloor IoU={val_metrics.get('seg/iou_class_0', 0):.3f}, Success Rate={val_metrics.get('nav/success_rate', 0):.3f}, fSPL={val_metrics.get('nav/spl', 0):.3f}")
    
    if config.get("use_wandb", True):
        log_data = {f"val/{k.replace('/', '_')}": v for k, v in val_metrics.items()}
        log_data.update({'epoch': epoch, 'stage': stage})
        wandb.log(log_data)
        
    # Calculate a single score for schedulers and identifying the 'best' model
    val_metrics['val_score'] = (
        val_metrics.get('nav/mean_goal_distance', 1.0) * 0.5 +
        val_metrics.get('nav/collision_rate', 1.0) * 0.3 +
        (1 - val_metrics.get('nav/spl', 0.0)) * 0.2
    )
    
    return val_metrics