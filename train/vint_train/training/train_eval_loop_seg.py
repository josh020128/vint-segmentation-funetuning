"""
Training and evaluation loop for SegmentationViNT with FuSe schedule
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
    
    # Also save periodic checkpoints for safety
    if (epoch + 1) % 10 == 0:
        periodic_path = os.path.join(project_folder, f'checkpoint_epoch_{epoch+1}.pth')
        shutil.copyfile(latest_path, periodic_path)
        print(f"Periodic checkpoint saved for epoch {epoch+1}")

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
    initial_lr: float = 1e-4,
    use_wandb: bool = True,
    print_log_freq: int = 100,
    gradient_accumulation_steps: int = 1,
    stage1_epochs: int = 20,
    stage2_epochs: int = 40,
    start_stage: int = 1, # Added for resuming
):
    """3-stage training for SegmentationViNT."""
    total_epochs_trained = 0
    model_module = get_model_module(model)
    
    loss_fn = SegmentationNavigationLoss(use_uncertainty_weighting=True)
    scaler = GradScaler() if device.type == 'cuda' else None
    
    # --- STAGE 1: Train new modules only ---
    
    if start_stage <= 1 :
        print(f"\n{'='*60}\nSTAGE 1: Training Segmentation & Fusion (ViNT Frozen)\n{'='*60}")
        stage1_optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=initial_lr)
        stage1_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(stage1_optimizer, T_max=stage1_epochs)
        for epoch in range(stage1_epochs):
            print(f"\n--- Stage 1, Epoch {epoch+1}/{stage1_epochs} ---")
            train_epoch(model, dataloader, stage1_optimizer, loss_fn, device, scaler, gradient_accumulation_steps, print_log_freq, use_wandb, total_epochs_trained)
            stage1_scheduler.step()
            if (epoch + 1) % 5 == 0:
                evaluate_and_log(model, test_dataloaders, device, total_epochs_trained, use_wandb, stage=1)
                visualize_batch(model, next(iter(dataloader)), device, total_epochs_trained, 1, project_folder, use_wandb)
            total_epochs_trained += 1
        save_checkpoint(model, stage1_optimizer, stage1_scheduler, total_epochs_trained, project_folder)

    # --- STAGE 2: Unfreeze and train with different LRs ---
    if start_stage <= 2:
        print(f"\n{'='*60}\nSTAGE 2: Unfreezing ViNT (Discriminative LRs)\n{'='*60}")
        model_module.unfreeze_vint_encoder()
        param_groups = [
            {'params': list(model_module.seg_model.parameters()) + list(model_module.fusion_module.parameters()), 'lr': initial_lr},
            {'params': model_module.vint_model.parameters(), 'lr': initial_lr * 0.001}
        ]
        stage2_optimizer = torch.optim.AdamW(param_groups)
        stage2_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(stage2_optimizer, T_max=stage2_epochs)
        
        for epoch in range(stage2_epochs):
            print(f"\n--- Stage 2, Epoch {epoch+1}/{stage2_epochs} ---")
            train_epoch(model, dataloader, stage2_optimizer, loss_fn, device, scaler, gradient_accumulation_steps, print_log_freq, use_wandb, total_epochs_trained)
            stage2_scheduler.step()
            if (epoch + 1) % 5 == 0:
                evaluate_and_log(model, test_dataloaders, device, total_epochs_trained, use_wandb, stage=2)
                visualize_batch(model, next(iter(dataloader)), device, total_epochs_trained, 2, project_folder, use_wandb)
            total_epochs_trained += 1
        save_checkpoint(model, stage2_optimizer, stage2_scheduler, total_epochs_trained, project_folder)


    # --- STAGE 3: Fine-tune everything ---
    remaining_epochs = epochs - total_epochs_trained
    if start_stage <= 3 and remaining_epochs > 0:
        print(f"\n{'='*60}\nSTAGE 3: Full Fine-tuning\n{'='*60}")
        stage3_optimizer = torch.optim.AdamW(model.parameters(), lr=initial_lr * 0.001)
        stage3_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(stage3_optimizer, mode='min', factor=0.5, patience=5, verbose=True)
        best_val_score = float('inf')
        
        for epoch in range(remaining_epochs):
            print(f"\n--- Stage 3, Epoch {epoch+1}/{remaining_epochs} ---")
            train_epoch(model, dataloader, stage3_optimizer, loss_fn, device, scaler, gradient_accumulation_steps, print_log_freq, use_wandb, total_epochs_trained)
            
            val_metrics = evaluate_and_log(model, test_dataloaders, device, total_epochs_trained, use_wandb, stage=3)
            val_loader = next(iter(test_dataloaders.values()))
            visualize_batch(model, next(iter(val_loader)), device, total_epochs_trained, 3, project_folder, use_wandb)
            
            val_score = val_metrics.get('val_score', float('inf'))
            stage3_scheduler.step(val_score)
            
            is_best = val_score < best_val_score
            if is_best: best_val_score = val_score
            
            save_checkpoint(model, stage3_optimizer, stage3_scheduler, total_epochs_trained, project_folder, is_best=is_best)
            total_epochs_trained += 1

    print(f"\n{'='*60}\n Training Complete! Best validation score: {best_val_score:.4f}\n{'='*60}")

def train_epoch(model, dataloader, optimizer, loss_fn, device, scaler, 
                grad_accum, log_freq, use_wandb, epoch):
    """Train for one epoch"""
    model.train()
    epoch_losses = {
        'total': [],
        'action': [],
        'dist': [],
        'seg': []
    }
    
    for batch_idx, batch in enumerate(dataloader):
        # Handle both dict and tuple formats
        if isinstance(batch, dict):
            obs_images = batch['obs_images'].to(device)
            goal_images = batch['goal_images'].to(device)
            actions = batch['actions'].to(device)
            distance = batch['distance'].to(device)
            obs_seg_mask = batch.get('obs_seg_mask')  # Ground truth mask
            action_mask = batch.get('action_mask')
        else:
            # Tuple format from original dataset
            obs_images, goal_images, _, _, actions, distance, _, _, action_mask = batch
            obs_images = obs_images.to(device)
            goal_images = goal_images.to(device)
            actions = actions.to(device)
            distance = distance.to(device)
            obs_seg_mask = None
        
        # Move masks to device if they exist
        if obs_seg_mask is not None:
            obs_seg_mask = obs_seg_mask.to(device)
        if action_mask is not None:
            action_mask = action_mask.to(device)
        
        # CRITICAL FIX: Model should NOT see ground truth segmentation!
        # The model generates its own segmentation from RGB images
        with autocast(enabled=(scaler is not None)):
            # Model only gets RGB images as input
            outputs = model(obs_images, goal_images)  # NO obs_seg_mask here!
            
            # Loss function uses ground truth for supervision
            losses = loss_fn(
                pred_actions=outputs['action_pred'],
                true_actions=actions,
                pred_dist=outputs['dist_pred'].squeeze(),
                true_dist=distance.float(),
                pred_seg=outputs['obs_seg_logits'],  # Model's prediction
                true_seg=obs_seg_mask,  # Ground truth for loss calculation
                action_mask=action_mask,
            )
        
        loss = losses['total_loss'] / grad_accum
        
        if scaler:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Gradient accumulation
        if (batch_idx + 1) % grad_accum == 0:
            if scaler:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()
            
            optimizer.zero_grad()
        
        # Record losses
        epoch_losses['total'].append(losses['total_loss'].item())
        epoch_losses['action'].append(losses['action_loss'].item())
        epoch_losses['dist'].append(losses['dist_loss'].item())
        epoch_losses['seg'].append(losses['seg_loss'].item())
        
        # Logging
        if batch_idx % log_freq == 0:
            print(f"  Batch [{batch_idx:4d}/{len(dataloader)}] "
                f"Loss: {losses['total_loss'].item():.4f} "
                f"(Act: {losses['action_loss'].item():.3f}, "
                f"Dist: {losses['dist_loss'].item():.3f}, "
                f"Seg: {losses['seg_loss'].item():.3f})")
            
            if use_wandb:
                wandb.log({
                    'train/total_loss': losses['total_loss'],
                    'train/action_loss': losses['action_loss'],
                    'train/dist_loss': losses['dist_loss'],
                    'train/seg_loss': losses['seg_loss'],
                    'train/batch_idx': epoch * len(dataloader) + batch_idx,
                    'epoch': epoch
                })
    
    # Return epoch averages
    return {
        'total_loss': np.mean(epoch_losses['total']),
        'action_loss': np.mean(epoch_losses['action']),
        'dist_loss': np.mean(epoch_losses['dist']),
        'seg_loss': np.mean(epoch_losses['seg'])
    }

def evaluate_and_log(model, test_dataloaders, device, epoch, use_wandb, stage):
    """Evaluate and log metrics"""
    model_module = get_model_module(model)
    
    # Get first validation dataloader
    val_loader_name = next(iter(test_dataloaders.keys()))
    val_loader = test_dataloaders[val_loader_name]
    
    print(f"  Evaluating on {val_loader_name}...")
    
    # Run evaluation
    val_metrics = evaluate_segmentation_vint(
        model, val_loader, device, 
        num_seg_classes=model_module.num_seg_classes
    )
    
    # Print key metrics
    print(f"  Stage {stage} Epoch {epoch} Validation:")
    print(f"    - mIoU: {val_metrics.get('seg/mIoU', 0):.3f}")
    print(f"    - Success Rate: {val_metrics.get('nav/success_rate', 0):.3f}")
    print(f"    - Collision Rate: {val_metrics.get('nav/collision_rate', 0):.3f}")
    print(f"    - SPL: {val_metrics.get('nav/spl', 0):.3f}")
    
    # Log to wandb
    if use_wandb:
        wandb.log({
            **{f'val/{k}': v for k, v in val_metrics.items()},
            'epoch': epoch,
            'stage': stage
        })
    
    # Calculate total validation loss for scheduler
    val_metrics['total_loss'] = (
        val_metrics.get('nav/mean_goal_distance', 0) * 0.5 +
        val_metrics.get('nav/collision_rate', 0) * 0.3 +
        (1 - val_metrics.get('nav/success_rate', 0)) * 0.2
    )
    
    return val_metrics