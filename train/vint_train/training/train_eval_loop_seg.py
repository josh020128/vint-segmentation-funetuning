# FILE: vint_train/training/train_eval_loop_seg.py

"""
Training and evaluation loop for the dual-input ViNT model with a 
"safety co-pilot" architecture.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
import wandb
from typing import Dict
import os
import shutil
from tqdm import tqdm
import numpy as np

# --- Main project imports ---
from vint_train.training.seg_losses import CoPilotNavigationLoss
from vint_train.training.seg_metrics import evaluate_segmentation_vint
from vint_train.visualizing.visualize_segmentation import visualize_segmentation_predictions

# --- Helper Functions ---

def get_model_module(model: nn.Module) -> nn.Module:
    """Gets the actual model module, handling nn.DataParallel wrappers."""
    return model.module if hasattr(model, 'module') else model

def save_checkpoint(model, optimizer, scheduler, epoch, project_folder, is_best=False):
    """Saves the latest and (optionally) the best model checkpoints."""
    if not os.path.exists(project_folder):
        os.makedirs(project_folder)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': get_model_module(model).state_dict(),
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

def visualize_batch(model, batch, device, epoch, config, project_folder, stage):
    """Generates and logs a visualization for a batch of validation data."""
    model.eval()
    
    # Prepare data and move to device
    obs_images = batch['obs_images'].to(device)
    goal_images = batch['goal_images'].to(device)
    obs_seg_mask_one_hot = batch['obs_seg_mask'].to(device)
    
    with torch.no_grad():
        outputs = model(obs_images, goal_images, obs_seg_mask_one_hot)

    # Convert one-hot mask back to class indices for visualization
    seg_mask_labels = torch.argmax(obs_seg_mask_one_hot, dim=1)

    # Call the main visualization function
    visualize_segmentation_predictions(
        batch_obs_images=to_numpy(obs_images),
        batch_goal_images=to_numpy(goal_images),
        batch_seg_preds=to_numpy(seg_mask_labels), # Use the input mask as the "prediction"
        batch_seg_labels=to_numpy(seg_mask_labels),
        batch_pred_waypoints=to_numpy(outputs['action_pred']),
        batch_label_waypoints=to_numpy(batch['actions']),
        batch_goals=to_numpy(batch.get('goal_pos', torch.zeros(obs_images.shape[0], 2))),
        save_folder=project_folder,
        epoch=epoch,
        eval_type=f"stage_{stage}_eval",
        use_wandb=config.get("use_wandb", True),
    )
    model.train() # Set model back to training mode

# --- Main Training & Evaluation Loop ---

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
    """Two-stage training loop for the co-pilot ViNT model."""
    total_epochs_trained = 0
    model_module = get_model_module(model)
    
    lr_config = config["lr_schedule"]
    stage1_epochs = config.get("stage1_epochs", 25)
    
    # <<< MODIFIED: Use the correct loss function for the co-pilot model >>>
    loss_fn = CoPilotNavigationLoss(
        action_loss_weight=config.get("action_loss_weight", 1.0),
        dist_loss_weight=config.get("dist_loss_weight", 0.5),
        consistency_weight=config.get("consistency_weight", 0.2)
    )
    scaler = GradScaler() if device.type == 'cuda' else None
    best_val_score = float('inf')

    # --- STAGE 1: Train co-pilot modules with ViNT frozen ---
    if start_stage <= 1:
        print(f"\n{'='*60}\nSTAGE 1: Training Co-Pilot Modules (ViNT Frozen)\n{'='*60}")
        
        # <<< MODIFIED: Train the seg_encoder, trajectory_adapter, and dist_predictor >>>
        trainable_params = list(model_module.seg_encoder.parameters()) + \
                           list(model_module.trajectory_adapter.parameters()) + \
                           list(model_module.dist_predictor.parameters())
        
        stage1_optimizer = torch.optim.AdamW(trainable_params, lr=lr_config["stage1"])
        stage1_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(stage1_optimizer, T_max=stage1_epochs)
        
        for epoch in range(stage1_epochs):
            print(f"\n--- Stage 1, Epoch {epoch+1}/{stage1_epochs} ---")
            train_epoch(model, dataloader, stage1_optimizer, loss_fn, device, scaler, config, total_epochs_trained)
            stage1_scheduler.step()
            
            if (epoch + 1) % config.get("eval_freq", 5) == 0:
                val_metrics = evaluate_and_log(model, test_dataloaders, device, total_epochs_trained, config, stage=1)

                vis_loader = next(iter(test_dataloaders.values()))
                vis_batch = next(iter(vis_loader))
                visualize_batch(model, vis_batch, device, total_epochs_trained, config, project_folder, stage=1)

                val_score = val_metrics.get('val_score', float('inf'))
                is_best = val_score < best_val_score
                if is_best: best_val_score = val_score
                save_checkpoint(model, stage1_optimizer, stage1_scheduler, total_epochs_trained, project_folder, is_best=is_best)

            total_epochs_trained += 1
        
        save_checkpoint(model, stage1_optimizer, stage1_scheduler, total_epochs_trained, project_folder)

    # --- STAGE 2: Fine-tune the entire model with discriminative learning rates ---
    remaining_epochs = epochs - total_epochs_trained
    if start_stage <= 2 and remaining_epochs > 0:
        print(f"\n{'='*60}\nSTAGE 2: Fine-tuning Full Model (Discriminative LRs)\n{'='*60}")
        model_module.unfreeze_vint()
        
        # <<< MODIFIED: Correctly group parameters for the co-pilot model >>>
        param_groups = [
            {
                'params': list(model_module.seg_encoder.parameters()) + 
                          list(model_module.trajectory_adapter.parameters()) + 
                          list(model_module.dist_predictor.parameters()), 
                'lr': lr_config["stage2_new_modules"]
            },
            {
                'params': model_module.vint_model.parameters(), 
                'lr': lr_config["stage2_vint_backbone"]
            }
        ]
        
        stage2_optimizer = torch.optim.AdamW(param_groups)
        stage2_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(stage2_optimizer, T_max=remaining_epochs)
        
        for epoch in range(remaining_epochs):
            print(f"\n--- Stage 2, Epoch {epoch+1}/{remaining_epochs} ---")
            train_epoch(model, dataloader, stage2_optimizer, loss_fn, device, scaler, config, total_epochs_trained)
            stage2_scheduler.step()
            
            if (epoch + 1) % config.get("eval_freq", 5) == 0:
                val_metrics = evaluate_and_log(model, test_dataloaders, device, total_epochs_trained, config, stage=2)
                
                vis_loader = next(iter(test_dataloaders.values()))
                vis_batch = next(iter(vis_loader))
                visualize_batch(model, vis_batch, device, total_epochs_trained, config, project_folder, stage=2)
                
                val_score = val_metrics.get('val_score', float('inf'))
                is_best = val_score < best_val_score
                if is_best: best_val_score = val_score
                save_checkpoint(model, stage2_optimizer, stage2_scheduler, total_epochs_trained, project_folder, is_best=is_best)
            
            total_epochs_trained += 1

    print(f"\n{'='*60}\nTraining Complete! Best validation score: {best_val_score:.4f}\n{'='*60}")

def train_epoch(model, dataloader, optimizer, loss_fn, device, scaler, config, epoch):
    """Runs a single training epoch for the co-pilot model."""
    model.train()
    grad_accum = config.get("gradient_accumulation_steps", 1)
    log_freq = config.get("print_log_freq", 100)
    use_wandb = config.get("use_wandb", True)

    for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch} Training")):
        obs_images = batch['obs_images'].to(device)
        goal_images = batch['goal_images'].to(device)
        obs_seg_masks = batch['obs_seg_mask'].to(device)
        
        targets = {
            'actions': batch['actions'].to(device),
            'distance': batch['distance'].to(device)
        }
        action_mask = batch.get('action_mask', None)
        if action_mask is not None:
            action_mask = action_mask.to(device)

        with autocast(enabled=(scaler is not None)):
            outputs = model(obs_images, goal_images, obs_seg_masks)
            
            # <<< MODIFIED: Call the new loss function with the correct arguments >>>
            losses = loss_fn(
                outputs=outputs, 
                targets=targets,
                action_mask=action_mask
            )
        
        loss = losses['total_loss']
        
        if not torch.isfinite(loss):
            print(f"Warning: Skipping batch {batch_idx} due to invalid loss: {loss.item()}")
            optimizer.zero_grad()
            continue

        loss_scaled = loss / grad_accum
        if scaler:
            scaler.scale(loss_scaled).backward()
        else:
            loss_scaled.backward()

        if (batch_idx + 1) % grad_accum == 0:
            if scaler:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            
            optimizer.zero_grad(set_to_none=True)

        # <<< MODIFIED: Update logging for the new loss components >>>
        if use_wandb and (batch_idx % log_freq == 0):
            log_data = {f'train/{k}': v.item() for k, v in losses.items()}
            log_data['train/seg_influence'] = outputs.get('seg_influence', 0)
            log_data['epoch'] = epoch
            wandb.log(log_data)

def evaluate_and_log(model, test_dataloaders, device, epoch, config, stage):
    """Runs evaluation and logs the navigation metrics."""
    val_loader = next(iter(test_dataloaders.values()))
    
    print(f"\n--- Running evaluation for epoch {epoch} (Stage {stage}) ---")
    val_metrics = evaluate_segmentation_vint(model, val_loader, device, config)
    
    print(f"Validation Metrics: Success={val_metrics['nav/success_rate']:.3f}, "
          f"Collision={val_metrics['nav/collision_rate']:.3f}, "
          f"SPL={val_metrics['nav/spl']:.3f}")
    
    if config.get("use_wandb", True):
        log_data = {f"val/{k.replace('/', '_')}": v for k, v in val_metrics.items()}
        log_data.update({'epoch': epoch, 'stage': stage})
        wandb.log(log_data)
    
    # Define a single validation score for checkpointing (lower is better)
    val_metrics['val_score'] = (
        val_metrics.get('nav/mean_goal_distance', 1.0) * 0.5 +
        val_metrics.get('nav/collision_rate', 1.0) * 0.5
    )
    return val_metrics
