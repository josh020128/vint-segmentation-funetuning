# FILE: train_segmentation.py

"""
Main training script for the dual-input ViNT model with a "safety co-pilot" architecture.
"""
import os
import yaml
import argparse
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
import wandb

# --- Main project imports ---
from vint_train.models.vint.segmentation_vint import SegmentationViNT
from vint_train.data.segmentation_dataset import ViNTSegmentationDataset
from vint_train.training.train_eval_loop_seg import train_eval_loop_segmentation
from vint_train.training.train_utils import load_model

def main(config, resume_path=None, start_stage=1):
    """Main function to set up and run the training process."""
    # --- Device Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # --- Data Loading ---
    train_datasets = []
    test_dataloaders = {}
    
    # --- Sanity Check for Segmentation Data ---
    print("\n" + "="*60)
    print("Verifying Segmentation Data Source for Co-Pilot Model:")
    first_dataset_key = next(iter(config.get("datasets", {})), None)
    if first_dataset_key:
        seg_folder = config["datasets"][first_dataset_key].get("seg_data_folder")
        if seg_folder and os.path.exists(seg_folder):
            print(f"✅ Loading pre-computed segmentation masks from: '{seg_folder}'")
        else:
            raise FileNotFoundError(f"❌ Critical Error: 'seg_data_folder' is required and was not found at '{seg_folder}'.")
    else:
        raise ValueError("❌ Critical Error: No datasets found in the configuration file.")
    print("="*60 + "\n")
    # -----------------------------------------

    for dataset_name, data_config in config["datasets"].items():
        dataset_args = {
            "data_folder": data_config["data_folder"],
            "dataset_name": dataset_name,
            "image_size": tuple(config["image_size"]),
            "waypoint_spacing": data_config.get("waypoint_spacing", 1),
            "min_dist_cat": config["distance"]["min_dist_cat"],
            "max_dist_cat": config["distance"]["max_dist_cat"],
            "min_action_distance": config["action"]["min_dist_cat"],
            "max_action_distance": config["action"]["max_dist_cat"],
            "negative_mining": data_config.get("negative_mining", True),
            "len_traj_pred": config["len_traj_pred"],
            "learn_angle": config["learn_angle"],
            "context_size": config["context_size"],
            "seg_data_folder": data_config.get("seg_data_folder"), # Mandatory
            "seg_model_name": config.get("seg_model_name", "scand"),
        }

        train_datasets.append(ViNTSegmentationDataset(**dataset_args, data_split_folder=data_config["train"], is_train=True, seg_augmentation_prob=0.5))
        test_dataset = ViNTSegmentationDataset(**dataset_args, data_split_folder=data_config["test"], is_train=False, seg_augmentation_prob=0.0)
        test_dataloaders[f"{dataset_name}_test"] = DataLoader(test_dataset, batch_size=config.get("eval_batch_size", config["batch_size"]), shuffle=False, num_workers=config.get("num_workers", 2))
    
    train_dataset = ConcatDataset(train_datasets)
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=config["num_workers"], drop_last=True, persistent_workers=True if config["num_workers"] > 0 else False)
    
    print(f"Created dataset with {len(train_dataset)} training samples.")
    
    # --- Model Initialization ---
    model = SegmentationViNT(
        context_size=config["context_size"],
        len_traj_pred=config["len_traj_pred"],
        learn_angle=config["learn_angle"],
        obs_encoder=config["obs_encoder"],
        obs_encoding_size=config["obs_encoding_size"],
        num_seg_classes=config.get("num_seg_classes", 5),
        seg_feature_dim=config.get("seg_feature_dim", 256),
        freeze_vint=True, # Always start with the ViNT backbone frozen for Stage 1
        **config.get("model_params", {})
    )
    
    # --- Load Checkpoints ---
    if resume_path:
        print(f"\nResuming training from checkpoint: {resume_path}")
        checkpoint = torch.load(resume_path, map_location='cpu')
        load_model(model, "segmentation_vint_resume", checkpoint)
    elif config.get("pretrained_vint_path"):
        print(f"\nLoading pretrained ViNT weights from {config['pretrained_vint_path']}")
        checkpoint = torch.load(config["pretrained_vint_path"], map_location='cpu')
        load_model(model.vint_model, "vint", checkpoint)
    
    model = model.to(device)
    
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)
    
    # --- W&B Initialization ---
    if config.get("use_wandb", True):
        wandb.init(project=config["project_name"], name=config["run_name"], config=config)
    
    # --- Start Training ---
    train_eval_loop_segmentation(
        model=model,
        dataloader=train_loader,
        test_dataloaders=test_dataloaders,
        epochs=config["epochs"],
        device=device,
        project_folder=config["project_folder"],
        config=config,
        start_stage=start_stage
    )
    
    print("\nTraining complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Main training script for the co-pilot ViNT model.")
    parser.add_argument("--config", "-c", default="config/segmentation_vint.yaml", help="Path to the configuration file.")
    parser.add_argument("--resume_path", type=str, default=None, help="Path to a checkpoint to resume training from.")
    parser.add_argument("--start_stage", type=int, default=1, choices=[1, 2], help="Which training stage to start from.")
    args = parser.parse_args()
    
    # Load configuration from YAML file
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    # <<< FIXED: Updated learning rate schedule for the 2-stage co-pilot training process >>>
    if 'lr' in config:
        base_lr = float(config['lr'])
        config['lr_schedule'] = {
            'stage1': base_lr,                 # LR for new co-pilot modules
            'stage2_new_modules': base_lr,     # LR for co-pilot modules during fine-tuning
            'stage2_vint_backbone': base_lr * 0.01, # A smaller LR for the ViNT backbone
        }
    
    # Create unique run directory
    config["run_name"] = config.get("run_name", "copilot_vint") + "_" + time.strftime("%Y%m%d-%H%M%S")
    config["project_folder"] = os.path.join("logs", config["project_name"], config["run_name"])
    os.makedirs(config["project_folder"], exist_ok=True)
    
    # Save the final config for this run
    with open(os.path.join(config["project_folder"], "config.yaml"), "w") as f:
        yaml.dump(config, f)
    
    main(config, args.resume_path, args.start_stage)
