# FILE: train_segmentation.py

"""
Main training script for SegmentationViNT
"""
import os
import yaml
import argparse
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
import wandb

# Assuming all these helper modules exist in your project structure
from vint_train.models.vint.segmentation_vint import SegmentationViNT
from vint_train.data.segmentation_dataset import ViNTSegmentationDataset
from vint_train.training.train_eval_loop_seg import train_eval_loop_segmentation
from vint_train.training.train_utils import load_model

def main(config):
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # --- Data Loading ---
    train_datasets = []
    test_dataloaders = {}
    
    for dataset_name in config["datasets"]:
        data_config = config["datasets"][dataset_name]
        
        # <<< FIXED: Removed 'data_split_folder' from this dictionary >>>
        # Common dataset arguments for both training and testing
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
            # Segmentation specific
            "seg_data_folder": data_config.get("seg_data_folder", None),
            "seg_model_name": config.get("seg_model_name", "scand"),
            "use_pseudo_labels": config.get("use_pseudo_labels", False),
            "pseudo_label_model": config.get("pseudo_label_model", None),
        }

        # Create the training dataset, now providing data_split_folder only once
        train_dataset = ViNTSegmentationDataset(
            **dataset_args,
            data_split_folder=data_config["train"],
            is_train=True,
            seg_augmentation_prob=0.5
        )
        train_datasets.append(train_dataset)
        
        # Create the test dataset
        test_dataset = ViNTSegmentationDataset(
            **dataset_args,
            data_split_folder=data_config["test"],
            is_train=False,
            seg_augmentation_prob=0.0
        )
        
        test_dataloaders[f"{dataset_name}_test"] = DataLoader(
            test_dataset,
            batch_size=config.get("eval_batch_size", config["batch_size"]),
            shuffle=False,
            num_workers=config.get("num_workers", 2)
        )
    
    # Combine all training datasets
    train_dataset = ConcatDataset(train_datasets)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        drop_last=True,
        persistent_workers=True if config["num_workers"] > 0 else False
    )
    
    print(f"Created dataset with {len(train_dataset)} training samples.")
    
    # --- Model Initialization ---
    model = SegmentationViNT(
        context_size=config["context_size"],
        len_traj_pred=config["len_traj_pred"],
        learn_angle=config["learn_angle"],
        obs_encoder=config["obs_encoder"],
        obs_encoding_size=config["obs_encoding_size"],
        late_fusion=config["late_fusion"],
        mha_num_attention_heads=config["mha_num_attention_heads"],
        mha_num_attention_layers=config["mha_num_attention_layers"],
        mha_ff_dim_factor=config["mha_ff_dim_factor"],
        # Segmentation specific
        num_seg_classes=config.get("num_seg_classes", 5),
        seg_model_type=config.get("seg_model_type", "unet"),
        seg_encoder=config.get("seg_encoder", "resnet34"),
        fusion_type=config.get("fusion_type", "cross_attention"),
        freeze_vint_encoder=True,
        seg_feature_dim=config.get("seg_feature_dim", 256),
        use_semantic_goals=config.get("use_semantic_goals", True),
    )
    
    # --- Load pretrained ViNT weights ---
    if config.get("pretrained_vint_path"):
        print(f"\nLoading pretrained ViNT from {config['pretrained_vint_path']}")
        checkpoint = torch.load(config["pretrained_vint_path"], map_location='cpu')
        load_model(model, "segmentation_vint", checkpoint)
    
    model = model.to(device)
    
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)
    
    # --- W&B Initialization ---
    if config["use_wandb"]:
        wandb.init(
            project=config["project_name"],
            name=config["run_name"],
            config=config,
        )
    
    # --- Start Training ---
    train_eval_loop_segmentation(
        model=model,
        dataloader=train_loader,
        test_dataloaders=test_dataloaders,
        epochs=config["epochs"],
        device=device,
        project_folder=config["project_folder"],
        use_wandb=config["use_wandb"],
        initial_lr=config["lr"], # This value is now guaranteed to be a float
        stage1_epochs=config.get("stage1_epochs", 30),
        stage2_epochs=config.get("stage2_epochs", 40),
        gradient_accumulation_steps=config.get("gradient_accumulation_steps", 1),
        print_log_freq=config.get("print_log_freq", 50),
    )
    
    print("\nTraining complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", "-c",
        default="config/segmentation_vint.yaml",
        help="Path to the training config file"
    )
    args = parser.parse_args()
    
    # Load default and user configs
    default_config_path = "config/defaults.yaml"
    if os.path.exists(default_config_path):
        with open(default_config_path, "r") as f:
            config = yaml.safe_load(f)
    else:
        config = {}
    
    with open(args.config, "r") as f:
        user_config = yaml.safe_load(f)
    config.update(user_config)
    
    # <<< FIXED: Ensure learning rate is a float >>>
    if 'lr' in config:
        config['lr'] = float(config['lr'])
    
    # Create unique run directory
    config["run_name"] = config.get("run_name", "seg_vint") + "_" + time.strftime("%Y%m%d-%H%M%S")
    config["project_folder"] = os.path.join("logs", config["project_name"], config["run_name"])
    os.makedirs(config["project_folder"], exist_ok=True)
    
    # Save the final config to the run folder for reproducibility
    with open(os.path.join(config["project_folder"], "config.yaml"), "w") as f:
        yaml.dump(config, f)
    
    main(config)
