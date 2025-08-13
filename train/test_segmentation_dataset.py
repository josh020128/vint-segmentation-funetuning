"""
Test script to verify SegmentationViNT dataset and model pipeline
"""
import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import yaml
import argparse
from typing import Dict, Any

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vint_train.data.segmentation_dataset import ViNTSegmentationDataset
from vint_train.models.vint.segmentation_vint import SegmentationViNT
from torch.utils.data import DataLoader

def visualize_segmentation(seg_mask: torch.Tensor, class_names: list, title: str = "Segmentation"):
    """Visualize segmentation mask with color coding"""
    # Define colors for each class
    colors = [
        [0, 255, 0],    # 0: floor/navigable - green
        [255, 0, 0],    # 1: wall/obstacle - red
        [255, 255, 0],  # 2: door - yellow
        [0, 0, 255],    # 3: furniture - blue
        [128, 128, 128] # 4: unknown - gray
    ]
    
    # Ensure we have enough colors
    while len(colors) < len(class_names):
        colors.append([np.random.randint(0, 255) for _ in range(3)])
    
    # Convert mask to RGB
    h, w = seg_mask.shape
    rgb_mask = np.zeros((h, w, 3), dtype=np.uint8)
    
    for class_id in range(len(class_names)):
        mask = (seg_mask == class_id).numpy()
        rgb_mask[mask] = colors[class_id]
    
    return rgb_mask

def test_dataset_loading(config: Dict[str, Any]):
    """Test basic dataset loading and data format"""
    print("\n" + "="*60)
    print("TEST 1: Dataset Loading")
    print("="*60)
    
    dataset_name = list(config['datasets'].keys())[0]
    data_config = config['datasets'][dataset_name]
    
    print(f"Dataset: {dataset_name}")
    print(f"Data folder: {data_config['data_folder']}")
    print(f"Seg folder: {data_config.get('seg_data_folder', 'None (using pseudo labels)')}")
    print(f"Use pseudo labels: {config.get('use_pseudo_labels', False)}")
    
    # Create dataset
    try:
        dataset = ViNTSegmentationDataset(
            data_folder=data_config['data_folder'],
            data_split_folder=data_config['train'],
            dataset_name=dataset_name,
            image_size=tuple(config['image_size']),
            waypoint_spacing=data_config.get('waypoint_spacing', 1),
            min_dist_cat=config['distance']['min_dist_cat'],
            max_dist_cat=config['distance']['max_dist_cat'],
            min_action_distance=config['action']['min_dist_cat'],
            max_action_distance=config['action']['max_dist_cat'],
            negative_mining=data_config.get('negative_mining', True),
            len_traj_pred=config['len_traj_pred'],
            learn_angle=config['learn_angle'],
            context_size=config['context_size'],
            # Segmentation specific
            seg_data_folder=data_config.get('seg_data_folder', None),
            seg_model_name=config.get('seg_model_name', 'scand'),
            use_pseudo_labels=config.get('use_pseudo_labels', False),
            pseudo_label_model=config.get('pseudo_label_model', None),
            seg_augmentation_prob=0.5,
            # is_train=True,
        )
        
        print(f"✓ Dataset created successfully!")
        print(f"  - Number of samples: {len(dataset)}")
        print(f"  - Number of classes: {dataset.num_classes}")
        print(f"  - Class names: {dataset.class_names}")
        
        return dataset
        
    except Exception as e:
        print(f"✗ Failed to create dataset: {e}")
        return None

def test_data_sample(dataset: ViNTSegmentationDataset, idx: int = 0):
    """Test loading a single sample"""
    print("\n" + "="*60)
    print(f"TEST 2: Loading Sample {idx}")
    print("="*60)
    
    try:
        sample = dataset[idx]
        
        print("Sample contents:")
        for key, value in sample.items():
            if isinstance(value, torch.Tensor):
                print(f"  - {key}: shape={value.shape}, dtype={value.dtype}, "
                      f"min={value.min():.3f}, max={value.max():.3f}")
            else:
                print(f"  - {key}: {value}")
        
        # Verify expected keys
        expected_keys = [
            'obs_images', 'goal_images', 'obs_seg_mask', 'goal_seg_mask',
            'actions', 'distance', 'goal_pos', 'dataset_idx', 'action_mask'
        ]
        
        missing_keys = [k for k in expected_keys if k not in sample]
        if missing_keys:
            print(f"✗ Missing keys: {missing_keys}")
        else:
            print("✓ All expected keys present!")
        
        # Check data shapes
        context_size = dataset.context_size
        expected_obs_shape = (3 * context_size, *dataset.image_size)
        expected_goal_shape = (3, *dataset.image_size)
        expected_seg_shape = dataset.image_size
        
        checks = [
            (sample['obs_images'].shape, expected_obs_shape, "obs_images"),
            (sample['goal_images'].shape, expected_goal_shape, "goal_images"),
            (sample['obs_seg_mask'].shape, expected_seg_shape, "obs_seg_mask"),
            (sample['goal_seg_mask'].shape, expected_seg_shape, "goal_seg_mask"),
        ]
        
        print("\nShape verification:")
        for actual, expected, name in checks:
            if actual == expected:
                print(f"  ✓ {name}: {actual}")
            else:
                print(f"  ✗ {name}: expected {expected}, got {actual}")
        
        return sample
        
    except Exception as e:
        print(f"✗ Failed to load sample: {e}")
        import traceback
        traceback.print_exc()
        return None

def visualize_sample(sample: Dict[str, torch.Tensor], dataset: ViNTSegmentationDataset, 
                     save_path: str = "test_visualization.png"):
    """Visualize RGB and segmentation for a sample"""
    print("\n" + "="*60)
    print("TEST 3: Visualizing Sample")
    print("="*60)
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    # Extract last observation frame (current frame)
    context_size = dataset.context_size
    last_obs_rgb = sample['obs_images'][-3:, :, :]  # Last 3 channels
    
    # Denormalize if needed
    if last_obs_rgb.min() < 0:
        # Assuming ImageNet normalization
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        last_obs_rgb = last_obs_rgb * std + mean
    
    # Convert to numpy for visualization
    obs_rgb_np = last_obs_rgb.permute(1, 2, 0).numpy()
    obs_rgb_np = np.clip(obs_rgb_np, 0, 1)
    
    goal_rgb = sample['goal_images']
    if goal_rgb.min() < 0:
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        goal_rgb = goal_rgb * std + mean
    goal_rgb_np = goal_rgb.permute(1, 2, 0).numpy()
    goal_rgb_np = np.clip(goal_rgb_np, 0, 1)
    
    # Visualize observation
    axes[0, 0].imshow(obs_rgb_np)
    axes[0, 0].set_title("Observation RGB (Last Frame)")
    axes[0, 0].axis('off')
    
    # Visualize observation segmentation
    obs_seg_rgb = visualize_segmentation(sample['obs_seg_mask'], dataset.class_names)
    axes[0, 1].imshow(obs_seg_rgb)
    axes[0, 1].set_title("Observation Segmentation")
    axes[0, 1].axis('off')
    
    # Visualize goal
    axes[0, 2].imshow(goal_rgb_np)
    axes[0, 2].set_title("Goal RGB")
    axes[0, 2].axis('off')
    
    # Visualize goal segmentation
    goal_seg_rgb = visualize_segmentation(sample['goal_seg_mask'], dataset.class_names)
    axes[0, 3].imshow(goal_seg_rgb)
    axes[0, 3].set_title("Goal Segmentation")
    axes[0, 3].axis('off')
    
    # Plot trajectory
    actions = sample['actions'].numpy()
    axes[1, 0].plot(actions[:, 0], actions[:, 1], 'b.-', markersize=8)
    axes[1, 0].scatter([0], [0], c='g', s=100, marker='o', label='Start')
    axes[1, 0].scatter(actions[-1, 0], actions[-1, 1], c='r', s=100, marker='*', label='End')
    axes[1, 0].set_title(f"Trajectory ({len(actions)} waypoints)")
    axes[1, 0].set_xlabel("X")
    axes[1, 0].set_ylabel("Y")
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    axes[1, 0].axis('equal')
    
    # Segmentation statistics
    axes[1, 1].axis('off')
    obs_seg = sample['obs_seg_mask']
    goal_seg = sample['goal_seg_mask']
    
    stats_text = "Observation Seg Stats:\n"
    for i, class_name in enumerate(dataset.class_names):
        count = (obs_seg == i).sum().item()
        percentage = count / obs_seg.numel() * 100
        stats_text += f"  {class_name}: {percentage:.1f}%\n"
    
    stats_text += "\nGoal Seg Stats:\n"
    for i, class_name in enumerate(dataset.class_names):
        count = (goal_seg == i).sum().item()
        percentage = count / goal_seg.numel() * 100
        stats_text += f"  {class_name}: {percentage:.1f}%\n"
    
    axes[1, 1].text(0.1, 0.5, stats_text, fontsize=10, verticalalignment='center')
    axes[1, 1].set_title("Segmentation Statistics")
    
    # Action statistics
    axes[1, 2].axis('off')
    action_text = f"Distance to goal: {sample['distance'].item()}\n"
    action_text += f"Goal position: ({sample['goal_pos'][0]:.2f}, {sample['goal_pos'][1]:.2f})\n"
    action_text += f"Action mask: {sample['action_mask'].item()}\n"
    action_text += f"Dataset index: {sample['dataset_idx'].item()}\n"
    action_text += f"Trajectory length: {np.linalg.norm(actions[1:] - actions[:-1], axis=1).sum():.2f}"
    
    axes[1, 2].text(0.1, 0.5, action_text, fontsize=10, verticalalignment='center')
    axes[1, 2].set_title("Navigation Info")
    
    # Legend for segmentation colors
    axes[1, 3].axis('off')
    legend_elements = []
    colors_normalized = [
        [0, 1, 0],      # green
        [1, 0, 0],      # red
        [1, 1, 0],      # yellow
        [0, 0, 1],      # blue
        [0.5, 0.5, 0.5] # gray
    ]
    
    for i, class_name in enumerate(dataset.class_names):
        if i < len(colors_normalized):
            color = colors_normalized[i]
        else:
            color = [np.random.random() for _ in range(3)]
        axes[1, 3].add_patch(plt.Rectangle((0.1, 0.8 - i*0.15), 0.1, 0.1, 
                                          facecolor=color))
        axes[1, 3].text(0.25, 0.85 - i*0.15, class_name, fontsize=12, 
                       verticalalignment='center')
    
    axes[1, 3].set_xlim(0, 1)
    axes[1, 3].set_ylim(0, 1)
    axes[1, 3].set_title("Segmentation Classes")
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Visualization saved to {save_path}")
    plt.show()

def test_dataloader(dataset: ViNTSegmentationDataset, batch_size: int = 4):
    """Test DataLoader with batching"""
    print("\n" + "="*60)
    print("TEST 4: DataLoader Batching")
    print("="*60)
    
    try:
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,  # Use 0 for testing to avoid multiprocessing issues
            drop_last=False
        )
        
        # Get one batch
        batch = next(iter(dataloader))
        
        print(f"Batch with size {batch_size}:")
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                print(f"  - {key}: shape={value.shape}, dtype={value.dtype}")
        
        print("✓ DataLoader working correctly!")
        return dataloader
        
    except Exception as e:
        print(f"✗ DataLoader failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_model_forward(config: Dict[str, Any], dataloader: DataLoader):
    """Test model forward pass"""
    print("\n" + "="*60)
    print("TEST 5: Model Forward Pass")
    print("="*60)
    
    try:
        # Create model
        model = SegmentationViNT(
            context_size=config['context_size'],
            len_traj_pred=config['len_traj_pred'],
            learn_angle=config['learn_angle'],
            obs_encoder=config['obs_encoder'],
            obs_encoding_size=config['obs_encoding_size'],
            late_fusion=config['late_fusion'],
            mha_num_attention_heads=config['mha_num_attention_heads'],
            mha_num_attention_layers=config['mha_num_attention_layers'],
            mha_ff_dim_factor=config['mha_ff_dim_factor'],
            num_seg_classes=config.get('num_seg_classes', 5),
            seg_model_type=config.get('seg_model_type', 'unet'),
            seg_encoder=config.get('seg_encoder', 'resnet34'),
            fusion_type=config.get('fusion_type', 'cross_attention'),
            freeze_vint_encoder=True,
            use_semantic_goals=config.get('use_semantic_goals', True),
        )
        
        print(f"✓ Model created successfully!")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  - Total parameters: {total_params:,}")
        print(f"  - Trainable parameters: {trainable_params:,}")
        
        # Test forward pass
        model.eval()
        batch = next(iter(dataloader))
        
        with torch.no_grad():
            # CRITICAL: Model should NOT receive ground truth segmentation
            outputs = model(
                batch['obs_images'],
                batch['goal_images']
            )
        
        print("\nModel outputs:")
        for key, value in outputs.items():
            if value is not None:
                print(f"  - {key}: shape={value.shape}, dtype={value.dtype}")
        
        # Verify output shapes
        batch_size = batch['obs_images'].shape[0]
        expected_shapes = {
            'dist_pred': (batch_size, 1),
            'action_pred': (batch_size, config['len_traj_pred'], 2),
            'obs_seg_logits': (batch_size, config['num_seg_classes'], *config['image_size']),
        }
        
        print("\nShape verification:")
        for key, expected_shape in expected_shapes.items():
            if key in outputs and outputs[key] is not None:
                actual_shape = tuple(outputs[key].shape)
                if actual_shape == expected_shape:
                    print(f"  ✓ {key}: {actual_shape}")
                else:
                    print(f"  ✗ {key}: expected {expected_shape}, got {actual_shape}")
        
        print("✓ Model forward pass successful!")
        return model, outputs
        
    except Exception as e:
        print(f"✗ Model forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def test_segmentation_quality(outputs: Dict[str, torch.Tensor], dataset: ViNTSegmentationDataset,
                            save_path: str = "test_model_segmentation.png"):
    """Visualize model's segmentation predictions"""
    print("\n" + "="*60)
    print("TEST 6: Model Segmentation Quality")
    print("="*60)
    
    # Get predicted segmentation
    seg_logits = outputs['obs_seg_logits'][0]  # First sample in batch
    seg_pred = torch.argmax(seg_logits, dim=0)
    
    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    # Predicted segmentation
    seg_rgb = visualize_segmentation(seg_pred, dataset.class_names)
    axes[0].imshow(seg_rgb)
    axes[0].set_title("Model's Predicted Segmentation")
    axes[0].axis('off')
    
    # Class probabilities
    seg_probs = torch.softmax(seg_logits, dim=0)
    axes[1].bar(range(len(dataset.class_names)), 
                [seg_probs[i].mean().item() for i in range(len(dataset.class_names))])
    axes[1].set_xticks(range(len(dataset.class_names)))
    axes[1].set_xticklabels(dataset.class_names, rotation=45)
    axes[1].set_ylabel("Mean Probability")
    axes[1].set_title("Class Distribution")
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Model segmentation saved to {save_path}")
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Test SegmentationViNT data pipeline")
    parser.add_argument("--config", "-c", default="config/segmentation_vint.yaml",
                       help="Path to config file")
    parser.add_argument("--samples", "-n", type=int, default=3,
                       help="Number of samples to test")
    parser.add_argument("--batch_size", "-b", type=int, default=4,
                       help="Batch size for testing")
    parser.add_argument("--save_dir", "-s", default="test_outputs",
                       help="Directory to save test outputs")
    args = parser.parse_args()
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Load config
    print(f"Loading config from: {args.config}")
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    # Run tests
    print("\n" + "="*60)
    print("STARTING SEGMENTATION DATASET TESTS")
    print("="*60)
    
    # Test 1: Dataset loading
    dataset = test_dataset_loading(config)
    if dataset is None:
        print("\n✗ Dataset loading failed! Exiting...")
        return
    
    # Test 2-3: Load and visualize samples
    for i in range(min(args.samples, len(dataset))):
        sample = test_data_sample(dataset, idx=i)
        if sample is not None:
            save_path = os.path.join(args.save_dir, f"sample_{i}_visualization.png")
            visualize_sample(sample, dataset, save_path)
    
    # Test 4: DataLoader
    dataloader = test_dataloader(dataset, batch_size=args.batch_size)
    if dataloader is None:
        print("\n✗ DataLoader test failed! Exiting...")
        return
    
    # Test 5-6: Model forward pass
    model, outputs = test_model_forward(config, dataloader)
    if model is not None and outputs is not None:
        save_path = os.path.join(args.save_dir, "model_segmentation.png")
        test_segmentation_quality(outputs, dataset, save_path)
    
    print("\n" + "="*60)
    print("ALL TESTS COMPLETED!")
    print("="*60)
    print(f"\nCheck '{args.save_dir}/' for visualization outputs")

if __name__ == "__main__":
    main()