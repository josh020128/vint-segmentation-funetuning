"""
Generate pseudo-segmentation labels for navigation dataset.
Saves both raw masks for training and colored visualizations.
"""
import torch
from transformers import OneFormerProcessor, OneFormerForUniversalSegmentation
from PIL import Image
import numpy as np
import argparse
import os
from typing import Tuple
from tqdm import tqdm

def get_navigation_class_map() -> Tuple[np.ndarray, np.ndarray]:
    """
    Defines mapping from ADE20K classes to navigation classes and colors.
    
    Navigation classes:
    0: floor (navigable surfaces)
    1: wall (boundaries)
    2: door (passable barriers)
    3: furniture (obstacles)
    4: unknown (everything else)
    """
    unknown_class_idx = 4
    
    # Initialize all 150 ADE20K classes to 'unknown'
    lookup_table = np.full(150, fill_value=unknown_class_idx, dtype=np.uint8)
    
    # Map ADE20K indices to navigation classes
    ade20k_to_nav = {
        # Floors/walkable surfaces
        3: 0,   # floor
        6: 0,   # road
        11: 0,  # sidewalk
        13: 0,  # earth/ground
        29: 0,  # rug
        53: 0,  # path
        
        # Walls/barriers
        0: 1,   # wall
        1: 1,   # building
        5: 1,   # ceiling
        25: 1,  # fence
        
        # Doors
        14: 2,  # door
        
        # Furniture/obstacles
        7: 3,   # bed
        10: 3,  # cabinet
        15: 3,  # table
        19: 3,  # chair
        24: 3,  # sofa
        31: 3,  # desk
        33: 3,  # counter
        34: 3,  # shelf
        65: 3,  # armchair
        
        # Other important classes to unknown
        2: 4,   # sky
        4: 4,   # tree
        8: 4,   # window
        9: 4,   # grass
        12: 4,  # person (could be dynamic)
    }
    
    for ade_class, nav_class in ade20k_to_nav.items():
        if ade_class < 150:
            lookup_table[ade_class] = nav_class
    
    # Colors for visualization (RGB)
    colors = np.array([
        [0, 255, 0],    # 0: floor - green
        [255, 0, 0],    # 1: wall - red
        [255, 255, 0],  # 2: door - yellow
        [0, 0, 255],    # 3: furniture - blue
        [128, 128, 128] # 4: unknown - gray
    ], dtype=np.uint8)
    
    return lookup_table, colors

def process_single_image(
    model, 
    processor, 
    image_path: str, 
    lookup_table: np.ndarray, 
    device: torch.device
) -> np.ndarray:
    """Process a single image to generate segmentation mask"""
    
    # Load image
    image = Image.open(image_path).convert("RGB")
    original_size = image.size  # (width, height)
    
    # Preprocess for semantic segmentation
    inputs = processor(
        images=[image], 
        task_inputs=["semantic"], 
        return_tensors="pt"
    ).to(device)
    
    # Generate prediction
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Post-process to original resolution
    ade_preds = processor.post_process_semantic_segmentation(
        outputs, 
        target_sizes=[(original_size[1], original_size[0])]  # (H, W)
    )
    
    # Convert to navigation classes
    ade_pred_np = ade_preds[0].cpu().numpy()
    nav_seg_np = lookup_table[np.clip(ade_pred_np, 0, 149)]
    
    return nav_seg_np

def generate_pseudo_labels_for_dataset(
    data_folder: str,
    raw_output_folder: str,
    color_output_folder: str,
    model_size: str = "tiny"
):
    """
    Generate pseudo-segmentation labels using OneFormer.
    
    Args:
        data_folder: Path to RGB images
        raw_output_folder: Path to save raw masks (0-4 values) for training
        color_output_folder: Path to save colored visualizations
        model_size: OneFormer model size (tiny/base/large)
    """
    
    # Load OneFormer model
    checkpoint = f"shi-labs/oneformer_ade20k_swin_{model_size}"
    print(f"Loading model: {checkpoint}")
    
    processor = OneFormerProcessor.from_pretrained(checkpoint)
    model = OneFormerForUniversalSegmentation.from_pretrained(
        checkpoint, 
        use_safetensors=True
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    print(f"Model loaded on device: {device}")
    
    # Get mapping and colors
    lookup_table, colors = get_navigation_class_map()
    
    # Find all images
    all_image_paths = []
    for root, _, files in os.walk(data_folder):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                if '_seg' not in file:  # Skip existing segmentation masks
                    all_image_paths.append(os.path.join(root, file))
    
    print(f"Found {len(all_image_paths)} images to process")
    print(f"Raw masks will be saved to: {raw_output_folder}")
    print(f"Colored visualizations will be saved to: {color_output_folder}")
    
    # Process each image
    success_count = 0
    error_count = 0
    
    for img_path in tqdm(all_image_paths, desc="Generating Pseudo-Labels"):
        try:
            # Generate segmentation mask
            nav_seg_np = process_single_image(
                model, processor, img_path, lookup_table, device
            )
            
            # Prepare output paths
            rel_path = os.path.relpath(img_path, data_folder)
            base_name = os.path.splitext(rel_path)[0]
            
            # Path for raw mask (for training)
            raw_save_path = os.path.join(raw_output_folder, base_name + "_seg.png")
            os.makedirs(os.path.dirname(raw_save_path), exist_ok=True)
            
            # Path for colored visualization
            color_save_path = os.path.join(color_output_folder, base_name + "_seg.png")
            os.makedirs(os.path.dirname(color_save_path), exist_ok=True)
            
            # Save raw mask (grayscale with values 0-4)
            # This is what the training code expects!
            Image.fromarray(nav_seg_np.astype(np.uint8), mode='L').save(raw_save_path)
            
            # Save colored visualization for human inspection
            colored_seg = colors[nav_seg_np]
            Image.fromarray(colored_seg, mode='RGB').save(color_save_path)
            
            success_count += 1
            
            # Print sample statistics every 500 images
            if success_count % 500 == 0:
                unique, counts = np.unique(nav_seg_np, return_counts=True)
                total_pixels = nav_seg_np.size
                print(f"\nSample {success_count} statistics:")
                class_names = ['floor', 'wall', 'door', 'furniture', 'unknown']
                for cls_id, count in zip(unique, counts):
                    if cls_id < len(class_names):
                        print(f"  {class_names[cls_id]}: {count/total_pixels*100:.1f}%")
                
        except Exception as e:
            print(f"\nError processing {img_path}: {e}")
            error_count += 1
            continue
    
    print(f"\n{'='*60}")
    print(f"Pseudo-label generation complete!")
    print(f"Successfully processed: {success_count}/{len(all_image_paths)} images")
    if error_count > 0:
        print(f"Errors encountered: {error_count}")
    print(f"Raw masks saved in: {raw_output_folder}")
    print(f"Colored visualizations saved in: {color_output_folder}")
    print(f"{'='*60}")

def verify_generated_masks(raw_folder: str, num_samples: int = 5):
    """Verify the generated raw masks"""
    
    print("\n" + "="*60)
    print("Verifying Generated Masks")
    print("="*60)
    
    mask_files = []
    for root, _, files in os.walk(raw_folder):
        for file in files:
            if file.endswith('_seg.png'):
                mask_files.append(os.path.join(root, file))
    
    if not mask_files:
        print("No masks found!")
        return
    
    print(f"Found {len(mask_files)} masks total")
    print(f"\nChecking {min(num_samples, len(mask_files))} samples:")
    
    class_names = ['floor', 'wall', 'door', 'furniture', 'unknown']
    
    for i in range(min(num_samples, len(mask_files))):
        mask = np.array(Image.open(mask_files[i]))
        unique, counts = np.unique(mask, return_counts=True)
        
        print(f"\n{i+1}. {os.path.basename(mask_files[i])}")
        print(f"   Shape: {mask.shape}")
        print(f"   Classes present: {unique.tolist()}")
        
        total = mask.size
        for cls_id, count in zip(unique, counts):
            if cls_id < len(class_names):
                print(f"   - {class_names[cls_id]}: {count} pixels ({count/total*100:.1f}%)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate pseudo-segmentation labels")
    parser.add_argument(
        "--data_folder", 
        default="/home/airlab/vint_fuse3/datasets/scand_processed",
        help="Path to source dataset with RGB images"
    )
    parser.add_argument(
        "--raw_output_folder", 
        default="/home/airlab/vint_fuse3/datasets/scand_seg",
        help="Path to save raw segmentation masks for training"
    )
    parser.add_argument(
        "--color_output_folder", 
        default="/home/airlab/vint_fuse3/datasets/scand_color_seg",
        help="Path to save colored visualizations"
    )
    parser.add_argument(
        "--model_size", 
        type=str, 
        default="tiny", 
        choices=["tiny", "base", "large"],
        help="OneFormer model size"
    )
    parser.add_argument(
        "--verify", 
        action="store_true",
        help="Verify generated masks after processing"
    )
    
    args = parser.parse_args()
    
    # Generate pseudo labels
    generate_pseudo_labels_for_dataset(
        args.data_folder,
        args.raw_output_folder,
        args.color_output_folder,
        args.model_size
    )
    
    # Verify if requested
    if args.verify:
        verify_generated_masks(args.raw_output_folder)