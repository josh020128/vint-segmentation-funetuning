import torch
# <<< FIXED: Import the specific OneFormerProcessor >>>
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation, Mask2FormerForUniversalSegmentation, OneFormerProcessor, OneFormerForUniversalSegmentation
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from typing import Dict
import segmentation_models_pytorch as smp
import torchvision.transforms as T

def get_navigation_class_map() -> (np.ndarray, Dict):
    """
    Defines and returns the mapping from ADE20K classes to our navigation classes.
    """
    # Our 5 navigation classes: 0:floor, 1:wall, 2:door, 3:furniture, 4:unknown
    unknown_class_idx = 4
    
    # Initialize a lookup table where all 150 ADE20K classes map to 'unknown' by default
    lookup_table = np.full(150, fill_value=unknown_class_idx, dtype=np.uint8)
    
    # Define the corrected mappings (ADE20K index -> Our navigation class index)
    ade20k_to_nav = {
        # Floors/walkable surfaces
        3: 0, 6: 0, 11: 0, 13: 0, 29: 0, 53: 0,
        # Walls/barriers
        0: 1, 1: 1, 5: 1, 25: 1,
        # Doors
        14: 2,
        # Furniture/obstacles
        7: 3, 10: 3, 15: 3, 19: 3, 24: 3, 31: 3, 33: 3, 65: 3,
    }
    for ade_class, nav_class in ade20k_to_nav.items():
        if ade_class < 150:
            lookup_table[ade_class] = nav_class
            
    # Define colors for our navigation classes for visualization
    colors = np.array([
        [0, 255, 0],    # 0: floor (green)
        [255, 0, 0],    # 1: wall (red)
        [255, 255, 0],  # 2: door (yellow)
        [0, 0, 255],    # 3: furniture (blue)
        [128, 128, 128] # 4: unknown (gray)
    ], dtype=np.uint8)
    
    return lookup_table, colors

def test_model_on_image(model_name: str, image_path: str, save_path: str):
    """
    Loads a pretrained segmentation model, runs it on a single image,
    and saves a side-by-side visualization.
    """
    # 1. Load the specified model and its preprocessor
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model '{model_name}' on device: {device}...")
    
    if model_name.lower() == 'segformer':
        checkpoint = "nvidia/segformer-b2-finetuned-ade-512-512"
        processor = AutoImageProcessor.from_pretrained(checkpoint)
        model = SegformerForSemanticSegmentation.from_pretrained(checkpoint, use_safetensors=True).to(device).eval()
    elif model_name.lower() == 'mask2former':
        checkpoint = "facebook/mask2former-swin-base-ade-semantic"
        processor = AutoImageProcessor.from_pretrained(checkpoint)
        model = Mask2FormerForUniversalSegmentation.from_pretrained(checkpoint, use_safetensors=True).to(device).eval()
    elif model_name.lower() == 'oneformer':
        checkpoint = "shi-labs/oneformer_ade20k_swin_tiny"
        # <<< FIXED: Use the full OneFormerProcessor which includes the tokenizer >>>
        processor = OneFormerProcessor.from_pretrained(checkpoint)
        model = OneFormerForUniversalSegmentation.from_pretrained(checkpoint, use_safetensors=True).to(device).eval()
    elif model_name.lower() == 'unet':
        model = smp.Unet(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            in_channels=3,
            classes=150,
        ).to(device).eval()
        processor = None 
    else:
        raise ValueError(f"Unknown model name: {model_name}. Choose 'segformer', 'mask2former', 'oneformer', or 'unet'.")

    # 2. Get the class mapping and visualization colors
    lookup_table, colors = get_navigation_class_map()

    # 3. Load and process the image
    image = Image.open(image_path).convert("RGB")
    original_size = image.size # (width, height)
    
    # 4. Run model inference
    with torch.no_grad():
        if processor: # For segformer, mask2former, and oneformer
            if model_name.lower() == 'oneformer':
                # <<< FIXED: Use the correct argument name 'task_inputs' >>>
                inputs = processor(images=image, task_inputs=["semantic"], return_tensors="pt").to(device)
            else:
                inputs = processor(images=image, return_tensors="pt").to(device)
            
            outputs = model(**inputs)
            
        else: # Manual preprocessing and forward pass for smp U-Net
            h, w = original_size[::-1]
            target_h = (h // 32) * 32
            target_w = (w // 32) * 32
            image_resized = image.resize((target_w, target_h))
            
            img_tensor = T.ToTensor()(image_resized)
            normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            img_tensor = normalize(img_tensor)
            
            input_tensor = img_tensor.unsqueeze(0).to(device)
            outputs = model(input_tensor)

    # 5. Post-process the output to get the final segmentation map
    if model_name.lower() in ['segformer', 'unet']:
        logits = outputs.logits.cpu() if hasattr(outputs, 'logits') else outputs.cpu()
        upsampled_logits = torch.nn.functional.interpolate(
            logits, size=original_size[::-1], mode="bilinear", align_corners=False
        )
        ade_pred = upsampled_logits.argmax(dim=1)[0].numpy()
    else: # For mask2former and oneformer
        # The post-processing is the same for both Mask2Former and OneFormer
        ade_pred_list = processor.post_process_semantic_segmentation(outputs, target_sizes=[original_size[::-1]])
        ade_pred = ade_pred_list[0].cpu().numpy()

    # 6. Convert the ADE20K class predictions to our navigation classes
    nav_seg = lookup_table[ade_pred]
    
    # 7. Create a colored visualization of the navigation mask
    colored_nav_mask = colors[nav_seg]

    # 8. Create and save the side-by-side plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    axes[0].imshow(image)
    axes[0].set_title("Original RGB Image")
    axes[0].axis('off')
    
    axes[1].imshow(colored_nav_mask)
    axes[1].set_title(f"{model_name}'s Predicted Nav-Mask")
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ Visualization saved successfully to: {save_path}")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test a segmentation model on a single image.")
    parser.add_argument(
        "--model", 
        type=str, 
        required=True, 
        # <<< CHANGED: Added 'unet' to choices >>>
        choices=['segformer', 'mask2former', 'unet', 'oneformer'],
        help="The segmentation model to test."
    )
    parser.add_argument(
        "--image", 
        type=str, 
        default="46.jpg", 
        help="Path to the input image file."
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default="test_output.png", 
        help="Path to save the output visualization."
    )
    args = parser.parse_args()
    
    if not os.path.exists(args.image):
        print(f"Error: Image path not found at '{args.image}'")
    else:
        test_model_on_image(args.model, args.image, args.output)
