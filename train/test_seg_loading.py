import torch
import yaml
from vint_train.data.segmentation_dataset import ViNTSegmentationDataset

# Load config
with open("config/segmentation_vint.yaml", "r") as f:
    config = yaml.safe_load(f)

# Create dataset
dataset_name = list(config["datasets"].keys())[0]
data_config = config["datasets"][dataset_name]

dataset = ViNTSegmentationDataset(
    data_folder=data_config["data_folder"],
    data_split_folder=data_config["train"],
    dataset_name=dataset_name,
    image_size=tuple(config["image_size"]),
    seg_data_folder=data_config.get("seg_data_folder"),
    use_pseudo_labels=config.get("use_pseudo_labels", True),
    seg_model_name=config.get("seg_model_name", "scand"),
    # Add other required params
    waypoint_spacing=1,
    min_dist_cat=0,
    max_dist_cat=20,
    min_action_distance=1,
    max_action_distance=10,
    negative_mining=True,
    len_traj_pred=8,
    learn_angle=False,
    context_size=5,
)

print(f"\nDataset created with {len(dataset)} samples")

# Test loading a sample
print("\nLoading sample 0...")
sample = dataset[0]

print("\nSample contents:")
for key, val in sample.items():
    if isinstance(val, torch.Tensor):
        print(f"  {key}: shape={val.shape}, dtype={val.dtype}")
        if 'seg' in key:
            unique = torch.unique(val)
            print(f"    Unique values: {unique.tolist()}")

print("\n✅ Segmentation loading is working!")