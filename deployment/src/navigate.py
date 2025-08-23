from __future__ import annotations

import argparse
import torch.nn as nn
import os
import time
from collections import deque
from pathlib import Path
import segmentation_models_pytorch as smp
from typing import Deque, List, Dict, Tuple
import torch.nn.functional as F

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from PIL import Image as PILImage
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Float32MultiArray
import torch
import yaml
import torchvision.models as models

# --- Transformer/Segmentation Imports ---
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation

# --- Project-Specific Imports ---
from utils import msg_to_pil, to_numpy, transform_images
from vint_train.models.base_model import BaseModel
from vint_train.models.vint.self_attention import MultiLayerDecoder

# ---------------------------------------------------------------------------
# CONFIG & CONSTANTS
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROBOT_CONFIG_PATH = PROJECT_ROOT / "deployment/config/robot.yaml"
MODEL_CONFIG_PATH = PROJECT_ROOT / "deployment/config/models.yaml"
TOPOMAP_IMAGES_DIR = PROJECT_ROOT / "deployment/topomaps/images"

with open(ROBOT_CONFIG_PATH, "r") as f:
    ROBOT_CONF = yaml.safe_load(f)
MAX_V = ROBOT_CONF["max_v"]
MAX_W = ROBOT_CONF["max_w"]
RATE = ROBOT_CONF["frame_rate"]  # Hz

# ---------------------------------------------------------------------------
# LOCAL MODEL DEFINITIONS (Self-Contained)
# ---------------------------------------------------------------------------
def build_vint_model(**kwargs) -> nn.Module:
    """Build a ViNT model or placeholder"""
    try:
        from vint_train.models.vint.vint import ViNT
        return ViNT(**kwargs)
    except:
        # Simplified placeholder for testing
        class SimpleViNT(nn.Module):
            def __init__(self, obs_encoding_size=512, context_size=5, **kwargs):
                super().__init__()
                self.obs_encoding_size = obs_encoding_size
                self.context_size = context_size
                
                # Simple encoders
                self.obs_encoder = nn.Sequential(
                    nn.Conv2d(3 * context_size, 64, 7, stride=2, padding=3),
                    nn.BatchNorm2d(64),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten()
                )
                self.goal_encoder = nn.Sequential(
                    nn.Conv2d(3, 64, 7, stride=2, padding=3),
                    nn.BatchNorm2d(64),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten()
                )
                self.compress_obs_enc = nn.Linear(64, obs_encoding_size)
                self.compress_goal_enc = nn.Linear(64, obs_encoding_size)
                
        return SimpleViNT(**kwargs)
    
def create_encoder(encoder_name: str, in_channels: int, pretrained: bool = True) -> Tuple[nn.Module, int]:
    """Creates a visual encoder (ResNet) and modifies it for our task."""
    if "resnet" in encoder_name:
        if encoder_name == "resnet34":
            encoder = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None)
            num_features = encoder.fc.in_features
        else:
            raise NotImplementedError(f"ResNet variant '{encoder_name}' not supported.")
        encoder.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        encoder.fc = nn.Identity()
        return encoder, num_features
    else:
        raise ValueError(f"Encoder '{encoder_name}' is not supported. Please use a ResNet variant.")

class ViNT(BaseModel):
    """Locally defined core ViNT model for RGB processing."""
    def __init__(
        self, context_size: int, len_traj_pred: int, learn_angle: bool,
        obs_encoder: str, obs_encoding_size: int, **kwargs
    ) -> None:
        super(ViNT, self).__init__(context_size, len_traj_pred, learn_angle)
        self.obs_encoding_size = obs_encoding_size
        self.obs_encoder, num_obs_features = create_encoder(obs_encoder, in_channels=3 * self.context_size)
        self.goal_encoder, num_goal_features = create_encoder(obs_encoder, in_channels=3)
        self.compress_obs_enc = nn.Linear(num_obs_features, self.obs_encoding_size) if num_obs_features != self.obs_encoding_size else nn.Identity()
        self.compress_goal_enc = nn.Linear(num_goal_features, self.obs_encoding_size) if num_goal_features != self.obs_encoding_size else nn.Identity()

    def forward(self, obs_img: torch.Tensor, goal_img: torch.Tensor) -> Dict[str, torch.Tensor]:
        # This forward pass is a placeholder. The wrapper calls the components directly.
        batch_size = obs_img.shape[0]
        dummy_dist = torch.zeros((batch_size, 1), device=obs_img.device)
        dummy_actions = torch.zeros((batch_size, self.len_trajectory_pred, self.num_action_params), device=obs_img.device)
        return {'dist_pred': dummy_dist, 'action_pred': dummy_actions}

class SegmentationViNT(nn.Module):
    """Locally defined dual-input ViNT model."""
    def __init__(
        self, context_size: int, len_traj_pred: int, learn_angle: bool,
        obs_encoder: str, obs_encoding_size: int, num_seg_classes: int,
        seg_feature_dim: int = 256, **kwargs
    ):
        super().__init__()
        self.len_traj_pred = len_traj_pred
        self.learn_angle = learn_angle
        self.num_seg_classes = num_seg_classes
        self.obs_encoding_size = obs_encoding_size
        vint_args = {
            'context_size': context_size, 'len_traj_pred': len_traj_pred,
            'learn_angle': learn_angle, 'obs_encoder': obs_encoder,
            'obs_encoding_size': obs_encoding_size, **kwargs
        }
        self.vint_model = build_vint_model(**vint_args)
        self.seg_encoder = nn.Sequential(
            nn.Conv2d(num_seg_classes, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, seg_feature_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(seg_feature_dim), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(seg_feature_dim, obs_encoding_size),
            nn.LayerNorm(obs_encoding_size),
        )
        fused_dim = obs_encoding_size * 2
        num_action_outputs = len_traj_pred * (3 if learn_angle else 2)
        self.action_predictor = nn.Sequential(
            nn.Linear(fused_dim, 512), nn.LayerNorm(512),
            nn.ReLU(inplace=True), nn.Dropout(0.1),
            nn.Linear(512, num_action_outputs),
        )
        dist_input_dim = fused_dim + obs_encoding_size
        self.dist_predictor = nn.Sequential(
            nn.Linear(dist_input_dim, 256), nn.LayerNorm(256),
            nn.ReLU(inplace=True), nn.Dropout(0.1),
            nn.Linear(256, 1),
        )

    def forward(self, obs_images: torch.Tensor, goal_images: torch.Tensor, obs_seg_masks: torch.Tensor) -> Dict[str, torch.Tensor]:
        batch_size = obs_images.shape[0]
        obs_encoding = self.vint_model.compress_obs_enc(self.vint_model.obs_encoder(obs_images))
        goal_encoding = self.vint_model.compress_goal_enc(self.vint_model.goal_encoder(goal_images))
        seg_features = self.seg_encoder(obs_seg_masks)
        fused_obs = torch.cat([obs_encoding, seg_features], dim=1)
        action_pred = self.action_predictor(fused_obs)
        action_pred = action_pred.view(batch_size, self.len_traj_pred, -1)
        if not self.learn_angle:
            action_pred = torch.cumsum(action_pred, dim=1)
        combined_for_dist = torch.cat([fused_obs, goal_encoding], dim=1)
        dist_pred = self.dist_predictor(combined_for_dist)
        return {'action_pred': action_pred, 'dist_pred': dist_pred}

# ---------------------------------------------------------------------------
# MODEL LOADING
# ---------------------------------------------------------------------------

def load_model_from_checkpoint(model_path: str, config: dict, device: torch.device) -> nn.Module:
    """Loads the appropriate model and filters config keys."""
    model_type = config["model_type"]
    known_model_args = {
        'context_size', 'len_traj_pred', 'learn_angle', 'obs_encoder',
        'obs_encoding_size', 'mha_num_attention_heads', 'mha_num_attention_layers',
        'mha_ff_dim_factor', 'num_seg_classes', 'seg_feature_dim'
    }
    model_config = {key: config[key] for key in known_model_args if key in config}
    if 'model_params' in config and isinstance(config['model_params'], dict):
        model_config.update(config['model_params'])

    if model_type == "segmentation_vint":
        model = SegmentationViNT(**model_config)
    else:
        raise ValueError(f"This script is for 'segmentation_vint', but found '{model_type}' in config.")
    
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    if hasattr(state_dict, 'state_dict'): state_dict = state_dict.state_dict()
    if all(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k[7:]: v for k, v in state_dict.items()}

    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    return model

def _load_model(model_name: str, device: torch.device):
    """Loads model and params from deployment config."""
    with open(MODEL_CONFIG_PATH, "r") as f: model_paths = yaml.safe_load(f)
    mconf_path = model_paths[model_name]["config_path"]
    ckpt_path = model_paths[model_name]["ckpt_path"]
    with open(mconf_path, "r") as f: model_params = yaml.safe_load(f)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Model weights not found at {ckpt_path}")
    print(f"Loading model from {ckpt_path}")
    model = load_model_from_checkpoint(ckpt_path, model_params, device).to(device).eval()
    return model, model_params

# ---------------------------------------------------------------------------
# ROS2 NODE
# ---------------------------------------------------------------------------

class NavigationNode(Node):
    """ROS2 Node for running the dual-input ViNT model."""

    def __init__(self, args: argparse.Namespace):
        super().__init__("navigation")
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.get_logger().info(f"Using device: {self.device}")

        # --- Load Finetuned Dual-Input Model ---
        self.model, self.model_params = _load_model(args.model, self.device)
        self.context_size: int = self.model_params["context_size"]
        
        # --- Load Standalone Mask2Former for Segmentation ---
        self.get_logger().info("Loading Mask2Former for real-time segmentation...")
        m2f_checkpoint = "facebook/mask2former-swin-base-ade-semantic"
        self.mask2former_processor = AutoImageProcessor.from_pretrained(m2f_checkpoint)
        self.mask2former_model = Mask2FormerForUniversalSegmentation.from_pretrained(m2f_checkpoint).to(self.device).eval()
        self.setup_class_map()
        self.get_logger().info("Mask2Former model loaded successfully.")

        # --- ROS2 & CV Bridge Setup ---
        self.bridge = CvBridge()
        self.context_queue = deque(maxlen=self.context_size + 1)
        self.last_ctx_time = self.get_clock().now()
        self.ctx_dt = 0.25

        # --- Topomap Setup ---
        self.topomap: List[PILImage.Image] = self._load_topomap(args.dir)
        self.goal_node = (len(self.topomap) - 1) if args.goal_node == -1 else args.goal_node
        self.closest_node = 0

        # --- ROS Interfaces ---
        image_topic, waypoint_topic, _ = self.get_robot_topics()
        self.create_subscription(Image, image_topic, self._image_cb, 1)
        self.waypoint_pub = self.create_publisher(Float32MultiArray, waypoint_topic, 1)
        self.viz_pub = self.create_publisher(Image, "navigation_viz", 1)
        self.seg_viz_pub = self.create_publisher(Image, "navigation_seg_viz", 1)
        self.subgoal_pub = self.create_publisher(Image, "navigation_subgoal", 1)
        self.goal_pub_img = self.create_publisher(Image, "navigation_goal", 1)
        self.goal_pub = self.create_publisher(Bool, "/topoplan/reached_goal", 1)
        self.create_timer(1.0 / RATE, self._timer_cb)
        
        # --- Visualization & Color Maps ---
        self.seg_colors = np.array([
            [0, 255, 0], [255, 0, 0], [255, 255, 0], [0, 0, 255], [128, 128, 128]
        ], dtype=np.uint8)
        self.get_logger().info("Navigation node initialised. Waiting for images…")

    def get_robot_topics(self) -> Tuple[str, str, str]:
        if self.args.robot == "locobot": return "/robot1/camera/image", "/robot1/waypoint", "/robot1/sampled_actions"
        elif self.args.robot == "robomaster": return "/camera/image_color", "/robot3/waypoint", "/robot3/sampled_actions"
        elif self.args.robot == "turtlebot4": return "/robot2/oakd/rgb/preview/image_raw", "/robot2/waypoint", "/robot2/sampled_actions"
        else: raise ValueError(f"Unknown robot type: {self.args.robot}")

    def _load_topomap(self, subdir: str) -> List[PILImage.Image]:
        dpath = TOPOMAP_IMAGES_DIR / subdir
        if not dpath.exists(): raise FileNotFoundError(f"Topomap directory {dpath} does not exist")
        img_files = sorted(os.listdir(dpath), key=lambda x: int(os.path.splitext(x)[0]))
        return [PILImage.open(dpath / f) for f in img_files]

    def setup_class_map(self):
        """Creates a lookup table to map ADE20K classes to our 5 navigation classes."""
        self.lookup_table = np.full(150, fill_value=4, dtype=np.uint8) # Default to unknown
        ade2_to_nav = {3: 0, 6: 0, 11: 0, 13: 0, 29: 0, 53: 0, 0: 1, 1: 1, 5: 1, 25: 1, 14: 2, 7: 3, 10: 3, 15: 3, 19: 3, 24: 3, 31: 3, 33: 3, 65: 3}
        for ade_class, nav_class in ade2_to_nav.items():
            if ade_class < 150: self.lookup_table[ade_class] = nav_class

    def _image_cb(self, msg: Image):
        now = self.get_clock().now()
        if (now - self.last_ctx_time).nanoseconds < self.ctx_dt * 1e9: return
        self.context_queue.append(msg_to_pil(msg))
        self.last_ctx_time = now

    def _timer_cb(self):
        # Wait for enough frames to fill the context
        if len(self.context_queue) <= self.context_size: return

        # --- 1. Get Segmentation Mask from Mask2Former ---
        latest_image = self.context_queue[-1]
        obs_seg_mask = self.run_mask2former(latest_image)

        # --- 2. Find Closest Node ---
        start = max(self.closest_node - self.args.radius, 0)
        end = min(self.closest_node + self.args.radius + 1, len(self.topomap) - 1)
        
        # <<< RUNTIME ERROR FIX: Prepare exactly `context_size` frames for the model >>>
        obs_images_pil = list(self.context_queue)[-self.context_size:]
        obs_tensor = transform_images(obs_images_pil, self.model_params["image_size"])
        
        batch_obs_tensor = obs_tensor.repeat(end - start + 1, 1, 1, 1).to(self.device)
        batch_goal_imgs = [transform_images(self.topomap[i], self.model_params["image_size"]) for i in range(start, end + 1)]
        batch_goal_tensor = torch.cat(batch_goal_imgs, dim=0).to(self.device)
        batch_seg_mask_tensor = obs_seg_mask.repeat(end - start + 1, 1, 1, 1).to(self.device)

        # --- 3. Run Dual-Input Model Inference ---
        with torch.no_grad():
            outputs = self.model(batch_obs_tensor, batch_goal_tensor, batch_seg_mask_tensor)
            distances_np = to_numpy(outputs['dist_pred'])
            waypoints_np = to_numpy(outputs['action_pred'])

        # --- 4. Select Subgoal and Waypoint ---
        min_dist_idx = np.argmin(distances_np)
        if distances_np[min_dist_idx] > self.args.close_threshold:
            chosen_waypoint_idx = min_dist_idx
            self.closest_node = start + min_dist_idx
        else:
            chosen_waypoint_idx = min(min_dist_idx + 1, len(waypoints_np) - 1)
            self.closest_node = min(start + min_dist_idx + 1, self.goal_node)
        selected_trajectory = waypoints_np[chosen_waypoint_idx]
        chosen_waypoint = selected_trajectory[self.args.waypoint]

        # --- 5. Publish ---
        self._publish_waypoint(chosen_waypoint)
        self._publish_viz_image(np.expand_dims(selected_trajectory, axis=0))
        sg_global_idx = start + chosen_waypoint_idx
        self._publish_goal_images(self.topomap[sg_global_idx], self.topomap[self.goal_node])
        reached_goal = bool(self.closest_node == self.goal_node)
        self.goal_pub.publish(Bool(data=reached_goal))
        if reached_goal: self.get_logger().info("Reached goal! Stopping...")

    def run_mask2former(self, image: PILImage.Image) -> torch.Tensor:
        """Runs Mask2Former, converts to nav classes, and returns a one-hot tensor."""
        with torch.no_grad():
            model_img_size = self.model_params["image_size"]
            inputs = self.mask2former_processor(images=image, return_tensors="pt").to(self.device)
            outputs = self.mask2former_model(**inputs)
            ade_preds = self.mask2former_processor.post_process_semantic_segmentation(outputs, target_sizes=[model_img_size])[0]
            nav_seg = self.lookup_table[ade_preds.cpu().numpy()]
            
            # Visualize the colored mask
            colored_mask = self.seg_colors[nav_seg]
            colored_mask_bgr = cv2.cvtColor(colored_mask, cv2.COLOR_RGB2BGR)
            seg_msg = self.bridge.cv2_to_imgmsg(colored_mask_bgr, encoding="bgr8")
            seg_msg.header.stamp = self.get_clock().now().to_msg()
            self.seg_viz_pub.publish(seg_msg)

            # Convert to one-hot tensor for the model
            nav_seg_tensor = torch.from_numpy(nav_seg).long()
            one_hot_mask = F.one_hot(nav_seg_tensor, num_classes=self.model_params["num_seg_classes"])
            return one_hot_mask.permute(2, 0, 1).float().unsqueeze(0) # (1, C, H, W)

    def _publish_waypoint(self, waypoint: np.ndarray):
        if self.model_params.get("normalize", False): waypoint[:2] *= MAX_V / RATE
        waypoint_msg = Float32MultiArray()
        waypoint_msg.data = [float(waypoint[0]), float(waypoint[1]), 0.0, 0.0]
        self.waypoint_pub.publish(waypoint_msg)

    def _publish_goal_images(self, sg_img: PILImage.Image, goal_img: PILImage.Image):
        for img, pub in [(sg_img, self.subgoal_pub), (goal_img, self.goal_pub_img)]:
            cv_img = cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)
            msg = self.bridge.cv2_to_imgmsg(cv_img, encoding="bgr8")
            msg.header.stamp = self.get_clock().now().to_msg()
            pub.publish(msg)

    def _publish_viz_image(self, traj_batch: np.ndarray):
        """Overlays the predicted trajectory on the current camera view."""
        frame = np.array(self.context_queue[-1].convert("RGB"))
        img_h, img_w, _ = frame.shape
        viz = frame.copy()

        cx, cy = img_w // 2, int(img_h * 0.95)
        
        # <<< VISUALIZATION FIX: Increased scaling factor for better visibility >>>
        pixels_per_m = 50.0 * (img_w / 640.0)
        
        for i, traj in enumerate(traj_batch):
            # <<< VISUALIZATION FIX: Plot absolute waypoints, not deltas >>>
            pts = []
            for waypoint_x, waypoint_y in traj[:, :2]:
                # Correctly map model's coordinate system to image's pixel system
                px = int(cx - waypoint_y * pixels_per_m) # Sideways motion on image X-axis
                py = int(cy - waypoint_x * pixels_per_m) # Forward motion on image Y-axis
                pts.append((px, py))
            
            # Prepend the robot's starting position to the beginning of the path
            if pts:
                pts.insert(0, (cx, cy))
            
            color = (0, 255, 0) if i == 0 else (255, 200, 0)
            if len(pts) > 1:
                cv2.polylines(viz, [np.array(pts, dtype=np.int32)], False, color, 2)

        img_msg = self.bridge.cv2_to_imgmsg(viz, encoding="rgb8")
        img_msg.header.stamp = self.get_clock().now().to_msg()
        self.viz_pub.publish(img_msg)

# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser("Topological navigation (ROS 2)")
    parser.add_argument("--model", "-m", default="vint", help="Name of the model in models.yaml")
    parser.add_argument("--dir", "-d", default="", help="Sub-directory under ../topomaps/images/")
    parser.add_argument("--goal-node", "-g", type=int, default=-1, help="Goal node index (-1 = last)")
    parser.add_argument("--waypoint", "-w", type=int, default=2, help="Which waypoint from the trajectory to use")
    parser.add_argument("--close-threshold", "-t", type=float, default=3.0, help="Distance to switch to next subgoal")
    parser.add_argument("--radius", "-r", type=int, default=4, help="Search radius for closest node")
    parser.add_argument("--robot", type=str, default="locobot", choices=["locobot", "robomaster", "turtlebot4"])
    args = parser.parse_args()

    rclpy.init()
    node = NavigationNode(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
