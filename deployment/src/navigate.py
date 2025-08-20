from __future__ import annotations

import argparse
import torch.nn as nn
import os
import time
from collections import deque
from pathlib import Path
import segmentation_models_pytorch as smp
from typing import Deque, List
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
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

# <<< ADDED: Imports for the standalone OneFormer test model >>>
from transformers import OneFormerProcessor, OneFormerForUniversalSegmentation

from utils import msg_to_pil, to_numpy, transform_images
from topic_names import IMAGE_TOPIC, WAYPOINT_TOPIC, SAMPLED_ACTIONS_TOPIC

# Configuration paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]

from vint_train.models.vint.vint import ViNT

# from UniDepth.unidepth.models import UniDepthV2
# from UniDepth.unidepth.utils.camera import Pinhole

# ---------------------------------------------------------------------------
# CONFIG & CONSTANTS
# ---------------------------------------------------------------------------

# Define all paths relative to the project root
ROBOT_CONFIG_PATH = PROJECT_ROOT / "deployment/config/robot.yaml"
MODEL_CONFIG_PATH = PROJECT_ROOT / "deployment/config/models.yaml"
TOPOMAP_IMAGES_DIR = PROJECT_ROOT / "deployment/topomaps/images"


with open(ROBOT_CONFIG_PATH, "r") as f:
    ROBOT_CONF = yaml.safe_load(f)
MAX_V = ROBOT_CONF["max_v"]
MAX_W = ROBOT_CONF["max_w"]
RATE = ROBOT_CONF["frame_rate"]  # Hz

def build_vint_model(**kwargs) -> nn.Module:
    """Build a ViNT model or placeholder"""
    try:
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
    
class SegmentationViNT(nn.Module):
    """
    ViNT enhanced with a semantic segmentation branch using late fusion for stability.
    """
    def __init__(
        self,
        # ViNT-specific arguments
        context_size: int,
        len_traj_pred: int,
        learn_angle: bool,
        obs_encoder: str,
        obs_encoding_size: int,
        
        # Segmentation-specific arguments
        num_seg_classes: int,
        seg_encoder: str = "resnet34",
        freeze_vint: bool = True,
        seg_feature_dim: int = 256,
        **kwargs # Absorb any extra unused parameters
    ):
        super().__init__()
        
        # Store key configuration parameters
        self.len_traj_pred = len_traj_pred
        self.learn_angle = learn_angle
        self.num_seg_classes = num_seg_classes
        
        # 1. Initialize the base ViNT model
        # The full ViNT model will be used as a feature extractor
        vint_args = {
            'context_size': context_size, 'len_traj_pred': len_traj_pred,
            'learn_angle': learn_angle, 'obs_encoder': obs_encoder,
            'obs_encoding_size': obs_encoding_size,
            **kwargs
        }
        self.vint_model = build_vint_model(**vint_args)
        
        # Freeze the entire ViNT model if specified
        if freeze_vint:
            print("Freezing all ViNT model parameters.")
            for param in self.vint_model.parameters():
                param.requires_grad = False
        
        # 2. Initialize the Segmentation Branch
        self.seg_model = smp.Unet(
            encoder_name=seg_encoder,
            encoder_weights="imagenet",
            in_channels=3,
            classes=num_seg_classes,
            activation=None,
        )
        
        # 3. Initialize a Segmentation Feature Extractor
        # This module processes segmentation logits into a 1D feature vector
        self.seg_feature_extractor = nn.Sequential(
            nn.Conv2d(num_seg_classes, seg_feature_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(seg_feature_dim),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),
            nn.Linear(seg_feature_dim * 49, obs_encoding_size),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )
        # 4. Initialize NEW Prediction Heads for the fused features
        # The input dimension is doubled because we concatenate the two feature vectors
        fused_dim = obs_encoding_size * 2
        num_action_outputs = len_traj_pred * (3 if learn_angle else 2)
        
        self.action_predictor = nn.Sequential(
            nn.Linear(fused_dim, 512),
            nn.ReLU(),
            nn.Linear(512, num_action_outputs),
        )
        
        # The distance predictor takes the fused observation and the goal
        dist_predictor_input_dim = fused_dim + obs_encoding_size
        self.dist_predictor = nn.Sequential(
            nn.Linear(dist_predictor_input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )
    
    def unfreeze_vint(self):
        """Unfreeze all ViNT parameters for Stage 2 finetuning."""
        print("Unfreezing ViNT model parameters...")
        for param in self.vint_model.parameters():
            param.requires_grad = True
    
    def forward(self, obs_images: torch.Tensor, goal_images: torch.Tensor) -> Dict[str, torch.Tensor]:
        batch_size = obs_images.shape[0]
        
        # --- Branch 1: Get ViNT Features ---
        # We get the final 1D feature vectors from the pre-trained model
        with torch.set_grad_enabled(next(self.vint_model.parameters()).requires_grad):
            obs_encoding = self.vint_model.compress_obs_enc(self.vint_model.obs_encoder(obs_images))
            goal_encoding = self.vint_model.compress_goal_enc(self.vint_model.goal_encoder(goal_images))

            # ADDED CODE 1
            # Normalize for stability
            obs_encoding = F.normalize(obs_encoding, p=2, dim=-1) * 10
            goal_encoding = F.normalize(goal_encoding, p=2, dim=-1) * 10
        
        # --- Branch 2: Get Segmentation Features ---
        last_obs_frame = obs_images[:, -3:, :, :]

        obs_seg_logits = self.seg_model(last_obs_frame)

        seg_probs = F.softmax(obs_seg_logits, dim=1).detach()
        seg_features_obs = self.seg_feature_extractor(seg_probs)
        seg_features_obs = F.normalize(seg_features_obs, p=2, dim=-1) * 10
        
        # --- Late Fusion Step ---
        # Concatenate the final 1D feature vectors from both branches
        fused_obs_features = torch.cat([obs_encoding, seg_features_obs], dim=1)
        
        # --- Final Prediction ---
        # The new prediction heads take the fused observation features and the original goal features
        
        # For simplicity, we assume the action predictor takes the fused observation
        # and the distance predictor takes the combined features.
        pred_actions = self.action_predictor(fused_obs_features)
        
        combined_features = torch.cat([fused_obs_features, goal_encoding], dim=1)
        pred_dist = self.dist_predictor(combined_features)

        return {
            'dist_pred': pred_dist,
            'action_pred': pred_actions.view(batch_size, self.len_traj_pred, -1),
            'obs_seg_logits': obs_seg_logits,
        }

def load_model(
    model_path: str,
    config: dict,
    device: torch.device = torch.device("cpu"),
) -> nn.Module:
    """Load a model from a checkpoint file."""
    model_type = config["model_type"]
    
    # Use the local model classes defined in this script
    if model_type == "vint":
        model = ViNT(**config)
    elif model_type == "segmentation_vint":
        model = SegmentationViNT(**config)
    else:
        raise ValueError(f"Invalid model type: {model_type}")
    
    checkpoint = torch.load(model_path, map_location=device)
    
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    if hasattr(state_dict, 'state_dict'):
        state_dict = state_dict.state_dict()
    if all(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k[7:]: v for k, v in state_dict.items()}

    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    return model

def _load_model(model_name: str, device: torch.device):
    with open(MODEL_CONFIG_PATH, "r") as f:
        model_paths = yaml.safe_load(f)

    mconf_path = model_paths[model_name]["config_path"]
    ckpt_path = model_paths[model_name]["ckpt_path"]
    with open(mconf_path, "r") as f:
        model_params = yaml.safe_load(f)

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Model weights not found at {ckpt_path}")

    print(f"Loading model from {ckpt_path}")
    # This now calls the local load_model function
    model = load_model(ckpt_path, model_params, device).to(device).eval()
    return model, model_params

class NavigationNode(Node):
    """Sub‑goal navigation with topomap + trajectory visualisation."""

    def __init__(self, args: argparse.Namespace):
        super().__init__("navigation")
        self.args = args

        # Torch / model ------------------------------------------------------
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.get_logger().info(f"Using device: {self.device}")

        self.model, self.model_params = _load_model(args.model, self.device)

        # ROS2 이제 모든 모델 타입 지원
        self.get_logger().info(f"Using model type: {self.model_params['model_type']}")

        self.context_size: int = self.model_params["context_size"]

        # NOMAD 모델인 경우 noise_scheduler 초기화
        if self.model_params["model_type"] == "nomad":
            self.noise_scheduler = DDPMScheduler(
                num_train_timesteps=self.model_params["num_diffusion_iters"],
                beta_schedule="squaredcos_cap_v2",
                clip_sample=True,
                prediction_type="epsilon",
            )

        self.bridge = CvBridge()
        self.context_queue = deque(maxlen=self.context_size + 1)
        self.last_ctx_time = self.get_clock().now()
        self.ctx_dt = 0.25

        self.current_waypoint = np.zeros(2)
        self.obstacle_points = None

        self.top_view_size = (400, 400)
        self.proximity_threshold = 0.8
        self.top_view_resolution = self.top_view_size[0] / self.proximity_threshold
        self.top_view_sampling_step = 5
        self.safety_margin = 0.17
        self.DIM = (640, 480)

        # Topological map ----------------------------------------------------
        self.topomap: List[PILImage] = self._load_topomap(args.dir)
        self.goal_node = (
            (len(self.topomap) - 1) if args.goal_node == -1 else args.goal_node
        )
        self.closest_node = 0

        # ROS interfaces -----------------------------------------------------
        # 로봇 타입에 따른 이미지 토픽 선택
        if args.robot == "locobot":
            image_topic = "/robot1/camera/image"  # 상수에서 가져옴
            waypoint_topic = "/robot1/waypoint"
            sampled_actions_topic = "/robot1/sampled_actions"
        elif args.robot == "robomaster":
            image_topic = "/camera/image_color"
            waypoint_topic = "/robot3/waypoint"
            sampled_actions_topic = "/robot3/sampled_actions"
        elif args.robot == "turtlebot4":
            image_topic = "/robot2/oakd/rgb/preview/image_raw"
            waypoint_topic = "/robot2/waypoint"
            sampled_actions_topic = "/robot2/sampled_actions"
        else:
            raise ValueError(f"Unknown robot type: {args.robot}")

        self.seg_colors = np.array([
            [0, 255, 0],    # 0: floor (green)
            [255, 0, 0],    # 1: wall (red)
            [255, 255, 0],  # 2: door (yellow)
            [0, 0, 255],    # 3: furniture (blue)
            [128, 128, 128] # 4: unknown (gray)
        ], dtype=np.uint8)



        self.create_subscription(Image, image_topic, self._image_cb, 1)
        self.waypoint_pub = self.create_publisher(Float32MultiArray, waypoint_topic, 1)
        self.sampled_actions_pub = self.create_publisher(
            Float32MultiArray, sampled_actions_topic, 1
        )
        self.goal_pub = self.create_publisher(Bool, "/topoplan/reached_goal", 1)
        self.viz_pub = self.create_publisher(Image, "navigation_viz", 1)
        self.seg_viz_pub = self.create_publisher(Image, "navigation_seg_viz", 1)
        self.subgoal_pub = self.create_publisher(Image, "navigation_subgoal", 1)
        self.goal_pub_img = self.create_publisher(Image, "navigation_goal", 1)
        self.create_timer(1.0 / RATE, self._timer_cb)
        self.get_logger().info("Navigation node initialised. Waiting for images…")

        # <<< ADDED: Logic for the standalone OneFormer test >>>

        self.get_logger().info("--- OneFormer Test Mode Enabled ---")
        self.oneformer_pub = self.create_publisher(Image, "/oneformer_seg_viz", 1)
        self.oneformer_processor = OneFormerProcessor.from_pretrained("shi-labs/oneformer_ade20k_swin_tiny")
        self.oneformer_model = OneFormerForUniversalSegmentation.from_pretrained(
                "shi-labs/oneformer_ade20k_swin_tiny",
                use_safetensors=True
            ).to(self.device).eval()
        self.setup_oneformer_class_map()
        self.get_logger().info("OneFormer model loaded for real-time testing.")

        self.get_logger().info("Navigation node initialised. Waiting for images…")

        # 시작하기 전에 중요한 파라미터들 출력
        self.get_logger().info("=" * 60)
        self.get_logger().info("NAVIGATION NODE PARAMETERS")
        self.get_logger().info("=" * 60)
        self.get_logger().info(f"Robot type: {self.args.robot}")
        self.get_logger().info(f"Image topic: {image_topic}")
        self.get_logger().info("-" * 60)
        self.get_logger().info("ROBOT CONFIGURATION:")
        self.get_logger().info(f"  - Max linear velocity: {MAX_V} m/s")
        self.get_logger().info(f"  - Max angular velocity: {MAX_W} rad/s")
        self.get_logger().info(f"  - Frame rate: {RATE} Hz")
        self.get_logger().info(f"  - Safety margin: {self.safety_margin} m")
        self.get_logger().info(f"  - Proximity threshold: {self.proximity_threshold} m")
        self.get_logger().info("-" * 60)
        self.get_logger().info("CAMERA CONFIGURATION:")
        self.get_logger().info(f"  - Image dimensions: {self.DIM}")
        self.get_logger().info("-" * 60)
        self.get_logger().info("MODEL CONFIGURATION:")
        self.get_logger().info(f"  - Model name: {self.args.model}")
        self.get_logger().info(f"  - Model type: {self.model_params['model_type']}")
        self.get_logger().info(f"  - Device: {self.device}")
        self.get_logger().info(f"  - Context size: {self.context_size}")
        self.get_logger().info(f"  - Context update interval: {self.ctx_dt} seconds")
        if self.model_params["model_type"] == "nomad":
            self.get_logger().info(
                f"  - Trajectory length: {self.model_params['len_traj_pred']}"
            )
            self.get_logger().info(
                f"  - Diffusion iterations: {self.model_params['num_diffusion_iters']}"
            )
        self.get_logger().info(f"  - Image size: {self.model_params['image_size']}")
        self.get_logger().info(
            f"  - Normalize: {self.model_params.get('normalize', False)}"
        )
        self.get_logger().info("-" * 60)
        self.get_logger().info("TOPOLOGICAL MAP CONFIGURATION:")
        self.get_logger().info(f"  - Topomap directory: {self.args.dir}")
        self.get_logger().info(f"  - Number of nodes: {len(self.topomap)}")
        self.get_logger().info(f"  - Goal node: {self.goal_node}")
        self.get_logger().info(f"  - Search radius: {self.args.radius}")
        self.get_logger().info(f"  - Close threshold: {self.args.close_threshold}")
        self.get_logger().info("-" * 60)
        self.get_logger().info("OBSTACLE AVOIDANCE CONFIGURATION:")
        self.get_logger().info(f"  - Top view size: {self.top_view_size}")
        self.get_logger().info(
            f"  - Top view resolution: {self.top_view_resolution:.2f} pixels/m"
        )
        self.get_logger().info(
            f"  - Top view sampling step: {self.top_view_sampling_step} pixels"
        )
        self.get_logger().info("-" * 60)
        self.get_logger().info("ROS TOPICS:")
        self.get_logger().info(f"  - Subscribing to: {image_topic}")
        self.get_logger().info(f"  - Publishing waypoints to: {WAYPOINT_TOPIC}")
        self.get_logger().info(
            f"  - Publishing sampled actions to: {SAMPLED_ACTIONS_TOPIC}"
        )
        self.get_logger().info(
            f"  - Publishing navigation visualization to: /navigation_viz"
        )
        self.get_logger().info(f"  - Publishing subgoal image to: /navigation_subgoal")
        self.get_logger().info(f"  - Publishing goal image to: /navigation_goal")
        self.get_logger().info(
            f"  - Publishing goal reached status to: /topoplan/reached_goal"
        )
        self.get_logger().info("-" * 60)
        self.get_logger().info("EXECUTION PARAMETERS:")
        self.get_logger().info(f"  - Waypoint index: {self.args.waypoint}")
        self.get_logger().info(f"  - Number of samples: {self.args.num_samples}")
        self.get_logger().info("-" * 60)
        self.get_logger().info("VISUALIZATION PARAMETERS:")
        self.get_logger().info(f"  - Pixels per meter: 3.0")
        self.get_logger().info(f"  - Lateral scale: 1.0")
        self.get_logger().info(f"  - Horizontal scale: 4.0")
        self.get_logger().info(f"  - Robot symbol length: 10 pixels")
        self.get_logger().info("=" * 60)

    # Helper: topomap
    # ------------------------------------------------------------------

    def _load_topomap(self, subdir: str) -> List[PILImage.Image]:
        dpath = TOPOMAP_IMAGES_DIR / subdir
        if not dpath.exists():
            raise FileNotFoundError(f"Topomap directory {dpath} does not exist")
        img_files = sorted(os.listdir(dpath), key=lambda x: int(os.path.splitext(x)[0]))
        return [PILImage.open(dpath / f) for f in img_files]

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def setup_oneformer_class_map(self):
        """Creates the lookup table to map ADE20K classes to our nav classes."""
        self.oneformer_lookup_table = np.full(150, fill_value=4, dtype=np.uint8) # Default to unknown
        ade20k_to_nav = {
            3: 0, 6: 0, 11: 0, 13: 0, 29: 0, 53: 0, # Floors/walkable
            0: 1, 1: 1, 5: 1, 25: 1,                 # Walls/barriers
            14: 2,                                   # Doors
            7: 3, 10: 3, 15: 3, 19: 3, 24: 3, 31: 3, 33: 3, 65: 3, # Furniture
        }
        for ade_class, nav_class in ade20k_to_nav.items():
            if ade_class < 150:
                self.oneformer_lookup_table[ade_class] = nav_class

    def _image_cb(self, msg: Image):
        now = self.get_clock().now()
        if (now - self.last_ctx_time).nanoseconds < self.ctx_dt * 1e9:
            return  # 아직 0.25 s 안 지났으면 무시
        self.context_queue.append(msg_to_pil(msg))
        self.last_ctx_time = now

        # <<< ADDED: Run OneFormer inference on every new frame if in test mode >>>
        self.run_and_publish_oneformer(msg_to_pil(msg))

        self.get_logger().info(
            f"Image added to context queue ({len(self.context_queue)})"
        )

    def run_and_publish_oneformer(self, image: PILImage.Image):
        """Runs OneFormer on a single image and publishes the colored mask."""
        with torch.no_grad():
            inputs = self.oneformer_processor(images=image, task_inputs=["semantic"], return_tensors="pt").to(self.device)
            outputs = self.oneformer_model(**inputs)
            
            # Post-process to get the final segmentation map at original resolution
            original_size = image.size[::-1] # (height, width)
            ade_preds = self.oneformer_processor.post_process_semantic_segmentation(outputs, target_sizes=[original_size])[0]
            
            # Convert to our navigation classes and then to a colored image
            nav_seg = self.oneformer_lookup_table[ade_preds.cpu().numpy()]
            colored_mask = self.seg_colors[nav_seg]
            
            # Publish the result
            seg_msg = self.bridge.cv2_to_imgmsg(cv2.cvtColor(colored_mask, cv2.COLOR_RGB2BGR), encoding="bgr8")
            seg_msg.header.stamp = self.get_clock().now().to_msg()
            self.oneformer_pub.publish(seg_msg)

    def _timer_cb(self):
        if len(self.context_queue) <= self.context_size:
            return

        # 모델 타입에 따라 다른 처리
        if self.model_params["model_type"] == "nomad":
            self._timer_cb_nomad()
        else:
            self._timer_cb_other()

        # 목표 도달 시 로그 출력 (원본과 동일)
        if self.closest_node == self.goal_node:
            self.get_logger().info("Reached goal! Stopping...")

    """def _timer_cb_nomad(self):
        # -----------------------------------------------------------------
        # 1. Compute closest node via distance prediction
        # -----------------------------------------------------------------
        start = max(self.closest_node - self.args.radius, 0)
        end = min(self.closest_node + self.args.radius + 1, self.goal_node)

        # Build batch of (obs, goal) tensors
        obs_images = transform_images(
            list(self.context_queue),
            self.model_params["image_size"],
            center_crop=False,
        ).to(self.device)
        obs_images = torch.split(obs_images, 3, dim=1)
        obs_images = torch.cat(obs_images, dim=1)  # merge context

        batch_goal_imgs = []
        for g_idx in range(start, end + 1):
            g_img = transform_images(
                self.topomap[g_idx], self.model_params["image_size"], center_crop=False
            )
            batch_goal_imgs.append(g_img)
        goal_tensor = torch.cat(batch_goal_imgs, dim=0).to(self.device)

        mask = torch.zeros(1, device=self.device, dtype=torch.long)
        with torch.no_grad():
            obsgoal_cond = self.model(
                "vision_encoder",
                obs_img=obs_images.repeat(len(goal_tensor), 1, 1, 1),
                goal_img=goal_tensor,
                input_goal_mask=mask.repeat(len(goal_tensor)),
            )
            dists = self.model("dist_pred_net", obsgoal_cond=obsgoal_cond)
            dists_np = to_numpy(dists.flatten())

        min_idx = int(np.argmin(dists_np))
        self.closest_node = start + min_idx
        sg_idx = min(
            min_idx + int(dists_np[min_idx] < self.args.close_threshold),
            len(goal_tensor) - 1,
        )
        obs_cond = obsgoal_cond[sg_idx].unsqueeze(0)
        sg_global_idx = start + sg_idx  # ← 새로 추가
        sg_pil = self.topomap[sg_global_idx]  # ← 새로 추가
        goal_pil = self.topomap[self.goal_node]  # ← 새로 추가

        # -----------------------------------------------------------------
        # 2. Sample trajectories towards sub‑goal (diffusion)
        # -----------------------------------------------------------------
        with torch.no_grad():
            if obs_cond.ndim == 2:
                obs_cond = obs_cond.repeat(self.args.num_samples, 1)
            else:
                obs_cond = obs_cond.repeat(self.args.num_samples, 1, 1)

            len_traj = self.model_params["len_traj_pred"]
            naction = torch.randn(
                (self.args.num_samples, len_traj, 2), device=self.device
            )
            self.noise_scheduler.set_timesteps(self.model_params["num_diffusion_iters"])
            for k in self.noise_scheduler.timesteps:
                noise_pred = self.model(
                    "noise_pred_net", sample=naction, timestep=k, global_cond=obs_cond
                )
                naction = self.noise_scheduler.step(noise_pred, k, naction).prev_sample

        traj_batch = to_numpy(get_action(naction))

        # -----------------------------------------------------------------
        # 3. Publish ROS messages
        # -----------------------------------------------------------------
        self._publish_msgs(traj_batch)
        self._publish_viz_image(traj_batch)
        self._publish_goal_images(sg_pil, goal_pil)"""

    def _timer_cb_other(self):
        """nomad 외의 모델을 위한 타이머 콜백 처리 - 원본 코드와 동일하게 구현"""
        # ROS1의 navigate_ros1.py의 else 부분을 ROS2 방식으로 구현
        start = max(self.closest_node - self.args.radius, 0)
        end = min(self.closest_node + self.args.radius + 1, self.goal_node)

        # Prepare batches
        if len(self.context_queue) > self.context_size:
            obs_images_pil = list(self.context_queue)[-self.context_size:]
        else:
            obs_images_pil = list(self.context_queue)

        batch_obs_imgs = []
        batch_goal_imgs = []

        for sg_img in self.topomap[start : end + 1]:
            batch_obs_imgs.append(transform_images(obs_images_pil, self.model_params["image_size"]))
            batch_goal_imgs.append(transform_images(sg_img, self.model_params["image_size"]))

        batch_obs_tensor = torch.cat(batch_obs_imgs, dim=0).to(self.device)
        batch_goal_tensor = torch.cat(batch_goal_imgs, dim=0).to(self.device)

        with torch.no_grad():
            outputs = self.model(batch_obs_tensor, batch_goal_tensor)
            
            if isinstance(outputs, dict):
                distances_np = to_numpy(outputs['dist_pred'])
                waypoints_np = to_numpy(outputs['action_pred'])

                if 'obs_seg_logits' in outputs and outputs['obs_seg_logits'] is not None:
                    seg_logits = outputs['obs_seg_logits']
                    self._publish_seg_viz(seg_logits)

            else: # Handle tuple output for older models
                distances, waypoints = outputs
                distances_np = to_numpy(distances)
                waypoints_np = to_numpy(waypoints)

        # Find closest node and select waypoint
        min_dist_idx = np.argmin(distances_np)

        # 서브골과 경로점 선택 - 원본과 동일하게 구현
        chosen_waypoint = np.zeros(4)  # 4차원 벡터 (원본과 동일)
        selected_waypoints = None  # 시각화용 전체 웨이포인트 저장 변수

        if distances_np[min_dist_idx] > self.args.close_threshold:
            chosen_waypoint[:2] = waypoints_np[min_dist_idx][self.args.waypoint][:2]
            selected_waypoints = waypoints_np[
                min_dist_idx
            ]  # 시각화용 전체 웨이포인트 저장
            self.closest_node = start + min_dist_idx
        else:
            next_idx = min(min_dist_idx + 1, len(waypoints_np) - 1)
            chosen_waypoint[:2] = waypoints_np[next_idx][self.args.waypoint][:2]
            selected_waypoints = waypoints_np[next_idx]  # 시각화용 전체 웨이포인트 저장
            self.closest_node = min(start + min_dist_idx + 1, self.goal_node)

        if self.model_params.get("normalize", False):
            chosen_waypoint[:2] *= MAX_V / RATE

        # 직접 waypoint 메시지 발행 (원본과 동일)
        waypoint_msg = Float32MultiArray()
        waypoint_msg.data = chosen_waypoint.tolist()
        self.waypoint_pub.publish(waypoint_msg)

        # 목표 도달 상태 발행 (원본과 동일)
        reached_goal = bool(self.closest_node == self.goal_node)
        self.goal_pub.publish(Bool(data=reached_goal))

        # 시각화를 위한 추가 코드 (ROS2 확장 기능)
        sg_global_idx = min(
            start
            + min_dist_idx
            + int(distances_np[min_dist_idx] <= self.args.close_threshold),
            self.goal_node,
        )
        sg_pil = self.topomap[sg_global_idx]
        goal_pil = self.topomap[self.goal_node]

        # 시각화를 위한 궤적 생성 (NOMAD 형식으로 변환)
        if selected_waypoints is not None:
            # NOMAD 스타일의 traj_batch 형식으로 변환 (Batch, TimestepCount, Dim)
            traj_vis = np.zeros((1, len(selected_waypoints), 2))
            for i in range(len(selected_waypoints)):
                traj_vis[0, i] = selected_waypoints[i][:2]

            # 정규화가 적용된 경우 시각화를 위해 정규화 적용
            # if self.model_params.get("normalize", False):
            #     traj_vis *= MAX_V / RATE

            self._publish_viz_image(traj_vis)

        # 목표 이미지 발행
        self._publish_goal_images(sg_pil, goal_pil)

    # ------------------------------------------------------------------
    # Publish helpers
    # ------------------------------------------------------------------
    def _publish_goal_images(self, sg_img: PILImage.Image, goal_img: PILImage.Image):
        """Publish current sub‑goal and final goal images as ROS sensor_msgs/Image."""
        for img, pub in [(sg_img, self.subgoal_pub), (goal_img, self.goal_pub_img)]:
            cv_img = cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)
            msg = self.bridge.cv2_to_imgmsg(cv_img, encoding="bgr8")
            msg.header.stamp = self.get_clock().now().to_msg()
            pub.publish(msg)

    def _publish_seg_viz(self, seg_logits: torch.Tensor):
        """Converts segmentation logits to a colored image, resizes it, and publishes it."""
        # Take the prediction for the first item in the batch
        pred_mask = torch.argmax(seg_logits[0], dim=0).cpu().numpy().astype(np.uint8)
        
        # Map the class indices to colors
        colored_mask = self.seg_colors[pred_mask]
        
        # Get the original camera image dimensions from the latest message in the queue
        if self.context_queue:
            latest_ros_img = self.context_queue[-1]
            original_dims = (latest_ros_img.width, latest_ros_img.height)
            
            # Resize the colored mask to match the original camera image size
            colored_mask = cv2.resize(colored_mask, original_dims, interpolation=cv2.INTER_NEAREST)

        # Convert to BGR for OpenCV
        colored_mask_bgr = cv2.cvtColor(colored_mask, cv2.COLOR_RGB2BGR)
        
        # Create and publish the ROS Image message
        seg_msg = self.bridge.cv2_to_imgmsg(colored_mask_bgr, encoding="bgr8")
        seg_msg.header.stamp = self.get_clock().now().to_msg()
        self.seg_viz_pub.publish(seg_msg)
    # ------------------------------------------------------------------
    # Publish helpers
    # ------------------------------------------------------------------

    def _publish_msgs(self, traj_batch: np.ndarray):
        # sampled actions
        actions_msg = Float32MultiArray()
        actions_msg.data = [0.0] + [float(x) for x in traj_batch.flatten()]
        self.sampled_actions_pub.publish(actions_msg)

        # chosen waypoint
        chosen = traj_batch[0][self.args.waypoint]
        if self.model_params.get("normalize", False):
            chosen *= MAX_V / RATE
        wp_msg = Float32MultiArray()
        wp_msg.data = [float(chosen[0]), float(chosen[1]), 0.0, 0.0]  # 4‑D compat
        self.waypoint_pub.publish(wp_msg)

        # goal status
        reached = bool(self.closest_node == self.goal_node)
        self.goal_pub.publish(Bool(data=reached))

    def _publish_viz_image(self, traj_batch: np.ndarray):
        frame = np.array(self.context_queue[-1])  # latest RGB frame
        img_h, img_w = frame.shape[:2]
        viz = frame.copy()

        cx = img_w // 2
        cy = int(img_h * 0.95)

        # 수정사항:
        pixels_per_m = 3.0
        lateral_scale = 1.0
        robot_symbol_length = 10

        cv2.line(
            viz,
            (cx - robot_symbol_length, cy),
            (cx + robot_symbol_length, cy),
            (255, 0, 0),
            2,
        )
        cv2.line(
            viz,
            (cx, cy - robot_symbol_length),
            (cx, cy + robot_symbol_length),
            (255, 0, 0),
            2,
        )

        # Draw each trajectory
        for i, traj in enumerate(traj_batch):
            pts = []
            # 수정: 첫 점을 로봇 위치(cx, cy)에서 시작
            pts.append((cx, cy))

            acc_x, acc_y = 0.0, 0.0
            for dx, dy in traj:
                acc_x += dx
                acc_y += dy
                # 수정: acc_y를 사용하여 누적값으로 계산
                px = int(cx - acc_y * pixels_per_m * lateral_scale)
                py = int(cy - acc_x * pixels_per_m)
                pts.append((px, py))

            if len(pts) >= 2:
                color = (
                    (0, 255, 0) if i == 0 else (255, 200, 0)
                )  # 첫 번째 trajectory는 녹색
                cv2.polylines(viz, [np.array(pts, dtype=np.int32)], False, color, 2)

        img_msg = self.bridge.cv2_to_imgmsg(viz, encoding="rgb8")
        img_msg.header.stamp = self.get_clock().now().to_msg()
        self.viz_pub.publish(img_msg)


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser("Topological navigation (ROS 2)")
    parser.add_argument("--model", "-m", default="vint")
    parser.add_argument(
        "--dir", "-d", default="", help="sub‑directory under ../topomaps/images/"
    )
    parser.add_argument(
        "--goal-node", "-g", type=int, default=-1, help="Goal node index (-1 = last)"
    )
    parser.add_argument("--waypoint", "-w", type=int, default=2)
    parser.add_argument("--close-threshold", "-t", type=float, default=3.0)
    parser.add_argument("--radius", "-r", type=int, default=4)
    parser.add_argument("--num-samples", "-n", type=int, default=8)
    parser.add_argument(
        "--robot",
        type=str,
        default="locobot",
        choices=["locobot", "robomaster", "turtlebot4"],
        help="Robot type (locobot, robomaster, turtlebot4)",
    )
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
