#!/usr/bin/env python3
"""explore_ros2.py – ROS 2 re‑implementation of the original ROS 1 exploration node.

Last updated: 2025‑04‑17
───────────────────────
* **Fix**: use `self.model_params["len_traj_pred"]` instead of nonexistent
  `self.model.len_traj_pred`.
* Store `model_params` as an instance attribute so it is accessible from the
  timer callback.
* Minor linters & type hints.
"""
from __future__ import annotations

import argparse
import os
import time
from collections import deque
from pathlib import Path
from typing import Deque, List

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
import torch
import yaml
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from utils import msg_to_pil, to_numpy, transform_images, load_model
from vint_train.training.train_utils import get_action
from topic_names import (
    IMAGE_TOPIC,
    WAYPOINT_TOPIC,
    SAMPLED_ACTIONS_TOPIC,
)

# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------
THIS_DIR = Path(__file__).resolve().parent
MODEL_WEIGHTS_PATH = THIS_DIR / "../model_weights"
ROBOT_CONFIG_PATH = THIS_DIR / "../config/robot.yaml"
MODEL_CONFIG_PATH = THIS_DIR / "../config/models.yaml"

with open(ROBOT_CONFIG_PATH, "r") as f:
    ROBOT_CONF = yaml.safe_load(f)
MAX_V = ROBOT_CONF["max_v"]
MAX_W = ROBOT_CONF["max_w"]
RATE = ROBOT_CONF["frame_rate"]  # Hz


def _load_model(model_name: str, device: torch.device):
    with open(MODEL_CONFIG_PATH, "r") as f:
        model_paths = yaml.safe_load(f)

    model_config_path = model_paths[model_name]["config_path"]
    with open(model_config_path, "r") as f:
        model_params = yaml.safe_load(f)

    ckpt_path = model_paths[model_name]["ckpt_path"]
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Model weights not found at {ckpt_path}")

    print(f"[INFO] Loading model from {ckpt_path}")
    model = load_model(ckpt_path, model_params, device)
    model = model.to(device).eval()
    return model, model_params


class ExplorationNode(Node):
    """ROS 2 node that performs image‑conditioned waypoint sampling."""

    def __init__(self, args: argparse.Namespace):
        super().__init__("exploration")
        self.args = args

        # Torch device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.get_logger().info(f"Using device: {self.device}")

        # Model & params
        self.model, self.model_params = _load_model(args.model, self.device)
        self.context_size: int = self.model_params["context_size"]

        # Scheduler
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=self.model_params["num_diffusion_iters"],
            beta_schedule="squaredcos_cap_v2",
            clip_sample=True,
            prediction_type="epsilon",
        )

        # Queues & state
        self.context_queue: Deque[np.ndarray] = deque(maxlen=self.context_size + 1)

        # ROS interfaces
        self.create_subscription(Image, IMAGE_TOPIC, self._image_cb, 1)
        self.waypoint_pub = self.create_publisher(Float32MultiArray, WAYPOINT_TOPIC, 1)
        self.sampled_actions_pub = self.create_publisher(
            Float32MultiArray, SAMPLED_ACTIONS_TOPIC, 1
        )

        self.create_timer(1.0 / RATE, self._timer_cb)
        self.get_logger().info("Exploration node initialised. Waiting for images…")

    # ---------------------------------------------------------------------
    # Callbacks
    # ---------------------------------------------------------------------

    def _image_cb(self, msg: Image):
        pil_img = msg_to_pil(msg)
        self.context_queue.append(pil_img)

    def _timer_cb(self):
        if len(self.context_queue) <= self.context_size:
            return  # not enough context yet

        # -----------------------------------------------------------------
        # 1. Pre‑process observations
        obs_imgs = transform_images(
            list(self.context_queue),
            self.model_params["image_size"],
            center_crop=False,
        ).to(self.device)

        fake_goal = torch.randn((1, 3, *self.model_params["image_size"]), device=self.device)
        mask = torch.ones(1, device=self.device, dtype=torch.long)  # ignore goal

        with torch.no_grad():
            # Encode observations
            obs_cond = self.model(
                "vision_encoder", obs_img=obs_imgs, goal_img=fake_goal, input_goal_mask=mask
            )
            if obs_cond.ndim == 2:
                obs_cond = obs_cond.repeat(self.args.num_samples, 1)
            else:
                obs_cond = obs_cond.repeat(self.args.num_samples, 1, 1)

            # -----------------------------------------------------------------
            # 2. Diffusion sampling
            len_traj_pred = self.model_params["len_traj_pred"]
            naction = torch.randn(
                (self.args.num_samples, len_traj_pred, 2), device=self.device
            )
            self.noise_scheduler.set_timesteps(self.model_params["num_diffusion_iters"])
            for k in self.noise_scheduler.timesteps:
                noise_pred = self.model(
                    "noise_pred_net", sample=naction, timestep=k, global_cond=obs_cond
                )
                naction = self.noise_scheduler.step(noise_pred, k, naction).prev_sample

        # -----------------------------------------------------------------
        # 3. Post‑process & publish
        naction_np = to_numpy(get_action(naction))
        sampled_actions_msg = Float32MultiArray()
        sampled_actions_msg.data = [0.0] + [float(x) for x in naction_np.flatten()]
        self.sampled_actions_pub.publish(sampled_actions_msg)

        chosen_waypoint = naction_np[0][self.args.waypoint]
        if self.model_params.get("normalize", False):
            chosen_waypoint *= (MAX_V / RATE)
        waypoint_msg = Float32MultiArray()
        waypoint_msg.data = [float(chosen_waypoint[0]), float(chosen_waypoint[1])]
        self.waypoint_pub.publish(waypoint_msg)
        self.get_logger().debug("Published waypoint")


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser("GNM‑Diffusion exploration (ROS 2)")
    parser.add_argument("--model", "-m", default="nomad", help="Model key in YAML config")
    parser.add_argument("--waypoint", "-w", type=int, default=2, help="Waypoint index")
    parser.add_argument("--num-samples", "-n", type=int, default=8, help="# of samples")
    args = parser.parse_args()

    rclpy.init()
    node = ExplorationNode(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
