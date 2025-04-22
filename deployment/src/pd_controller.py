#!/usr/bin/env python3
"""pd_controller_ros2.py – ROS 2 version of the original PD controller node.

Behaviour
---------
* Listens to WAYPOINT_TOPIC (`Float32MultiArray`) and REACHED_GOAL_TOPIC (`Bool`).
* Computes linear/angular velocity commands with a simple PD‑style heuristic.
* Publishes geometry_msgs/Twist on the velocity topic defined in robot.yaml.

Run after installing your Python package (or directly with `ros2 run` / `python`).
"""
from __future__ import annotations

import time
from typing import Optional, Tuple

import numpy as np
import yaml

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32MultiArray, Bool

from topic_names import WAYPOINT_TOPIC, REACHED_GOAL_TOPIC
from utils import clip_angle  # assumes utils.py provides this helper

# ────────────────────────────────────────────────────────────────────────────────
# Robot‑specific constants
# ────────────────────────────────────────────────────────────────────────────────
CONFIG_PATH = "../config/robot.yaml"
with open(CONFIG_PATH, "r") as f:
    robot_cfg = yaml.safe_load(f)

MAX_V: float = robot_cfg["max_v"]
MAX_W: float = robot_cfg["max_w"]
VEL_TOPIC: str = robot_cfg["vel_navi_topic"]
DT: float = 1.0 / robot_cfg["frame_rate"]

# Controller parameters
RATE: int = 9  # control loop Hz
EPS: float = 1e-8
WAYPOINT_TIMEOUT: float = 1.0  # seconds – drop stale waypoint


# ────────────────────────────────────────────────────────────────────────────────
class PDControllerNode(Node):
    """ROS 2 node implementing a planar PD controller."""

    def __init__(self, controller_type: str) -> None:
        super().__init__("pd_controller")
        self.controller_type = controller_type

        # internal state
        self.waypoint: Optional[np.ndarray] = None
        self._last_wp_time: float = 0.0
        self.reached_goal: bool = False
        self.reverse_mode: bool = False  # flip linear.x if required

        # pubs / subs
        self.vel_pub = self.create_publisher(Twist, VEL_TOPIC, 1)
        self.create_subscription(
            Float32MultiArray, WAYPOINT_TOPIC, self._waypoint_cb, 1
        )
        self.create_subscription(Bool, REACHED_GOAL_TOPIC, self._goal_cb, 1)

        # timer for control loop
        self.create_timer(1.0 / RATE, self._timer_cb)
        self.get_logger().info(
            "PD controller node initialised – waiting for waypoints…"
        )

    # ─────────────────────── callbacks ────────────────────────
    def _waypoint_cb(self, msg: Float32MultiArray) -> None:
        self.waypoint = np.asarray(msg.data, dtype=float)
        self._last_wp_time = time.time()
        self.get_logger().info(f"Waypoint received: {self.waypoint.tolist()}")

    def _goal_cb(self, msg: Bool) -> None:
        self.reached_goal = msg.data

    # ─────────────────────── helpers ──────────────────────────
    def _waypoint_valid(self) -> bool:
        return (
            self.waypoint is not None
            and (time.time() - self._last_wp_time) < WAYPOINT_TIMEOUT
        )

    def _pd_control(self, wp: np.ndarray) -> Tuple[float, float]:
        """Compute (v, w) for 2‑D or 4‑D waypoint."""
        if wp.size == 2:
            dx, dy = wp
            use_heading = False
        elif wp.size == 4:
            dx, dy, hx, hy = wp
            use_heading = np.abs(dx) < EPS and np.abs(dy) < EPS
        else:
            raise ValueError("Waypoint must be 2‑D or 4‑D vector")

        # # heading‑only case
        # if wp.size == 4 and abs(dx) < EPS and abs(dy) < EPS:
        #     v = 0.0
        #     w = clip_angle(np.arctan2(hy, hx)) / DT
        # # rotate in place when dx ≈ 0
        # elif abs(dx) < EPS:
        #     v = 0.0
        #     w = np.sign(dy) * np.pi / (2 * DT)
        # else:
        #     v = dx / DT
        #     w = np.arctan(dy / dx) / DT

        # === 각도 계산 ===
        if use_heading:
            v = 0.0
            desired_yaw = np.arctan2(hy, hx)
        elif abs(dx) < EPS:
            v = 0.0
            desired_yaw = np.sign(dy) * np.pi / 2
        else:
            v = dx / DT
            desired_yaw = np.arctan(dy / dx)

        # === 회전만 수행하는 조건 ===
        if self.controller_type != "nomad":
            MAX_ROTATION_ONLY_ANGLE = np.deg2rad(30)
            if abs(desired_yaw) > MAX_ROTATION_ONLY_ANGLE:
                v = 0.0  # 직진 억제

        w = clip_angle(desired_yaw) / DT

        return float(np.clip(v, 0.0, MAX_V)), float(np.clip(w, -MAX_W, MAX_W))

    # ─────────────────────── timer ────────────────────────────
    def _timer_cb(self) -> None:
        vel_msg = Twist()

        if self.reached_goal:
            self.vel_pub.publish(vel_msg)  # publish zero velocity to halt
            self.get_logger().info("Reached goal – stopping controller.")
            rclpy.shutdown()
            return

        if self._waypoint_valid():
            v, w = self._pd_control(self.waypoint)
            if self.reverse_mode:
                v *= -1.0
            vel_msg.linear.x = v
            vel_msg.angular.z = w
            self.get_logger().debug(f"Publishing velocity: v={v:.3f}, w={w:.3f}")

        self.vel_pub.publish(vel_msg)


# ────────────────────────────────────────────────────────────────────────────────


def main(args=None):  # pragma: no cover
    rclpy.init(args=args)

    # 명령줄 인자 파싱 추가
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--control", type=str, default="nomad", help="control type (nomad, apf)"
    )
    args, unknown = parser.parse_known_args()

    node = PDControllerNode(controller_type=args.control)
    rclpy.spin(node)


if __name__ == "__main__":  # pragma: no cover
    main()
