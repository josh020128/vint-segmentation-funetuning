#!/usr/bin/env python3
"""pd_controller_ros2.py – ROS 2 version of the original PD controller node.

Behaviour
---------
* Listens to WAYPOINT_TOPIC (`Float32MultiArray`) and REACHED_GOAL_TOPIC (`Bool`).
* Computes linear/angular velocity commands with a simple PD‑style heuristic.
* Publishes geometry_msgs/Twist on the velocity topic defined in robot.yaml.
* Continuously measures and reports distance traveled.
* Also checks total elapsed time and triggers goal reached when time exceeds 1 minute.

Run after installing your Python package (or directly with `ros2 run` / `python`).
"""
from __future__ import annotations

import time
from typing import Optional, Tuple

import numpy as np
import yaml
import argparse

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
DISTANCE_REPORT_INTERVAL: float = 0.1  # 이동 거리 보고 간격(초)
MAX_TIME: float = 30000.0  # 최대 실행 시간 (1분)


# ────────────────────────────────────────────────────────────────────────────────
class PDControllerNode(Node):
    """ROS 2 node implementing a planar PD controller."""

    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__("pd_controller")
        self.controller_type = args.control

        # 로봇 타입에 따른 이미지 토픽 선택
        if args.robot == "locobot":
            waypoint_topic = "/robot1/waypoint"
            vel_topic = "/robot1/cmd_vel"
        elif args.robot == "locobot2":
            waypoint_topic = "/robot3/waypoint"
            vel_topic = "/robot3/cmd_vel"
        elif args.robot == "robomaster":
            waypoint_topic = "/robot3/waypoint"
            vel_topic = "/cmd_vel"
        elif args.robot == "turtlebot4":
            waypoint_topic = "/robot2/waypoint"
            vel_topic = "/robot2/cmd_vel"
        else:
            raise ValueError(f"Unknown robot type: {args.robot}")

        self.get_logger().info(f"로봇 타입: {args.robot}, 사용 토픽: {waypoint_topic}, {vel_topic}")

        # 내부 상태
        self.waypoint: Optional[np.ndarray] = None
        self._last_wp_time: float = 0.0
        self.reached_goal: bool = False
        self.reverse_mode: bool = False  # flip linear.x if required

        # 이동 거리 측정을 위한 변수 추가
        self.total_distance: float = 0.0
        self.last_velocity_time: float = time.time()
        self.last_report_time: float = time.time()
        self.current_velocity: float = 0.0

        # 총 시간 측정을 위한 변수 추가
        self.start_time: float = time.time()
        self.total_time: float = 0.0

        # pubs / subs - 선택된 토픽 사용
        self.vel_pub = self.create_publisher(Twist, vel_topic, 1)
        self.create_subscription(
            Float32MultiArray, waypoint_topic, self._waypoint_cb, 1
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
        self.get_logger().debug(f"Waypoint received: {self.waypoint.tolist()}")

    def _goal_cb(self, msg: Bool) -> None:
        self.reached_goal = msg.data
        if self.reached_goal:
            self.get_logger().info(f"최종 이동 거리: {self.total_distance:.3f} 미터")
            self.get_logger().info(f"총 소요 시간: {self.total_time:.3f} 초")

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

    def _update_distance(self, velocity: float) -> None:
        """현재 속도를 기반으로 이동 거리 업데이트"""
        current_time = time.time()
        dt = current_time - self.last_velocity_time

        # 현재 속도로 이동한 거리 계산 (속도 * 시간)
        if velocity > 0.0:  # 양수 속도일 때만 거리 계산
            distance = velocity * dt
            self.total_distance += distance
            if self.total_distance > 100.0:
                self._goal_cb(Bool(data=True))  # 5m 이동 시 목표 도달로 간주

        self.last_velocity_time = current_time

        # 일정 간격으로 이동 거리 보고
        if current_time - self.last_report_time >= DISTANCE_REPORT_INTERVAL:
            self.get_logger().info(f"현재 이동 거리: {self.total_distance:.3f} 미터")
            self.last_report_time = current_time

    # ─────────────────────── timer ────────────────────────────
    def _timer_cb(self) -> None:
        vel_msg = Twist()

        # 총 시간 업데이트
        current_time = time.time()
        self.total_time = current_time - self.start_time

        # 1분 경과 시 목표 도달로 처리
        print("현재 이동 시간:", self.total_time)
        if self.total_time > MAX_TIME:
            self.get_logger().info(f"최대 시간({MAX_TIME}초) 초과 - 목표 도달 처리")
            self._goal_cb(Bool(data=True))

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

            # 현재 속도 저장 및 이동 거리 업데이트
            self.current_velocity = abs(v)  # 절대값 사용 (방향 무관)
            self._update_distance(self.current_velocity)

            self.get_logger().debug(f"Publishing velocity: v={v:.3f}, w={w:.3f}")
        else:
            # waypoint가 없을 때도 이동 거리 업데이트 (속도 0으로)
            self._update_distance(0.0)

        self.vel_pub.publish(vel_msg)


# ────────────────────────────────────────────────────────────────────────────────


def main(args=None):  # pragma: no cover
    rclpy.init(args=args)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--control", type=str, default="nomad", help="control type (nomad, apf)"
    )
    parser.add_argument(
        "--robot",
        type=str,
        default="locobot",
        choices=["locobot", "locobot2", "robomaster", "turtlebot4"],
        help="Robot type (locobot, robomaster, turtlebot4)",
    )
    args, unknown = parser.parse_known_args()

    # 로봇 타입을 노드에 직접 전달
    node = PDControllerNode(args)
    rclpy.spin(node)


if __name__ == "__main__":  # pragma: no cover
    main()
