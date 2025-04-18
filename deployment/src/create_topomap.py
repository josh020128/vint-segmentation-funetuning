from __future__ import annotations

import argparse
import os
import shutil
import time
from pathlib import Path
from typing import Optional

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, Joy

from utils import msg_to_pil  # existing helper converting Image → PIL.Image
from topic_names import IMAGE_TOPIC, TOPOMAP_IMAGES_DIR


def _remove_files_in_dir(dir_path: Path):
    for f in dir_path.iterdir():
        try:
            if f.is_file() or f.is_symlink():
                f.unlink()
            elif f.is_dir():
                shutil.rmtree(f)
        except Exception as e:
            print(f"Failed to delete {f}: {e}")


class TopomapCreator(Node):
    """Node that stores incoming images periodically to build a topomap."""

    def __init__(self, dir_name: str, dt: float):
        super().__init__("create_topomap")
        assert dt > 0.0, "dt must be positive"
        self.dir_path: Path = Path(TOPOMAP_IMAGES_DIR) / dir_name
        self.dir_path.mkdir(parents=True, exist_ok=True)
        if any(self.dir_path.iterdir()):
            self.get_logger().info("Target directory exists. Removing previous images…")
            _remove_files_in_dir(self.dir_path)

        self.dt = dt
        self.index = 0
        self.latest_img: Optional[Image] = None
        self.last_save_time = self.get_clock().now()

        # ROS interfaces
        self.create_subscription(Image, IMAGE_TOPIC, self._image_cb, 10)
        self.create_subscription(Joy, "joy", self._joy_cb, 10)
        self.create_timer(self.dt, self._timer_cb)
        self.get_logger().info(
            f"Recording images every {self.dt}s to {self.dir_path} (topic: {IMAGE_TOPIC})"
        )

    # ---------------------------------------------------------------------
    # Callbacks
    # ---------------------------------------------------------------------

    def _image_cb(self, msg: Image):
        self.latest_img = msg

    def _joy_cb(self, msg: Joy):
        if msg.buttons and msg.buttons[0]:
            self.get_logger().info("Button 0 pressed – shutting down.")
            rclpy.shutdown()

    def _timer_cb(self):
        if self.latest_img is None:
            return
        now = self.get_clock().now()
        if (now - self.last_save_time).nanoseconds < self.dt * 1e9:
            return  # not yet time
        try:
            pil_img = msg_to_pil(self.latest_img)
            out_path = self.dir_path / f"{self.index}.png"
            pil_img.save(out_path)
            self.get_logger().info(f"Saved frame {self.index} → {out_path.name}")
            self.index += 1
            self.last_save_time = now
        except Exception as e:
            self.get_logger().warn(f"Failed to save image: {e}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=f"Record images from {IMAGE_TOPIC} to build a topomap."
    )
    parser.add_argument(
        "--dir",
        "-d",
        default="topomap",
        help="Sub‑directory name under ../topomaps/images/",
    )
    parser.add_argument(
        "--dt",
        "-t",
        type=float,
        default=1.0,
        help="Interval (seconds) between saved frames",
    )
    args = parser.parse_args()

    rclpy.init()
    node = TopomapCreator(args.dir, args.dt)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
