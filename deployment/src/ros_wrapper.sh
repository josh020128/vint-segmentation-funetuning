#!/bin/bash
source ~/.bashrc       # (또는 source /opt/ros/humble/setup.bash)
exec "$@"              # 전달받은 명령어 실행
