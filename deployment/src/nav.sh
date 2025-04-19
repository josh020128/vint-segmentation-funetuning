#!/bin/bash

SESSION_NAME="navigate_control"

# 실행할 경로
PROJECT_DIR=~/visualnav-transformer  # TODO: 여기에 실제 프로젝트 경로로 수정

# Conda 환경 이름
CONDA_ENV_NAME=nomad_train  # 필요시 수정

# 시작 전 이전 세션 종료
tmux kill-session -t $SESSION_NAME 2>/dev/null

# 새 tmux 세션 생성
tmux new-session -d -s $SESSION_NAME

# Pane 0: navigate.py
tmux send-keys -t $SESSION_NAME "cd $PROJECT_DIR" C-m
tmux send-keys -t $SESSION_NAME "conda activate $CONDA_ENV_NAME" C-m
tmux send-keys -t $SESSION_NAME "source /opt/ros/humble/setup.bash" C-m
tmux send-keys -t $SESSION_NAME "python ./deployment/src/navigate.py" C-m

# Pane 1: pd_controller.py
tmux split-window -h -t $SESSION_NAME
tmux send-keys -t $SESSION_NAME "cd $PROJECT_DIR" C-m
tmux send-keys -t $SESSION_NAME "conda activate $CONDA_ENV_NAME" C-m
tmux send-keys -t $SESSION_NAME "source /opt/ros/humble/setup.bash" C-m
tmux send-keys -t $SESSION_NAME "python ./deployment/src/pd_controller.py" C-m

tmux attach-session -t $SESSION_NAME
