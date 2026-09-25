#!/bin/bash
# 機器人端：固定使用 rmenv
cd "$(dirname "$0")"
exec ~/rmenv/bin/python -u robot_server.py --mode "${1:-sim}" "${@:2}"
