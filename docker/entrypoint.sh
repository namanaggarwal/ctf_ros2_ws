#!/usr/bin/env bash
set -eo pipefail

# ROS setup scripts use unbound variables — disable -u around them
set +u
source /opt/ros/humble/setup.bash
source /opt/dynus_ws/install/setup.bash
set -u

# Build workspace on first run (install/ doesn't exist yet)
if [ ! -f "/ros2_ws/install/setup.bash" ]; then
  echo "[entrypoint] Building ctf_ros2_ws..."
  cd /ros2_ws
  colcon build --symlink-install
fi

set +u
source /ros2_ws/install/setup.bash
set -u

exec "$@"
