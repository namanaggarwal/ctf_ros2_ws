FROM nvidia/cuda:12.6.0-cudnn-devel-ubuntu22.04

# ── ROS Humble install ────────────────────────────────────────────────────────
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    curl gnupg lsb-release \
  && curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
     -o /usr/share/keyrings/ros-archive-keyring.gpg \
  && echo "deb [arch=$(dpkg --print-architecture) \
     signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
     http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" \
     > /etc/apt/sources.list.d/ros2.list \
  && apt-get update && apt-get install -y \
    ros-humble-ros-base \
    ros-humble-tf2-ros \
    ros-humble-nav-msgs \
    ros-humble-visualization-msgs \
    python3-colcon-common-extensions \
    python3-pip \
    git \
  && rm -rf /var/lib/apt/lists/*

# ── dynus_interfaces ──────────────────────────────────────────────────────────
RUN mkdir -p /opt/dynus_ws/src \
  && git clone https://github.com/kotakondo/dynus_interfaces.git \
               /opt/dynus_ws/src/dynus_interfaces \
  && bash -c "source /opt/ros/humble/setup.bash && \
              cd /opt/dynus_ws && \
              colcon build --packages-select dynus_interfaces"

# ── Python dependencies ───────────────────────────────────────────────────────
COPY requirements_marl_ctf.txt /tmp/requirements_marl_ctf.txt
RUN pip3 install --no-cache-dir -r /tmp/requirements_marl_ctf.txt

# ── Entrypoint ────────────────────────────────────────────────────────────────
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

WORKDIR /ros2_ws
ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
