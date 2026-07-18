# Multi Rover Capture the Flag

## Setup

Clone the MIGHTY repo and follow the installation steps
```bash
mkdir -p ~/code
cd ~/code
git clone https://github.com/mit-acl/mighty.git mighty_ws/src/mighty
```

Also clone and build the `mighty` branch of the mpc package in the same repo
```bash
cd ~/code/mighty_ws
git clone -b mighty https://gitlab.com/mit-acl/ugv/ugv_control/mpc.git src
colcon build --packages-select mpc
```

Clone and build the `ctf_ros2_ws` repo
```bash
cd ~/code
git clone https://github.com/namanaggarwal/ctf_ros2_ws.git ctf_ros2_ws
cd ctf_ros2_ws && colcon build
```

Before running anything, set your workspace path as an environment variable.
Add the following lines to your `~/.bashrc`:
```bash
export CTF_WS=<path/to/ctf_ws>
export MIGHTY_WS=<path/to/mighty_ws>
```

Then source your bashrc:
```bash
source ~/.bashrc
```

---

## Update Config
Update the `policy_zip_path` in the `ctf_rover/config` yaml files to your directory.

---

## Running in Simulation

From your workspace root, run:
```bash
python3 src/ctf_game_server/launch/game_server_sim.launch.py --mode swap-multiagent-ground --setup-bash ~/code/mighty_ws/install/setup.bash
```

This will launch the multi-agent ground robot swap simulation in RViz.

---

## Running on Hardware

**Host side:**

Run the CTF game server:
```bash
ros2 launch ctf_game_server game_server.launch.py
```

Run CTF RViz:
```bash
rviz2 -d /home/${USER}/code/ctf_ros2_ws/src/ctf_game_server/rviz/ctf_rviz_config.rviz
```

**Rover side (SSH into each rover):**

Run the CTF rover node:
```bash
python3 /home/${USER}/code/ctf_ros2_ws/src/ctf_rover/tmux/run_hw_red_rover_ctf.py --odom-type mocap
```

> **Note:** The rover node has a `use_hardware` parameter that controls whether it applies the world→map TF transform (needed on real hardware without mocap) or uses ground-truth pose directly (sim, or hardware with mocap). Confirm `use_hardware` is set correctly for your run — either via the launch/config files or as a launch argument — before running on physical rovers.