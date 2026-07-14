# CTF Mighty Ground Simulation

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

Clone and built the `mighty_sim` branch of the ctf_repo
```bash
cd ~/code
git clone -b mighty_sim https://github.com/namanaggarwal/ctf_ros2_ws/tree/mighty_sim ctf_ros2_ws
cd ctf_ros2_ws && colcon build
```

Before running the simulation, set your workspace path as an environment variable.
Add the following lines to your `~/.bashrc`:
```bash
export CTF_WS=<path/to/ctf_ws>
export MIGHTY_WS=<path/to/mighty_ws>
```

Then resource your bashrc:
```bash
source ~/.bashrc
```

---

## Update Config
Update the `policy_zip_path` in the `ctf_rover/config` yaml files to your directory

## Running the Simulation

From your workspace root, run:
```bash
python3 src/ctf_game_server/launch/game_server_sim.launch.py --mode swap-multiagent-ground --setup-bash ~/code/mighty_ws/install/setup.bash
```

This will launch the multi-agent ground robot swap simulation in RViz.