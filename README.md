# CTF Mighty Ground Simulation

## Setup

Before running the simulation, set your workspace path as an environment variable.

Add the following line to your `~/.bashrc`:

```bash
export MIGHTY_WS=<path/to/mighty_ws>
```

Then resource your bashrc:

```bash
source ~/.bashrc
```

---

## Running the Simulation

From your workspace root, run:

```bash
python3 src/ctf_game_server/launch/game_server_sim.launch.py --mode swap-multiagent-ground
```

This will launch the multi-agent ground robot swap simulation in RViz.
