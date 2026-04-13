# Integration Plan: Graph Policies on Physical Rovers

## Questions to Answer Before Implementation

---

### A. Policy Checkpoint

Which policy zip file(s) to load, and for which team(s)?

Available checkpoints:
- `blue_v4_hybrid_mappo_new_map/` — `blue_mappo_final.zip`, `blue_mappo_stage*.zip`
- `curriculum_red_v2/` — `red_curriculum_final.zip`, `red_stage*.zip`
- `psro_run=ctf_10x10_51_rf59_final/` — various `blue_*` and `red_*` seeds/levels
- `GNN_PPO_RedBR_opp_fh1_flag_uniform_best.zip`
- `curriculum_blue_ego6_s0/blue_stage0_ego6.zip`

**Answer:**
- Blue policy path: 'blue_v4_hybrid_mappo_new_map/blue_mappo_final.zip'
- Red policy path (or `heuristic` / `random` if not running a learned red): 'robust_psro_iter1/iter3_red_br.zip'
- MAPPO or IPPO? (determines how obs are batched across teammates): MAPPO

---

### B. Sim-to-Vicon Coordinate Transform

The basin-corridor graph (`make_basin_corridor_medium_v1`) has node positions in sim units:
- X ∈ [0, ~10], Y ∈ [-10, ~0]  (column index, negative row index, with small jitter)
- Nodes are ~1 sim-unit apart

These need to map to Vicon frame meters so DYNUS can navigate to them.

**Option 1** — Define an affine transform (scale + rotation + translation):
```
vicon_pos = scale * R @ sim_pos + translation
```
- Scale (meters per sim unit): My answer is, 1 meter per sim unit
- Rotation (yaw from sim to Vicon, degrees): My answer is, +y in VICON is -x in SIM and +x in VICON is -y in SIM. Figure out the Rotation yourself from this axes correspondence.
- Translation (Vicon position of sim origin [0,0], meters [x, y]): (+5 meters, +5 meters)

**Option 2** — Use the existing `sim_frame_to_vicon_frame()` grid logic (8×8 grid, `a = 0.762 m`), but swap in the GNN policy running on a regular grid instead of the basin-corridor graph.

**Answer (circle one and fill in):** Option 1 / Option 2

---

### C. Policy Step Trigger

When should the game server query the policy and send a new goal to each rover?

- [ ] **Fixed timer** — every N seconds regardless of whether the rover reached its goal
  - N (seconds): 
- [ ] **Goal-reached event** — only when a rover arrives within tolerance of its current goal node
  - Arrival tolerance (meters): 
- [ ] **Both** — timer fires as fallback if goal not reached within timeout

**Answer:** Instead of the game server sending new goals to each rover, I want each rover to compute it's next waypoint / goal from the saved graph policy (Red and Blue pre-computed GNN from path depending on team) and querying the state of the world from the server or VICON. Rover reads world state of all rovers; maps rover poses to an approximate node on the graph and then creates an observation_tensor based on GraphCTF in customCTF.py and forward passes it through the computed GNN to compute the next waypoint. Once it reaches the goal, it recomputes a new goal by the latest state of the world.

---

### D. Special Actions

The policy action space is `{0, ..., max_degree-1, stay, tag}`:
- `0..max_degree-1` → move to neighbor node
- `max_degree` → **stay** (no movement)
- `max_degree + 1` → **tag** (attempt to tag an opponent)

What should happen physically for:

- **Stay**: Hold current position (republish current goal)?
  **Answer:** Yes.

- **Tag**: Is the tag action present/relevant in the trained policy you're running? If so, what should the rover do? (e.g. hold position, spin, nothing)
  **Answer:** Do nothing. The rover which issues a tag does nothing, and the rover which is tagged is respawned according to the logic in GraphCtF in customCTF.py.

---

### E. Python Path / Imports

`graph_policy.py`, `graphpolicy.py`, and `customCTF.py` live in `/home/namanagg/marl_ctf/`. How should the game server access them?

- [ ] **`sys.path` injection** — add `/home/namanagg/marl_ctf/` at runtime inside the node
- [ ] **Symlink** into the ROS package source directory
- [ ] **Copy** the relevant files into `ctf_game_server/`
- [ ] **Install as a Python package** (add `marl_ctf` to the workspace and install via `setup.py`)

**Answer:** Copy the relevant files into ctf_game_server/

---

### F. Game Mode / Teams

- How many rovers are physically present and which teams? (e.g. 2 blue + 2 red, or 1 blue + 1 red)
  **Answer:** 2 blue + 2 red. If the physical game is run with lesser number of rovers than 3 such as 2 Red plus 1 Blue -- populate the observation_tensor by teammate_distance = 1 (normalized by graph diameter hence 1) as a proxy for nearest ally.

- Should the game server control **both** teams with learned policies, or just one (with the other team as a heuristic/scripted opponent)?
  **Answer:** Rover-side policy step. Yes both teams.

- Should the flag positions be **fixed** (e.g. always `fixed_flag_hypothesis=0`) or **sampled** at game start?
  **Answer:** Fixed at flag_hypothesis=1.

---

### G. Goal Height

The current `goal_height = -0.01` m is used for all goals (rovers fly just below zero). Should graph node goals use the same height?

**Answer:** Yes.

---

## Architecture Summary (confirmed: rover-side policy step)

Rovers compute their own waypoints via local GNN inference. The game server handles only spawn/init and world-state broadcasting. Rovers subscribe to global rover poses (from server or VICON directly).

```
GameServer
├── Handles /ctf/join_game registration (existing)
├── On all rovers joined: assigns spawn positions, publishes 'INIT' (existing)
└── Publishes shared world-state topic (all rover poses) for rovers to consume

RoverNode (extended — main changes here)
├── GraphCTF instance         — graph topology, flag/spawn config (flag_hypothesis=1)
├── LearnedPolicyWrapper      — loads team-specific .zip at startup
├── World-state subscriber    — reads all rover poses (server broadcast or VICON)
├── On 'INIT' received:
│   ├── Set current_node = nearest graph node to spawn pose
│   └── Trigger first policy_step()
├── policy_step()
│   ├── Map all rover poses → nearest graph nodes
│   ├── Build observation tensor (GraphCTF format)
│   ├── policy.forward() → action int
│   ├── action → neighbor node → sim coords → Vicon coords → local (map) frame
│   └── Publish PoseStamped to /{rover_name}/term_goal → DYNUS
└── On goal-reached (arrival within tolerance of current node):
    └── Re-query world state → policy_step()
```

ServerToRoverMessage commands:
- `'INIT'` — existing, sends initial spawn pose (no new server→rover commands needed)

---

## Implementation Decisions (confirmed)

### Decentralized GraphCTF instances
Each `RoverNode` owns its own `GraphCTF(fixed_flag_hypothesis=1)` instance. GraphCTF is used purely as an **obs builder and graph data store** — `env.step()` is never called. The rover manually writes real-world state into `env.agent_nodes` (and related fields) before calling `env.get_observation_v2()`.

### Flag discovery synchronization
Blue flag discovery (`env.blue_flag_known`) is inferred **locally on each rover** — no extra comms needed. Since every rover already receives all rover poses via the world-state topic, each Blue rover independently checks whether any Blue rover's nearest graph node is in the frontier zone (`env.frontier_nodes`), and sets `env.blue_flag_known = True` accordingly. This mirrors the centralized logic in `GraphCTF` without requiring a dedicated flag-discovery broadcast.

### Coordinate transforms
- **Vicon → sim**: `sim_pos = R.T @ (vicon_pos - translation) / scale`
  - `scale = 1.0` m/unit
  - `R`: rotation where +x_vicon = -y_sim, +y_vicon = -x_sim → 90° CW in the sim→vicon direction, so R = [[0, -1], [1, 0]] (sim→vicon), R.T = [[0, 1], [-1, 0]] (vicon→sim)
  - `translation = [5.0, 5.0]` m (Vicon position of sim origin)
- **Sim → Vicon**: `vicon_pos = scale * R @ sim_pos + translation`
- **Vicon → local map frame**: existing `X_map_world` transform in `RoverNode`

### Missing teammate handling
If fewer than 2 teammates are registered (e.g. 1 Blue + 2 Red), set `min_teammate_distance_to_ego = 1.0` (normalized by graph diameter) as a proxy — already answered in Section F.

---

## Implementation Plan

### Files

| File | Action |
|---|---|
| `src/ctf_rover/ctf_rover/rover_node.py` | Major extension (Steps 2–11 below) |
| `src/ctf_rover/ctf_rover/customCTF.py` | Copy from `/home/namanagg/marl_ctf/customCTF.py` |
| `src/ctf_rover/ctf_rover/graph_policy.py` | Copy from `/home/namanagg/marl_ctf/graph_policy.py` |

No changes needed to `game_server.py` — each rover subscribes to `/{name}/world` directly.

Note: `customCTF.py` imports `supersuit` (used only in training wrappers, not `GraphCTF`).
Install with `pip install supersuit` or wrap the import in `try/except ImportError: pass`.

---

### Coordinate Transform (derived)

Axes: +x_vicon = −y_sim, +y_vicon = −x_sim → R = [[0, −1], [−1, 0]] (self-inverse).

```
sim_to_vicon(p) = R @ p + [5, 5]
vicon_to_sim(p) = R @ (p − [5, 5])
```

---

### New ROS Parameters

| Parameter | Type | Example (Blue_0 rover) |
|---|---|---|
| `policy_zip_path` | str | `.../blue_mappo_final.zip` |
| `all_rover_names` | list[str] | `["RR01","RR02","RR03","RR04"]` |
| `all_rover_teams` | list[str] | `["BLUE","BLUE","RED","RED"]` |
| `all_rover_team_indices` | list[int] | `[0, 1, 0, 1]` |
| `team_index` | int | `0` (MAPPO head selector) |
| `arrival_tolerance` | float | `0.5` m |

Rover-to-CTF-agent mapping built from these at startup:
`rr_to_ctf = {"RR01": "Blue_0", "RR02": "Blue_1", "RR03": "Red_0", "RR04": "Red_1"}`

---

### New Methods in RoverNode

| Method | Purpose |
|---|---|
| `_init_policy()` | Instantiate `GraphCTF`, call `reset()`, load `LearnedPolicyWrapper` from zip. Called once on first INIT. |
| `_vicon_to_sim(xy)` | Apply affine inverse to get sim-frame coords |
| `_sim_to_vicon(xy)` | Apply affine forward to get Vicon-frame coords |
| `_nearest_node_idx(sim_xy)` | L2 nearest-node lookup over `env.node_pose_dict` |
| `_update_env_state()` | Map all rover Vicon poses → graph nodes → write `env.agent_nodes`. Also infer `env.blue_flag_known` locally for Blue rovers. |
| `policy_step()` | Call `_update_env_state()`, build obs via `env.get_observation_v2()`, call `policy.batch_action()`, decode action → next node → publish `PoseStamped` to `term_goal`. |
| `_world_state_callback(msg, name)` | Store latest Vicon pose. If this rover's pose arrives and is within `arrival_tolerance` of `goal_node`, trigger `policy_step()`. |
| `_first_policy_step_once()` | One-shot 2s timer callback to fire the first `policy_step()` after INIT. |

---

### Modified: `server_to_rover_callback`

Existing INIT logic (TF + spawn goal publish) is kept. After it, append:
1. Call `_init_policy()` (once, guarded by `if self.env is None`)
2. Set `current_node_idx` and `goal_node_idx` from spawn Vicon pose
3. Start a 2s one-shot timer → `_first_policy_step_once()`

---

### World-State Subscriptions

Each rover subscribes to `/{name}/world` (PoseStamped) for all names in `all_rover_names`.
Callbacks store the latest pose in `self.all_rover_poses[name]`.

---

### MAPPO Obs Batching (decentralized)

Each rover constructs obs for itself **and** its teammate (both rovers have the same world
state, so teammate obs can be built locally). `batch_action([obs_agent0, obs_agent1])` is
called with the correct ordering so the right MAPPO actor head is used:

```python
if team_index == 0:
    actions = policy.batch_action([obs_self, obs_teammate])
    action = actions[0]
else:
    actions = policy.batch_action([obs_teammate, obs_self])
    action = actions[1]
```

---

### Dead Code to Remove

- `seed()`, `_heading_to_direction_vector()`, `vicon_callback()` — removed (copy-paste remnants from game server)

---

## Changelog (implemented 2026-04-12; updated 2026-04-13)

### New files
| File | Description |
|---|---|
| `src/ctf_rover/ctf_rover/customCTF.py` | Copied from `/home/namanagg/marl_ctf/customCTF.py` — GraphCTF environment used as obs builder |
| `src/ctf_rover/ctf_rover/graph_policy.py` | Copied from `/home/namanagg/marl_ctf/graph_policy.py` — LearnedPolicyWrapper and policy loading |
| `src/ctf_rover/config/params_blue_0.yaml` | ROS params for Blue team rover with team_index=0 |
| `src/ctf_rover/config/params_blue_1.yaml` | ROS params for Blue team rover with team_index=1 |
| `src/ctf_rover/config/params_red_0.yaml` | ROS params for Red team rover with team_index=0 |
| `src/ctf_rover/config/params_red_1.yaml` | ROS params for Red team rover with team_index=1 |

### Modified files
| File | Change |
|---|---|
| `src/ctf_rover/ctf_rover/rover_node.py` | Full rewrite — GNN policy integration (see Implementation Plan above) |
| `src/ctf_rover/setup.py` | Added config/ data_files so YAML params are installed with the package |
| `src/ctf_rover/tmux/tmux_ctf_launch.yaml` | Updated CTF pane command to use `--params-file` |

### Bug fixes (2026-04-13)

| File | Bug | Fix |
|---|---|---|
| `src/ctf_rover/ctf_rover/rover_node.py` | `RuntimeError: mat1 and mat2 shapes cannot be multiplied (89x26 and 32x64)` — policy's `mpnn1` linear layer expects input dim 32 (`2×16`) but received 26 (`2×13`). Root cause: `_init_policy()` instantiated `GraphCTF` with `obs_version=2` (F=13), while the loaded checkpoint (`blue_mappo_final.zip`, `iter3_red_br.zip`) was trained with `obs_version=3` (F=16). | Changed `obs_version=2` → `obs_version=3` in `_init_policy()`. Changed `get_observation_v2(...)` → `get_observation_v3(...)` (×2) in `policy_step()`. |
| `src/ctf_game_server/ctf_game_server/game_server.py` | `sim_frame_to_vicon_frame()` used the old 8×8 grid transform (`a=0.762 m`, `yaw=−π/2`, `T=[3.429, 3.429]`) while `rover_node._vicon_to_sim()` / `_sim_to_vicon()` use the graph-consistent transform (`R=[[0,−1],[−1,0]]`, `T=[5,5]`, `scale=1 m/unit`). This caused the initial `current_node_idx` snap in `server_to_rover_callback` (line 221) to map the spawn Vicon pose through the wrong inverse transform, landing on an arbitrary graph node. | Replaced the old grid transform in `sim_frame_to_vicon_frame()` with `vicon_xy = R2 @ sim_xy + T2` (R2=[[0,−1],[−1,0]], T2=[5,5]). Heading corrected to `θ_vicon = −π/2 − θ_sim` (derived analytically from applying R to unit direction vector). Debug TF publisher updated to match (3D R3, t3). |

### Attribute name corrections (vs. plan)
- `env.state[agent]` — agent positions (not `env.agent_nodes`)
- `env.enemy_flag_known[agent]` — per-agent dict (not a single team flag)
- `env.is_frontier[node]` — per-node bool dict (not `env.frontier_nodes` set)

---

## Launch Instructions

### Prerequisites
```bash
# On the physical machine, build the workspace:
cd ~/ctf_ros2_ws
source /opt/ros/humble/setup.bash
colcon build --packages-select ctf_rover ctf_game_server ctf_msgs
source install/setup.bash

# Ensure marl_ctf dependencies are installed:
pip install supersuit pettingzoo stable-baselines3 torch torch-geometric networkx gymnasium
```

### Before launching
Update `total_rovers` in `ctf_game_server/config/params.yaml` to match the number of
physical rovers (e.g. `4`).

Each rover's `config/params_*.yaml` now only contains three fields — no rover names,
teams, or indices to keep in sync:

```yaml
# params_blue_0.yaml (same as params_blue_1.yaml)
rover_node:
  ros__parameters:
    team: "BLUE"
    policy_zip_path: "/home/swarm/code/ctf_ros2_ws/install/ctf_rover/share/ctf_rover/policies/blue_mappo_final.zip"
    arrival_tolerance: 0.5
```

The full rover-to-CTF-agent mapping is assigned by the game server at join time and
broadcast to every rover inside the `INIT` message — no manual coordination needed.

### Game server (one machine or ground station)
```bash
ros2 launch ctf_game_server game_server.launch.py
```

### Each rover (run on the rover's onboard computer)
```bash
# Set rover identity (already set in the rover's .bashrc normally):
export VEHTYPE=RR
export VEHNUM=01   # change per rover: 01, 02, 03, 04

# Pick the params file for this rover's team — both rovers on the same team use the
# same file (team_index is assigned automatically by the server):
#   Blue rovers  →  params_blue_0.yaml  (or params_blue_1.yaml — identical)
#   Red rovers   →  params_red_0.yaml   (or params_red_1.yaml  — identical)

source ~/ctf_ros2_ws/install/setup.bash
PARAMS=$(ros2 pkg prefix ctf_rover)/share/ctf_rover/config/params_blue_0.yaml

ros2 run ctf_rover rover_node --ros-args --params-file $PARAMS
```

Or via tmux (edit the params filename in `tmux/tmux_ctf_launch.yaml` per rover):
```bash
tmuxinator start -p src/ctf_rover/tmux/tmux_ctf_launch.yaml
```

### Sequence of events after launch
1. Game server waits for `total_rovers` (4) join-game calls.
2. Each rover starts, calls `/ctf/join_game` with its name and team; the server assigns
   a CTF agent name (`Blue_0`, `Blue_1`, `Red_0`, `Red_1`) based on join order within
   each team.
3. Server assigns spawn positions, sends `INIT` to each rover. The `INIT` message
   includes the full roster (`roster_rover_names` / `roster_ctf_agent_names`).
4. Each rover on receiving `INIT`:
   - Builds `rr_to_ctf` and derives its own `ctf_agent_name` and `team_index` from
     the roster.
   - Subscribes to `/{name}/world` for all roster members.
   - Publishes spawn goal to DYNUS (existing behaviour).
   - Initialises its local `GraphCTF` env and loads its team GNN policy (~5–10 s).
   - After 2 s delay, fires first `policy_step()`.
5. On each `/{rover_name}/world` pose update:
   - If this rover is within `arrival_tolerance` (0.5 m) of its current goal node → `policy_step()` → new waypoint published to DYNUS.

### Tuning
| Parameter | Where | Notes |
|---|---|---|
| `arrival_tolerance` | `config/params_*.yaml` | Increase if rovers overshoot nodes; decrease for tighter tracking |
| `total_rovers` | `ctf_game_server/config/params.yaml` | Must equal number of physical rovers |
| First-step delay | `rover_node.py` line with `create_timer(2.0, ...)` | Increase if world-state hasn't arrived within 2 s of INIT |
