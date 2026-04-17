# CTF ROS2 Workspace — Context

## Overview

This workspace implements a **Capture-the-Flag (CTF) robot game** using ROS2 (Humble). Physical ground rovers are coordinated via a central game server node. The game server assigns initial positions and sends navigation goals; each rover translates those into local-frame commands for its onboard planner (DYNUS/Mighty).

---

## Repository Structure

```
ctf_ros2_ws/
  src/
    ctf_game_server/     # Central game authority
    ctf_rover/           # Per-rover ROS2 node
    ctf_msgs/            # Custom message/service definitions
```

---

## Packages

### `ctf_msgs`
Custom ROS2 interfaces used across the game.

| File | Type | Purpose |
|---|---|---|
| `JoinGameMessage.msg` | msg | `rover_name`, `rover_team_name` (unused in favour of the service) |
| `ServerToRoverMessage.msg` | msg | `command` (string) + `commanded_goal` (dynus_interfaces/State) |
| `GameState.msg` | msg | (empty — placeholder) |
| `JoinGame.srv` | srv | Request: `rover_name`, `rover_team_name` → Response: `accepted` (bool), `message` (string) |

---

### `ctf_game_server`

**Node:** `GameServer` (`game_server.py`) — ROS2 node name `"ctf"`

#### Responsibilities
1. **Rover registration** — exposes `/ctf/join_game` service; rovers call it on startup to register their name and team.
2. **Game initialization** — once `total_rovers` have joined, calls `reset()` to randomly sample starting grid positions, converts them to Vicon (world) frame coordinates, and publishes `ServerToRoverMessage` with `command = 'INIT'` to each rover's `/{rover_name}/server_to_rover` topic.
3. **Pose tracking** — subscribes to each rover's pose topic after it joins:
   - VICON mode: `/{rover_name}/world` → `PoseStamped` → `vicon_callback`
   - DLIO mode: `/{rover_name}/dlio/odom_node/odom` → `Odometry` → `dlio_callback` (converts local DLIO pose to world frame using the stored `X_world_map` transform)
4. **Coordinate conversion** — `sim_frame_to_vicon_frame()` converts discrete grid `(x, y, heading)` to continuous Vicon/world-frame `(x, y, z, yaw)` using a fixed affine transform (90° yaw rotation, grid spacing `a = 0.762 m`).
5. **Debug visualization** — publishes sphere `Marker` messages on `initial_pose_marker` (red for RED team, blue for BLUE team).

#### Key Parameters (`config/params.yaml`)
| Parameter | Default | Meaning |
|---|---|---|
| `grid_size` | 8 | NxN discrete game grid |
| `goal_height` | -0.01 | Z coordinate sent in all goals |
| `total_rovers` | 2 | Number of rovers before game starts |
| `seed` | 0 | RNG seed for spawn positions |
| `ctf_player_config` | `"2v2"` | Game mode (informational) |
| `use_dlio` | False | Use DLIO odometry instead of VICON |

#### Spawn Logic (`reset()`)
- **Blue team** spawns in rows `y ∈ [0, blue_init_spawn_y_lim]` (bottom of grid).  
  Possible initial headings: `[0, 1, 2, 3]` → (East, NE, North, NW).
- **Red team** spawns in rows `y ∈ [grid_size - red_init_spawn_y_lim - 1, grid_size]` (top of grid).  
  Possible initial headings: `[0, 7, 6, 5]` → (East, SE, South, SW).
- Collision-free: re-samples until no two same-team agents share the same `(x, y)`.

#### Coordinate Frames
- **Sim/game frame** — discrete grid, origin at corner, `a = 0.762 m` spacing, X = East, Y = North.
- **Vicon/world frame** — physical lab frame. Sim→Vicon transform: 90° CCW yaw, translation `(4a + Δy, -(4a + Δx), 0)` where `Δ = a/2`.
- **`/{rover_name}/map`** — rover-local frame used by DYNUS; origin = rover's initial world-frame pose.

#### Launch
```bash
ros2 launch ctf_game_server game_server.launch.py
```

---

### `ctf_rover`

**Node:** `RoverNode` (`rover_node.py`) — ROS2 node name `"ctf"`

#### Responsibilities
1. **Identity** — rover name built from env vars `VEHTYPE + VEHNUM` (e.g. `RR03`). Team read from ROS parameter `team` (default `"RED"`).
2. **Join handshake** — on startup, calls `/ctf/join_game` service with its name and team; waits until the service is available.
3. **TF initialization** (`initialize_tf`) — after joining, polls (timer at 5 Hz) for the TF from `world` → `{rover_name}`. Once found, stores `X_map_world = inv(X_world_rr)` and broadcasts a static TF `world → {rover_name}/map` (so DYNUS can plan in its local frame). Timer cancels itself once initialized.
4. **Goal forwarding** (`server_to_rover_callback`) — subscribes to `/{rover_name}/server_to_rover`. On `command == 'INIT'`:
   - Takes the world-frame `State` goal from the server message.
   - Transforms position `p_local = X_map_world @ p_global` and velocity `v_local = R_map_world @ v_global`.
   - Publishes as `PoseStamped` (or `State` if `USE_VEL = True`) to `/{rover_name}/term_goal` (DYNUS terminal goal topic).

#### Topics Summary
| Direction | Topic | Message Type |
|---|---|---|
| Subscribe | `/{rover_name}/server_to_rover` | `ServerToRoverMessage` |
| Publish | `/{rover_name}/term_goal` | `PoseStamped` (or `State`) |
| Service call | `/ctf/join_game` | `JoinGame` |
| TF broadcast (static) | `world → {rover_name}/map` | TransformStamped |

#### Launch (per rover, from tmux config)
```bash
ros2 run ctf_rover rover_node --ros-args -p team:="BLUE"
# Requires VEHTYPE and VEHNUM env vars to be set
```

---

### Supporting Nodes (`ctf_rover`)

#### `PublishGlobalGoal` (`publish_global_goal.py`)
Utility/debug node. Publishes a hardcoded goal `[2.0, 0.0, 0.0]` in the world frame to `/{rover_name}/global_term_goal` every 2 seconds. Used for testing the goal pipeline without the game server.

#### `GlobalToLocalGoal` (`global_to_local_goal.py`)
Standalone version of the coordinate transform logic in `RoverNode`. Subscribes to `/{rover_name}/global_term_goal` (world-frame `State`) and publishes the transformed `/{rover_name}/term_goal` (local-frame `State`) for DYNUS. Uses the same TF initialization pattern (`X_map_world`).

---

## Full System Flow

```
[Rovers boot]
    └─ RoverNode starts, waits for /ctf/join_game
    └─ Calls join_game service with rover_name + team

[GameServer: handle_join_request]
    └─ Registers rover, creates pose subscriber + server→rover publisher
    └─ If total_rovers reached: calls pre_start_game_utils() + start_game_callback()

[GameServer: start_game_callback]
    └─ reset() → samples random initial (x, y, heading) per agent in sim frame
    └─ sim_frame_to_vicon_frame() → converts to world (x, y, z, yaw)
    └─ Publishes ServerToRoverMessage{command='INIT', commanded_goal=State} to each rover

[RoverNode: server_to_rover_callback]
    └─ On 'INIT': transforms world goal → local (map) frame
    └─ Publishes PoseStamped to /{rover_name}/term_goal → consumed by DYNUS planner
```

---

## External Dependencies
- **ROS2 Humble**
- **DYNUS / Mighty** — onboard motion planner, consumes `/{rover_name}/term_goal`
- **dynus_interfaces/State** — position + velocity + quaternion message used for planner goals
- **tf2_ros** — coordinate frame management
- **scipy** — rotation conversions
- **gymnasium** — seeded RNG (`gymnasium.utils.seeding.np_random`)
- **Livox MID360 + DLIO** — optional lidar-inertial odometry (activated via `use_dlio` param)
- **VICON mocap** — default pose source (`/{rover_name}/world`)

---

## Known TODOs / Incomplete Areas
- `GameState.msg` is empty — game state service (`ctf/get_state`) is commented out in both nodes.
- `dlio_callback` in `game_server.py` transforms the pose into world frame but never saves/uses the result (missing assignment at the end).
- `discrete_grid_abstraction_to_highbay_coordinates()` references undefined variables (`a`, `delta_x`, `tft`) — dead/legacy code.
- `vicon_callback` appears in `rover_node.py` but `RoverNode` doesn't subscribe to any pose topic — copy-paste remnant from the game server.
- `USE_VEL = False` is hardcoded in `RoverNode`; the `State`-based publisher path is unreachable.
- `seed()` and `_heading_to_direction_vector()` in `RoverNode` are not used — copied from game server.

## Changelog

### 2026-04-17
- **Physical tagging layer (`game_server.py`, `rover_node.py`)**: Implemented end-to-end physical tagging. When the rover policy outputs a tag action, `_check_and_publish_tag()` validates two conditions using live Vicon data: (1) the closest enemy rover is within `tag_radius` (default 0.55 m), and (2) the tagger's yaw is within `tag_angle_tolerance` (default 45°) of the bearing to that rover. If both pass, the rover publishes `"tagger_rr:tagged_rr"` to `/ctf/tag_event`. The game server re-validates both conditions against its own authoritative Vicon poses, enforces a per-rover `tag_cooldown` (default 10 s), then calls `_send_respawn()` which samples a random point from the tagged team's spawn zone (sim coords → Vicon), builds a `ServerToRoverMessage{command='RESPAWN'}`, and dispatches it. The rover handles `RESPAWN` in `server_to_rover_callback` by converting the Vicon goal to local map frame, publishing to `term_goal`, updating `goal_node_idx`, and clearing `_waiting_to_depart` so normal policy resumes on arrival. `all_rover_poses` extended from 3-element `[x,y,z]` to 7-element `[x,y,z,qx,qy,qz,qw]` to carry orientation; all existing `[:2]`/`[:3]` slices remain valid. New ROS parameters: `tag_radius`, `tag_angle_tolerance`, `tag_cooldown`. Post-tag respawn flow: (1) tagged rover overrides current `term_goal` with respawn location and updates `goal_node_idx`; (2) since `_game_started=True`, arrival at respawn triggers `policy_step()` directly with no re-handshake — game resumes seamlessly; (3) `_spawn_reached_sent` and `_game_running` remain True throughout, so the rover re-enters normal play immediately; (4) `env.state[ctf_agent]` is snapped to the respawn node on the first `_update_env_state()` inside the next `policy_step()`, so the policy sees the correct position straight away.
- **Synchronised game start (`game_server.py`, `rover_node.py`)**: Replaced the independent per-rover 5 s countdown with a server-coordinated barrier. Each rover publishes its name to `/ctf/rover_ready` upon first arrival at its spawn node (`_spawn_reached_sent` prevents duplicate publishes). The game server collects confirmations in `_rovers_at_spawn`; once `total_rovers` have confirmed, `_broadcast_start()` sets `_game_running = True` and sends `ServerToRoverMessage{command='START'}` to all rovers simultaneously. Rovers handle `START` by setting `_game_started = True` and running the first `policy_step()`. Tag events are gated on `_game_running` (not `game_started`) so no spurious tags can fire during the spawn phase. Removed `_first_step_timer` and `_first_policy_step_once`.
- **One-shot flag discovery log (`rover_node.py`)**: Added `_flag_ambiguity_resolved` boolean; the `[FLAG RESOLVED]` warn-level log fires exactly once (the first time any Blue rover's nearest graph node is a frontier node), regardless of how many policy steps are spent in the frontier zone. Message includes physical rover name, CTF agent name, and frontier node ID.



### 2026-04-13 (updated 2026-04-14)
- **Coordinate transform update (`rover_node.py`, `game_server.py`)**: Axes redefined as `x_sim = -y_vicon, y_sim = +x_vicon` (proper 90° CW rotation, det=+1). `_R_SIM_VICON` split into `_R_SIM_TO_VICON = [[0,1],[-1,0]]` and `_R_VICON_TO_SIM = [[0,-1],[1,0]]`. `sim_frame_to_vicon_frame` heading formula corrected to `θ_vicon = θ_sim − π/2`. Blue/Red spawn headings explicitly set. Debug TF now uses valid rotation.
- **GraphCTF spawn (`game_server.py`)**: `compute_initial_poses()` replaced with GraphCTF-based logic — calls `ctf_env.reset()`, reads `node_pose_dict` for sim [x,y] per agent. Old 8×8 grid `reset()` removed. `customCTF.py` copied into game_server package.
- **Bug fix — stale pairwise distances (`rover_node.py`)**: `min_opp_distance` / `min_teammate_distance` (obs v3 features 12 & 13) were frozen at spawn since `env.step()` is never called. Fixed by recomputing BFS pairwise distances in `_update_env_state()` on every policy step.
- **Bug fix — policy_step spam (`rover_node.py`)**: `_world_state_callback` at ~100 Hz was calling `policy_step()` repeatedly while rover sat at goal. Fixed with `_waiting_to_depart` flag: blocks re-triggering until rover leaves goal area (`dist > arrival_tolerance`). Stay action holds position indefinitely by design.

### 2026-04-13
- **Game-start synchronisation (`rover_node.py`)**: Replaced fixed 2s post-INIT timer with arrival-triggered 5s countdown. After receiving INIT, the rover navigates to its spawn node; once `_world_state_callback` detects arrival (`dist < arrival_tolerance`) the first time (`_game_started = False`), a 5s one-shot timer starts. When it fires, `_game_started` is set to `True` and the first `policy_step()` runs. Subsequent goal arrivals trigger `policy_step()` immediately. Added `_game_started` flag to `RoverNode`.
- **Dead code removal (`game_server.py`)**: Removed `make_seeded_rngs()`, `reset()`, `_sample_init_heading()`, `_heading_to_direction_vector()`, `discrete_grid_abstraction_to_highbay_coordinates()`, `seed()` methods, associated unused imports (`random`, `functools`, `gymnasium.utils.seeding`), and the no-op `for rover in self.rovers_list` loop from `__init__`.

### 2026-04-13
- **Bug fix (`rover_node.py`)**: Fixed `RuntimeError: mat1 and mat2 shapes cannot be multiplied (89×26 and 32×64)` on first GNN inference. Policy checkpoints (`blue_mappo_final.zip`, `iter3_red_br.zip`) were trained with `obs_version=3` (F=16 node features, MPNN message dim = 2×16 = 32). `_init_policy()` was creating `GraphCTF` with `obs_version=2` (F=13, message dim = 26). Fixed by changing `obs_version=2` → `obs_version=3` and `get_observation_v2` → `get_observation_v3` (×2) in `rover_node.py`.
- **Bug fix (`game_server.py`)**: `sim_frame_to_vicon_frame()` used the old 8×8 grid transform (`a=0.762 m`, `yaw=−π/2`, `T=[3.429, 3.429]`), inconsistent with the graph-based transform in `rover_node._vicon_to_sim()` (`R=[[0,−1],[−1,0]]`, `T=[5,5]`, scale=1 m/unit). This caused the initial `current_node_idx` in `server_to_rover_callback` to snap to an arbitrary graph node. Fixed by replacing the old grid math with `vicon_xy = R @ sim_xy + [5,5]` and heading `θ_vicon = −π/2 − θ_sim`.
