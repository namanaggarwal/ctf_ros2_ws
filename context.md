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

### 2026-04-13
- **Bug fix (`rover_node.py`)**: Fixed `RuntimeError: mat1 and mat2 shapes cannot be multiplied (89×26 and 32×64)` on first GNN inference. Policy checkpoints (`blue_mappo_final.zip`, `iter3_red_br.zip`) were trained with `obs_version=3` (F=16 node features, MPNN message dim = 2×16 = 32). `_init_policy()` was creating `GraphCTF` with `obs_version=2` (F=13, message dim = 26). Fixed by changing `obs_version=2` → `obs_version=3` and `get_observation_v2` → `get_observation_v3` (×2) in `rover_node.py`.
