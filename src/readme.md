Workspace: ctf_ws/
└── src/
    ├── ctf_msgs/       <-- Custom ROS2 messages
    ├── ctf_server/     <-- Game server node
    └── ctf_rover/      <-- Rover nodes

────────────────────────────────────────────
Build Phase (colcon build)
────────────────────────────────────────────

1️⃣ Build messages first
colcon build --packages-select ctf_msgs
source install/setup.bash

   [ctf_msgs]
   ├─ Generates Python modules for msg definitions
   ├─ Generates C++ headers (if needed)
   └─ Populates install/lib/python3.x/site-packages/ctf_msgs/msg/

2️⃣ Build dependent packages
colcon build
source install/setup.bash

   [ctf_server] → depends on ctf_msgs
       ├─ Can now import JoinGameMessage, GameState, etc.
       └─ Compiled Python/C++ nodes ready to run

   [ctf_rover] → depends on ctf_msgs
       ├─ Can now import JoinGameMessage
       └─ Node scripts ready to run

────────────────────────────────────────────
Runtime Phase (ros2 run / launch)
────────────────────────────────────────────

  [ROS2 Runtime Environment]
  ├─ Source workspace: source install/setup.bash
  ├─ Launch server node
  │   ros2 run ctf_server game_server_node
  │       ↓ subscribes/publishes messages from ctf_msgs
  ├─ Launch rover nodes (multiple instances)
  │   ros2 run ctf_rover rover_node --ros-args -p rover_name:=rover1 -p team:=red
  │       ↓ publishes JoinGameMessage
  │       ↓ subscribes to GameState messages
  └─ ROS2 topics/services flow
        JoinGameMessage → GameServer
        PoseStamped → GameServer
        GameState → Rover nodes

────────────────────────────────────────────
Notes:
- The server and rover nodes **cannot import ctf_msgs** before it is built and sourced.
- Using `--symlink-install` allows Python changes to be live without rebuilding.
- Multiple rover nodes can run simultaneously using namespaces or parameters.
