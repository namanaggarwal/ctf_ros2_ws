import os
import sys
import numpy as np
import networkx as nx

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, TransformStamped
from dynus_interfaces.msg import State
from scipy.spatial.transform import Rotation as R

from ctf_msgs.msg import ServerToRoverMessage
import tf2_ros
from tf2_ros import TransformException
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster
from ctf_msgs.srv import JoinGame

# Coordinate transform: x_sim = -y_vicon, y_sim = +x_vicon, scale = 1 m/unit.
# sim → vicon: R_SIM_TO_VICON @ sim_xy + T   (90° CW rotation, det=+1)
# vicon → sim: R_VICON_TO_SIM @ (vicon_xy - T)  (90° CCW = transpose)
_R_SIM_TO_VICON = np.array([[0.0, 1.0], [-1.0, 0.0]])
_R_VICON_TO_SIM = np.array([[0.0, -1.0], [1.0, 0.0]])
_T_VICON = np.array([5.0, 5.0])  # Vicon position of sim origin [0, 0]


class RoverNode(Node):
    def __init__(self, **kwargs):
        # super().__init__("ctf")
        super().__init__("rover_node")
        self.ctf_player_config = kwargs.get('ctf_player_config', '2v2')
        self.goal_height = -0.01

        # Rover identity from environment variables
        vehtype = os.getenv("VEHTYPE")
        vehnum = os.getenv("VEHNUM")
        self.rover_name = vehtype + vehnum

        # ROS parameters
        self.declare_parameter("team", "RED")
        self.declare_parameter("policy_zip_path", "")
        self.declare_parameter("arrival_tolerance", 0.5)

        self.rover_team_name = self.get_parameter("team").value
        self.arrival_tolerance = self.get_parameter("arrival_tolerance").value

        # Roster and CTF agent identity — built from the server's INIT message.
        # Not available at startup; populated in server_to_rover_callback.
        self.rr_to_ctf = {}
        self.ctf_agent_name = None
        self.team_index = None

        self.get_logger().info(
            f"Rover {self.rover_name} | team={self.rover_team_name} | waiting for INIT roster"
        )

        # Join game service
        self.join_client = self.create_client(JoinGame, "/ctf/join_game")
        while not self.join_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Waiting for join service...")
        self.send_join_request()

        # Subscribe to server→rover commands
        self.create_subscription(
            ServerToRoverMessage,
            f"/{self.rover_name}/server_to_rover",
            self.server_to_rover_callback,
            10,
        )

        # Publisher for DYNUS terminal goal
        self.publisher_local_dynus_command_goal = self.create_publisher(
            PoseStamped,
            f"/{self.rover_name}/term_goal",
            10,
        )

        # TF infrastructure
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.world_map_broadcaster = StaticTransformBroadcaster(self)
        self.X_map_world = None
        self.global_frame = "world"
        self.initial_frame = self.rover_name
        self.local_frame = f"{self.rover_name}/map"
        self.init_timer = self.create_timer(0.2, self.initialize_tf)

        # World-state: populated once the server's INIT roster arrives
        self.all_rover_poses = {}

        # Policy / env (lazy — initialised on first INIT command)
        self.env = None
        self.policy = None
        self.policy_ready = False
        self.current_node_idx = None
        self.goal_node_idx = None
        self._first_step_timer = None
        # Prevents re-triggering policy_step until the rover physically departs the goal area.
        # Set True after every policy_step(); cleared once dist to goal > arrival_tolerance.
        self._waiting_to_depart = False
        # False until the rover arrives at its spawn node and the 5s countdown completes.
        self._game_started = False

        self.get_logger().info("RoverNode initialised, waiting for INIT from server.")

    # ------------------------------------------------------------------
    # Join-game handshake
    # ------------------------------------------------------------------

    def send_join_request(self):
        request = JoinGame.Request()
        request.rover_name = self.rover_name
        request.rover_team_name = self.rover_team_name
        future = self.join_client.call_async(request)
        future.add_done_callback(self.join_response_callback)

    def join_response_callback(self, future):
        try:
            response = future.result()
        except Exception as e:
            self.get_logger().error(f"Service call failed: {e}")
            return
        if response.accepted:
            self.get_logger().info("Successfully joined game.")
        else:
            self.get_logger().warn(f"Join rejected: {response.message}")

    # ------------------------------------------------------------------
    # TF initialisation (world → rover/map static broadcast)
    # ------------------------------------------------------------------

    def initialize_tf(self):
        if self.X_map_world is not None:
            return
        try:
            tf = self.tf_buffer.lookup_transform(
                self.global_frame, self.initial_frame, rclpy.time.Time()
            )
        except TransformException:
            self.get_logger().info("Waiting for initial TF…")
            return

        t = tf.transform.translation
        q = tf.transform.rotation
        r = R.from_quat([q.x, q.y, q.z, q.w])

        X_world_rr = np.eye(4)
        X_world_rr[:3, :3] = r.as_matrix()
        X_world_rr[:3, 3] = [t.x, t.y, t.z]

        T = TransformStamped()
        T.header.stamp = self.get_clock().now().to_msg()
        T.header.frame_id = "world"
        T.child_frame_id = self.local_frame
        T.transform.translation.x = t.x
        T.transform.translation.y = t.y
        T.transform.translation.z = t.z
        T.transform.rotation.x = q.x
        T.transform.rotation.y = q.y
        T.transform.rotation.z = q.z
        T.transform.rotation.w = q.w
        self.world_map_broadcaster.sendTransform(T)

        self.X_map_world = np.linalg.inv(X_world_rr)
        self.init_timer.cancel()
        self.get_logger().info("world→map transform saved.")

    # ------------------------------------------------------------------
    # Server → rover command handler
    # ------------------------------------------------------------------

    def server_to_rover_callback(self, msg: ServerToRoverMessage):
        command = msg.command
        self.get_logger().info(f"server_to_rover_callback: command={command}")

        if command == "INIT":
            # Build rr_to_ctf from the server-assigned roster (first INIT only).
            if not self.rr_to_ctf:
                for rr, ctf in zip(msg.roster_rover_names, msg.roster_ctf_agent_names):
                    self.rr_to_ctf[rr] = ctf
                self.ctf_agent_name = self.rr_to_ctf[self.rover_name]
                # team_index is the trailing digit in the CTF agent name (e.g. "Blue_1" → 1)
                self.team_index = int(self.ctf_agent_name.split("_")[1])
                self.all_rover_poses = {name: None for name in self.rr_to_ctf}
                for name in self.rr_to_ctf:
                    self.create_subscription(
                        PoseStamped,
                        f"/{name}/world",
                        lambda m, n=name: self._world_state_callback(m, n),
                        10,
                    )
                self.get_logger().info(
                    f"[INIT] Roster received: {self.rr_to_ctf} | "
                    f"ctf_agent={self.ctf_agent_name} team_index={self.team_index}"
                )

            self.initialize_tf()

            goal = msg.commanded_goal
            p = goal.pos
            self.get_logger().info(
                f"[ROVER] Converting global goal [{p.x:.3f}, {p.y:.3f}, {p.z:.3f}] to local frame"
            )

            global_point = np.array([p.x, p.y, p.z, 1.0])
            local_point = self.X_map_world @ global_point

            v = goal.vel
            local_vel = self.X_map_world[:3, :3] @ np.array([v.x, v.y, v.z])

            local_goal = PoseStamped()
            local_goal.header.stamp = goal.header.stamp
            local_goal.header.frame_id = self.local_frame
            local_goal.pose.position.x = local_point[0]
            local_goal.pose.position.y = local_point[1]
            local_goal.pose.position.z = self.goal_height
            local_goal.pose.orientation.w = 1.0
            self.publisher_local_dynus_command_goal.publish(local_goal)
            self.get_logger().info(
                f"[ROVER] Published spawn goal [{local_point[0]:.3f}, {local_point[1]:.3f}]"
            )

            # One-time policy initialisation
            if self.env is None:
                self._init_policy()

            # Snap spawn Vicon pose to nearest graph node
            spawn_sim = self._vicon_to_sim(np.array([p.x, p.y]))
            self.current_node_idx = self._nearest_node_idx(spawn_sim)
            self.goal_node_idx = self.current_node_idx
            self.get_logger().info(
                f"[ROVER] Spawn node idx={self.current_node_idx}"
            )

            # Policy step fires once the rover physically arrives at its spawn node
            # (detected in _world_state_callback) and a 5s countdown completes.

    # ------------------------------------------------------------------
    # Policy initialisation (called once on first INIT)
    # ------------------------------------------------------------------

    def _init_policy(self):
        # Make customCTF.py and graph_policy.py importable as top-level modules
        _pkg_dir = os.path.dirname(os.path.abspath(__file__))
        if _pkg_dir not in sys.path:
            sys.path.insert(0, _pkg_dir)

        from customCTF import GraphCTF
        from graph_policy import LearnedPolicyWrapper

        self.get_logger().info("Initialising GraphCTF environment…")
        self.env = GraphCTF(
            ctf_player_config=self.ctf_player_config,
            fixed_flag_hypothesis=1,
            obs_version=3,
        )
        self.env.reset()
        self.get_logger().info(
            f"GraphCTF ready: {self.env.num_nodes} nodes, "
            f"max_degree={self.env.max_degree}, diameter={self.env.graph_diameter}"
        )

        zip_path = self.get_parameter("policy_zip_path").value
        self.get_logger().info(f"Loading policy from {zip_path}…")
        self.policy = LearnedPolicyWrapper(zip_path, deterministic=True)
        self.policy_ready = True
        self.get_logger().info("Policy loaded and ready.")

    # ------------------------------------------------------------------
    # Coordinate transforms
    # ------------------------------------------------------------------

    def _vicon_to_sim(self, vicon_xy: np.ndarray) -> np.ndarray:
        """Vicon [x, y] (metres) → sim [x, y] (graph units)."""
        return _R_VICON_TO_SIM @ (vicon_xy - _T_VICON)

    def _sim_to_vicon(self, sim_xy: np.ndarray) -> np.ndarray:
        """Sim [x, y] (graph units) → Vicon [x, y] (metres)."""
        return _R_SIM_TO_VICON @ sim_xy + _T_VICON

    # ------------------------------------------------------------------
    # Nearest graph node lookup
    # ------------------------------------------------------------------

    def _nearest_node_idx(self, sim_xy: np.ndarray) -> int:
        """Return global node index of graph node closest to sim_xy."""
        best_idx, best_d2 = None, float('inf')
        for node_id, pos in self.env.node_pose_dict.items():
            d2 = (pos[0] - sim_xy[0]) ** 2 + (pos[1] - sim_xy[1]) ** 2
            if d2 < best_d2:
                best_d2 = d2
                best_idx = self.env.node_to_idx[node_id]
        return best_idx

    # ------------------------------------------------------------------
    # World-state callback + goal-reached detection
    # ------------------------------------------------------------------

    def _world_state_callback(self, msg: PoseStamped, rover_name: str):
        p = msg.pose.position
        self.all_rover_poses[rover_name] = np.array([p.x, p.y, p.z])

        # Check goal-reached only for this rover's own pose update
        if rover_name != self.rover_name or not self.policy_ready:
            return
        if self.goal_node_idx is None:
            return

        goal_node = self.env.idx_to_node[self.goal_node_idx]
        goal_sim = np.array(self.env.node_pose_dict[goal_node])
        goal_vicon = self._sim_to_vicon(goal_sim)
        dist = np.linalg.norm(np.array([p.x, p.y]) - goal_vicon)

        if self._waiting_to_depart:
            # Block re-triggering until the rover physically leaves the goal area.
            if dist > self.arrival_tolerance:
                self._waiting_to_depart = False
            return

        if dist < self.arrival_tolerance:
            if not self._game_started:
                # First arrival at spawn — start 5s countdown before beginning GNN inference.
                if self._first_step_timer is None:
                    self.get_logger().info(
                        f"[POLICY] Spawn reached (dist={dist:.3f}m). Game starts in 5s."
                    )
                    self._first_step_timer = self.create_timer(5.0, self._first_policy_step_once)
            else:
                self.get_logger().info(
                    f"[POLICY] Goal reached (dist={dist:.3f}m). Stepping policy."
                )
                self._waiting_to_depart = True
                self.policy_step()

    # ------------------------------------------------------------------
    # Env state sync
    # ------------------------------------------------------------------

    def _update_env_state(self):
        """Write real-world rover positions into the local GraphCTF instance."""
        for rr_name, ctf_name in self.rr_to_ctf.items():
            pose = self.all_rover_poses.get(rr_name)
            if pose is None:
                continue
            sim_xy = self._vicon_to_sim(pose[:2])
            node_idx = self._nearest_node_idx(sim_xy)
            self.env.state[ctf_name] = self.env.idx_to_node[node_idx]

        # Recompute pairwise distances for min_opp_distance / min_teammate_distance (obs features 12 & 13).
        # env.step() is never called so these must be updated here after every state write.
        self.env.min_opp_distance = {a: 1.0 for a in self.env.agents}
        self.env.min_teammate_distance = {a: 1.0 for a in self.env.agents}
        for agent_0, agent_1 in self.env.all_agent_pairings:
            if self.env.state[agent_0] is None or self.env.state[agent_1] is None:
                continue
            try:
                d = nx.shortest_path_length(
                    self.env.graph,
                    source=self.env.state[agent_0],
                    target=self.env.state[agent_1],
                ) / self.env.graph_diameter
            except nx.NetworkXNoPath:
                d = 1.0
            if agent_0[0] == agent_1[0]:
                self.env.min_teammate_distance[agent_0] = min(d, self.env.min_teammate_distance[agent_0])
                self.env.min_teammate_distance[agent_1] = min(d, self.env.min_teammate_distance[agent_1])
            else:
                self.env.min_opp_distance[agent_0] = min(d, self.env.min_opp_distance[agent_0])
                self.env.min_opp_distance[agent_1] = min(d, self.env.min_opp_distance[agent_1])

        # Flag discovery: any Blue rover in a frontier node → all Blue agents know flag
        if self.rover_team_name.upper() == "BLUE":
            for rr_name, ctf_name in self.rr_to_ctf.items():
                if not ctf_name.startswith("Blue"):
                    continue
                if self.env.enemy_flag_known.get(ctf_name, False):
                    # Already known — propagate to all Blue agents and stop checking
                    for cn in self.env.enemy_flag_known:
                        if cn.startswith("Blue"):
                            self.env.enemy_flag_known[cn] = True
                    break
                pose = self.all_rover_poses.get(rr_name)
                if pose is None:
                    continue
                sim_xy = self._vicon_to_sim(pose[:2])
                node_idx = self._nearest_node_idx(sim_xy)
                node = self.env.idx_to_node[node_idx]
                if self.env.is_frontier[node]:
                    self.get_logger().info(
                        f"[FLAG] {rr_name} in frontier zone — Blue flag discovered."
                    )
                    for cn in self.env.enemy_flag_known:
                        if cn.startswith("Blue"):
                            self.env.enemy_flag_known[cn] = True
                    break

    # ------------------------------------------------------------------
    # Policy step
    # ------------------------------------------------------------------

    def _first_policy_step_once(self):
        self._game_started = True
        self._first_step_timer.cancel()
        self._first_step_timer = None
        self._waiting_to_depart = False
        self.get_logger().info("[POLICY] 5s elapsed — game starting, running first policy step.")
        self.policy_step()
        self._waiting_to_depart = True  # rover must physically depart before next policy_step

    def policy_step(self):
        if not self.policy_ready:
            self.get_logger().warn("policy_step called but policy not ready.")
            return
        if self.X_map_world is None:
            self.get_logger().warn("policy_step called but TF not ready.")
            return

        self._update_env_state()

        team_cap = "Blue" if self.rover_team_name.upper() == "BLUE" else "Red"
        teammate_ctf = f"{team_cap}_{1 - self.team_index}"

        obs_self = self.env.get_observation_v3(self.ctf_agent_name)
        obs_teammate = self.env.get_observation_v3(teammate_ctf)

        # MAPPO: batch_action expects [agent_0_obs, agent_1_obs] in team order.
        # team_index=0 → self is agent 0, teammate is agent 1.
        # team_index=1 → teammate is agent 0, self is agent 1.
        if self.team_index == 0:
            actions = self.policy.batch_action([obs_self, obs_teammate])
            action = actions[0]
        else:
            actions = self.policy.batch_action([obs_teammate, obs_self])
            action = actions[1]

        # Decode action → next graph node
        current_node = self.env.state[self.ctf_agent_name]
        current_node_idx = self.env.node_to_idx[current_node]
        neighbours = self.env.neighbours[current_node_idx]

        if action < len(neighbours):
            next_node_idx = neighbours[action]
        elif action == self.env.max_degree:
            # Stay — republish current node
            next_node_idx = current_node_idx
        else:
            # Tag — do nothing physically
            self.get_logger().info("[POLICY] Tag action — holding position.")
            return

        self.current_node_idx = current_node_idx
        self.goal_node_idx = next_node_idx

        # Convert next node: sim → Vicon → local map frame
        next_node = self.env.idx_to_node[next_node_idx]
        sim_xy = np.array(self.env.node_pose_dict[next_node])
        vicon_xy = self._sim_to_vicon(sim_xy)
        vicon_pt = np.array([vicon_xy[0], vicon_xy[1], self.goal_height, 1.0])
        local_pt = self.X_map_world @ vicon_pt

        goal_msg = PoseStamped()
        goal_msg.header.stamp = self.get_clock().now().to_msg()
        goal_msg.header.frame_id = self.local_frame
        goal_msg.pose.position.x = local_pt[0]
        goal_msg.pose.position.y = local_pt[1]
        goal_msg.pose.position.z = self.goal_height
        goal_msg.pose.orientation.w = 1.0
        self.publisher_local_dynus_command_goal.publish(goal_msg)

        self.get_logger().info(
            f"[POLICY] action={action} → node_idx={next_node_idx} "
            f"sim={sim_xy} vicon={vicon_xy} local=[{local_pt[0]:.3f}, {local_pt[1]:.3f}]"
        )


def main(args=None):
    rclpy.init(args=args)
    node = RoverNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    pass
