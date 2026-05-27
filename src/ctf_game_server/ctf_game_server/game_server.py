import os
import sys
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from dynus_interfaces.msg import State

from ctf_msgs.msg import JoinGameMessage, ServerToRoverMessage
#from ctf_msgs.srv import RequestGameState
from geometry_msgs.msg import TransformStamped
from scipy.spatial.transform import Rotation as R_scipy
import tf2_ros
from tf2_ros import TransformException
from visualization_msgs.msg import Marker
from nav_msgs.msg import Odometry
import networkx as nx
from geometry_msgs.msg import Point
from scipy.spatial import ConvexHull

import copy
import numpy as np
from std_msgs.msg import String

from ctf_msgs.srv import JoinGame

# GraphCTF lives in the same package directory
_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
if _PKG_DIR not in sys.path:
    sys.path.insert(0, _PKG_DIR)
from customCTF import GraphCTF

# Default ACL pkl path: acl_graph.pkl sits one directory above game_server.py
_ACL_PKL_DEFAULT = os.path.normpath(os.path.join(_PKG_DIR, '..', 'acl_graph.pkl'))

# ---------------------------------------------------------------------------
# Hard-coded 3v3 spawn positions for the ACL office map (Vicon metres).
# Blue team: top-left corner  | Red team: top-right corner
# Headings: Blue faces East (0 rad), Red faces West (π rad).
# ---------------------------------------------------------------------------
_ACL_SPAWNS_3V3 = {
    # Blue team: left-of-centre cluster, facing East toward Red side (0 rad)
    'Blue_0': (( 0.95,  3.85), 0.0),   # node 86
    'Blue_1': (( 2.75,  3.10), 0.0),   # node 71
    'Blue_2': (( 2.45,  4.75), 0.0),   # node 90
    # Red team: right-of-centre cluster, facing West toward Blue side (π rad)
    'Red_0':  (( 8.75,  3.25), np.pi), # node 51
    'Red_1':  ((10.55,  2.95), np.pi), # node 50
    'Red_2':  (( 9.50,  4.00), np.pi), # node 282
}

class GameServer(Node):
    def __init__(self, **kwargs):
        super().__init__("ctf")
        # self.grid_size = 8
        # self.ctf_player_config = kwargs.get('ctf_player_config', '2v2')

        # self.goal_height = -0.01
        
        self.declare_parameter("goal_height", -0.01)
        self.declare_parameter("total_rovers", 3)
        self.declare_parameter("seed", 0)
        self.declare_parameter("ctf_player_config", "2v2")
        self.declare_parameter("use_dlio", False)
        self.declare_parameter("tag_radius", 0.55)
        self.declare_parameter("tag_angle_tolerance", 0.785)  # ~45 degrees
        self.declare_parameter("tag_cooldown", 10.0)
        self.declare_parameter("acl_graph_pkl_path", _ACL_PKL_DEFAULT)
        self.declare_parameter("fixed_flag_hypothesis", 1)

        self.goal_height = self.get_parameter("goal_height").value
        self.total_rovers = self.get_parameter("total_rovers").value
        self._seed = self.get_parameter("seed").value
        self.ctf_player_config = self.get_parameter("ctf_player_config").value
        self.use_dlio = self.get_parameter("use_dlio").value
        _acl_pkl = self.get_parameter("acl_graph_pkl_path").value
        self._use_acl_mode = bool(_acl_pkl) and os.path.isfile(_acl_pkl)

        self.get_logger().info(f"TOTAL ROVERS {self.total_rovers}")
        self.get_logger().info(f"USE DLIO = {self.use_dlio}")

        # Subscriptions for each rover pose from VICON
        # (You can generate these dynamically)
        """
        Rover_names: Updated from rover subscriptions, subscribing and participating in the game.
        """
        self.num_rovers = 0
        # self.total_rovers = 3 # TODO make this a parma
        self.rovers_list = []
        self.rovers_info = {}
        self.rovers_state = {}
        self.rover_pose_subscriptions = {}
        self.server_to_rover_publishers = {}

        self.num_agents_blue_team = 0
        self.num_agents_red_team = 0

        self.ctf_red_agents = ['Red_0', 'Red_1', 'Red_2']
        self.ctf_blue_agents = ['Blue_0', 'Blue_1', 'Blue_2']
        self.rr_to_ctf_agent_map = {}

        # GraphCTF env — used for spawn / respawn position generation only.
        # Skipped in ACL mode; hard-coded Vicon spawn positions are used instead.
        if not self._use_acl_mode:
            self.ctf_env = GraphCTF(
                ctf_player_config=self.ctf_player_config,
                fixed_flag_hypothesis=1,
                obs_version=3,
                seed=self._seed,
                map_variant="highbay"
            )
            self.get_logger().info("GraphCTF environment initialised for spawn generation.")
        else:
            self.ctf_env = None
            self.get_logger().info(
                "ACL mode active — using hard-coded 3v3 Vicon spawn positions; "
                "GraphCTF not initialised."
            )
            # Load flag hypothesis positions from the ACL pkl.
            import pickle as _pickle
            fh_idx = self.get_parameter("fixed_flag_hypothesis").value
            try:
                with open(_acl_pkl, 'rb') as _f:
                    _acl_g = _pickle.load(_f)
                fh_nodes = _acl_g.graph.get('flag_hypothesis_nodes', [])
                if fh_nodes:
                    fh_node = fh_nodes[fh_idx % len(fh_nodes)]
                    _d = _acl_g.nodes[fh_node]
                    self._acl_flag_vicon = np.array([_d['x'], _d['y']])
                    self.get_logger().info(
                        f"ACL flag hypothesis {fh_idx} → node {fh_node} "
                        f"at vicon=({_d['x']:.2f}, {_d['y']:.2f})"
                    )
                else:
                    self._acl_flag_vicon = None
                    self.get_logger().warn(
                        "ACL graph has no 'flag_hypothesis_nodes' attribute — "
                        "flag marker will not be published."
                    )
            except Exception as e:
                self._acl_flag_vicon = None
                self.get_logger().error(f"Failed to load ACL flag hypotheses: {e}")

        self.global_frame = "world"
        self.tf_buffer = tf2_ros.Buffer()

        # initilaize join game service for rovers
        self.join_service = self.create_service(JoinGame, "/ctf/join_game", self.handle_join_request)

        """Pending: rover_name hardware to rover_name in sim mapping: for eg. rro3:red_01, rr06:blue_02"""
        self.get_logger().info("GameServer started, waiting for rovers to join...")
        
        self.game_started = False

        # debug iniital pose flag, tests code with just one rover
        self.DEBUG_INIT_POSE = False

        # Tracks the last time (seconds) each rover was tagged, for cooldown enforcement.
        self._tag_cooldowns = {}

        # Rovers that have confirmed they are at their spawn position.
        self._rovers_at_spawn = set()
        # True only after all rovers confirm spawn and START has been broadcast.
        self._game_running = False

        # Subscribe to rover-reported tag events and spawn confirmations.
        self.create_subscription(String, "/ctf/tag_event", self.handle_tag_event, 10)
        self.create_subscription(String, "/ctf/rover_ready", self.handle_rover_ready, 10)
        self.create_subscription(String, "/ctf/flag_discovered", self.handle_flag_discovered, 10)

        self._flag_discovered_text = {}  # team -> "BLUE flag discovered by RR01" etc.
        self.flag_text_pub = self.create_publisher(Marker, '/flag_discovered_text', 10)

        self.marker_pub = self.create_publisher(Marker, 'initial_pose_marker', 10)

        # publish graph every 1.0 seconds
        self.graph_timer_period = 1.0
        self.graph_pub = self.create_publisher(Marker, 'ctf_graph', 10)
        self.graph_pub_timer = self.create_timer(self.graph_timer_period, self.graph_pub_cb)

        # clock publisher
        self.text_marker_pub = self.create_publisher(Marker, f"/clock_text", 10)

        """
        # Service for rover to request game state
        self.srv = self.create_service(
            RequestGameState,
            "ctf/get_state",
            self.handle_get_state
        ) 
        """
    
    ################# BEGIN VISUALIZE GRAPH #################
    # convert networkx to nodes, edges for markers
    def nx_to_marker_data(self, G, flag):
        node_pose_dict = nx.get_node_attributes(G, 'pos')

        nodes = []
        node_index = {}

        # assign indices
        for idx, node in enumerate(G.nodes()):
            x, y = node_pose_dict[node]
            nodes.append((float(x), float(y)))
            node_index[node] = idx

        edges = []
        for u, v in G.edges():
            edges.append((node_index[u], node_index[v]))

        return nodes, edges, node_pose_dict[flag]

    def _publish_acl_flag_marker(self):
        """Publish a flag cylinder at the active ACL flag hypothesis position."""
        if self._acl_flag_vicon is None:
            return
        fx, fy = float(self._acl_flag_vicon[0]), float(self._acl_flag_vicon[1])
        height = 2.0

        flag_marker = Marker()
        flag_marker.header.frame_id = "world"
        flag_marker.header.stamp = self.get_clock().now().to_msg()
        flag_marker.ns = "ctf_flag"
        flag_marker.id = 2
        flag_marker.type = Marker.CYLINDER
        flag_marker.action = Marker.ADD
        flag_marker.scale.x = 0.2
        flag_marker.scale.y = 0.2
        flag_marker.scale.z = height
        flag_marker.color.r = 1.0
        flag_marker.color.g = 0.0
        flag_marker.color.b = 0.0
        flag_marker.color.a = 1.0
        flag_marker.pose.position.x = fx
        flag_marker.pose.position.y = fy
        flag_marker.pose.position.z = height / 2.0
        flag_marker.pose.orientation.w = 1.0
        self.graph_pub.publish(flag_marker)

    # publish graph
    def graph_pub_cb(self):
        # ACL mode: graph nodes/edges are published by the publish_real_graph node;
        # we only publish the active flag hypothesis marker here.
        if self._use_acl_mode:
            self._publish_acl_flag_marker()
            return

        # TODO much of this can be initialized once as instance variables
        if hasattr(self.ctf_env, '_bc') and self.ctf_env._bc is not None:
            bc = self.ctf_env._bc
            G_map = bc.G

            red_flag_node = bc.red_flag_L
            red_flag_2_node = bc.red_flag_R
            blue_flag_node = self.ctf_env.blue_flag_node

            flag_nodes = np.array([red_flag_node, red_flag_2_node, blue_flag_node])

            flag_idx = 1

            graph_nodes, graph_edges, flag = self.nx_to_marker_data(G_map, flag_nodes[flag_idx]) # TODO fix flag logic

            graph_frame_id = "sim" # vicon frame

            # create node markers
            node_marker = Marker()
            node_marker.header.frame_id = graph_frame_id
            node_marker.header.stamp = self.get_clock().now().to_msg()

            node_marker.ns = "ctf_graph_nodes"
            node_marker.id = 0
            node_marker.type = Marker.SPHERE_LIST
            node_marker.action = Marker.ADD

            node_marker.scale.x = 0.1  # point width
            node_marker.scale.y = 0.1
            node_marker.scale.z = 0.1

            node_marker.color.r = 1.0
            node_marker.color.g = 1.0
            node_marker.color.b = 0.0
            node_marker.color.a = 1.0

            # iterate through graph and publish nodes
            for x, y in graph_nodes:

                # point for each node
                p = Point()
                p.x = float(x)
                p.y = float(y)
                p.z = 0.0
                node_marker.points.append(p)

            # publish nodes
            self.graph_pub.publish(node_marker)

            # create edge markers
            edge_marker = Marker()
            edge_marker.header.frame_id = graph_frame_id
            edge_marker.header.stamp = self.get_clock().now().to_msg()

            edge_marker.ns = "ctf_graph_nodes"
            edge_marker.id = 1
            edge_marker.type = Marker.LINE_LIST
            edge_marker.action = Marker.ADD

            edge_marker.scale.x = 0.05

            edge_marker.color.r = 1.0
            edge_marker.color.g = 1.0
            edge_marker.color.b = 0.0
            edge_marker.color.a = 0.1

            for i, j in graph_edges:
                p1 = Point()
                p1.x = float(graph_nodes[i][0])
                p1.y = float(graph_nodes[i][1])
                p1.z = 0.0

                p2 = Point()
                p2.x = float(graph_nodes[j][0])
                p2.y = float(graph_nodes[j][1])
                p2.z = 0.0

                edge_marker.points.append(p1)
                edge_marker.points.append(p2)

            # publish edges
            self.graph_pub.publish(edge_marker)

            # create flag markers
            flag_marker = Marker()
            flag_marker.header.frame_id = graph_frame_id
            flag_marker.header.stamp = self.get_clock().now().to_msg()

            flag_marker.ns = "ctf_flag"
            flag_marker.id = 2
            flag_marker.type = Marker.CYLINDER
            flag_marker.action = Marker.ADD

            flag_marker.scale.x = 0.2
            flag_marker.scale.y = 0.2
            height = 2.0
            flag_marker.scale.z = height # height

            flag_marker.color.r = 1.0
            flag_marker.color.g = 0.0
            flag_marker.color.b = 0.0
            flag_marker.color.a = 1.0

            # flag location
            flag_marker.pose.position.x = float(flag[0])
            flag_marker.pose.position.y = float(flag[1])
            flag_marker.pose.position.z = height/2  

            flag_marker.pose.orientation.x = 0.0
            flag_marker.pose.orientation.y = 0.0
            flag_marker.pose.orientation.z = 0.0
            flag_marker.pose.orientation.w = 1.0

            # publish nodes
            self.graph_pub.publish(flag_marker)

            # shade flag region
            nodes = np.array(graph_nodes)
            flag_pos = np.array(flag)

            # find closest node to flag
            dists = np.linalg.norm(nodes - flag_pos, axis=1)
            flag_node_idx = int(np.argmin(dists))

            # build adjacency from edges
            adjacency = {i: set() for i in range(len(nodes))}
            for i, j in graph_edges:
                adjacency[i].add(j)
                adjacency[j].add(i)

            # 1-hop neighbors
            region_idxs = set([flag_node_idx]) | adjacency[flag_node_idx]
            region_pts = nodes[list(region_idxs)]

            if len(region_pts) >= 3:
                hull = ConvexHull(region_pts)
                hull_pts = region_pts[hull.vertices]

                shade_marker = Marker()
                shade_marker.header.frame_id = graph_frame_id
                shade_marker.header.stamp = self.get_clock().now().to_msg()
                shade_marker.ns = "flag_region"
                shade_marker.id = 100
                shade_marker.type = Marker.TRIANGLE_LIST
                shade_marker.action = Marker.ADD
                shade_marker.scale.x = shade_marker.scale.y = shade_marker.scale.z = 1.0
                shade_marker.color.r = 1.0
                shade_marker.color.g = 0.0
                shade_marker.color.b = 0.0
                shade_marker.color.a = 0.15  # light shading

                # triangle fan from centroid
                center = np.mean(hull_pts, axis=0)
                for i in range(len(hull_pts)):
                    p1 = hull_pts[i]
                    p2 = hull_pts[(i + 1) % len(hull_pts)]
                    for pt in [center, p1, p2]:
                        p = Point()
                        p.x = float(pt[0])
                        p.y = float(pt[1])
                        p.z = 0.0
                        shade_marker.points.append(p)

                self.graph_pub.publish(shade_marker)
    ################# END VISUALIZE GRAPH #################

    def clock_pub_cb(self):
        # clock_str = "TEST"
        time_passed = (self.get_clock().now() - self.start_time).nanoseconds / 1e9
        clock_str = f"TIMER: {time_passed:.1f}"
        self.text_publisher(clock_str)

    # publish text helper
    def text_publisher(self, text_str):

        # create flag markers
        text_marker = Marker()
        text_marker.header.frame_id = "world"
        text_marker.header.stamp = self.get_clock().now().to_msg()

        text_marker.ns = f"server_text"
        text_marker.id = 11
        text_marker.type = Marker.TEXT_VIEW_FACING
        text_marker.action = Marker.ADD

        text_marker.scale.z = 0.8

        text_marker.color.a = 1.0
        text_marker.color.r = 1.0
        text_marker.color.g = 1.0
        text_marker.color.b = 1.0

        # location
        text_marker.pose.position.x = -8.0
        text_marker.pose.position.y = 6.0
        text_marker.pose.position.z = 1.0

        text_marker.pose.orientation.x = 0.0
        text_marker.pose.orientation.y = 0.0
        text_marker.pose.orientation.z = 0.0
        text_marker.pose.orientation.w = 1.0
        text_marker.text = text_str

        # publish nodes
        self.text_marker_pub.publish(text_marker)
    
    # initialize tf from world to map for a given rover
    def initialize_tf(self, rover_name):

        # try to get the transform
        try:

            # get transform from world to init pose
            # X_world_RR
            tf = self.tf_buffer.lookup_transform(
                self.global_frame,             
                rover_name,              
                rclpy.time.Time()
            )
        except TransformException:
            self.get_logger().info("Waiting for initial TF")
            return 

        self.get_logger().info("Successfully got TF!")

        # Convert TF to 4x4 matrix
        t = tf.transform.translation
        q = tf.transform.rotation
        r = R_scipy.from_quat([q.x, q.y, q.z, q.w])

        X_world_rr = np.eye(4)
        X_world_rr[:3, :3] = r.as_matrix()
        X_world_rr[:3, 3] = np.array([t.x, t.y, t.z])

        # # publish transform for map --> init pose (world --> RR0X/map)
        # T_world_map = TransformStamped()

        # T_world_map.header.stamp = self.get_clock().now().to_msg()
        # T_world_map.header.frame_id = 'world'
        # T_world_map.child_frame_id = f"{rover_name}/map"

        # T_world_map.transform.translation.x = t.x
        # T_world_map.transform.translation.y = t.y
        # T_world_map.transform.translation.z = t.z # 0.0?

        # T_world_map.transform.rotation.x = q.x
        # T_world_map.transform.rotation.y = q.y
        # T_world_map.transform.rotation.z = q.z
        # T_world_map.transform.rotation.w = q.w

        # self.world_map_broadcaster.sendTransform(T_world_map)
        # TODO since ^ is published in rover node, can i just grab it?

        # X world wrt map = inverse(X RR wrt world)
        # X_map_world = np.linalg.inv(X_world_rr)

        # p map wrt world
        X_world_map = X_world_rr
        return X_world_map

    # handle rover joins
    def handle_join_request(self, request, response):

        # get rover name and team
        rover_name = request.rover_name
        rover_team_name = request.rover_team_name
        assert rover_team_name.startswith('R') or rover_team_name.startswith('B')

        # return if rover already joined
        if rover_name in self.rovers_list:
            response.accepted = False
            response.message = f"ROVER {rover_name} already joined."
            return response

        # return if game full
        if self.num_rovers >= self.total_rovers:
            response.accepted = False
            response.message = "Game full."
            return response
        
        # init rover pose topic
        if self.use_dlio:
            rover_pose_topic = "/{}/dlio/odom_node/odom".format(rover_name)
        else:
            rover_pose_topic = "/{}/world".format(rover_name)

        server_to_rover_topic = "/{}/server_to_rover".format(rover_name)

        # set teams
        self.rr_to_ctf_agent_map[rover_name] = self.ctf_red_agents[self.num_agents_red_team] if rover_team_name.startswith('R') else self.ctf_blue_agents[self.num_agents_blue_team]
        self.num_rovers += 1
        if rover_team_name.startswith('R'): self.num_agents_red_team += 1
        elif rover_team_name.startswith('B'): self.num_agents_blue_team += 1

        # # get transform world to map
        # if self.use_dlio:
        #     X_world_map = self.initialize_tf(rover_name)

        # add rover info to the game
        self.rovers_list.append(rover_name)
        self.rovers_info[rover_name] = {
            'team': rover_team_name,
            'pose_topic': rover_pose_topic,
            'server_to_rover_topic': server_to_rover_topic,
            'ctf_agent_name': self.rr_to_ctf_agent_map[rover_name],
        }
        # if self.use_dlio:
        #     self.rovers_info[rover_name]['X_world_map'] = X_world_map
        self.rovers_state[rover_name] = {
            'pose': [],
            'last_seen': []
        }

        self.get_logger().info("[GAMESERVER]: ROVER {} JOINED FROM TEAM {} ".format(rover_name, rover_team_name))

        # create pose subscriber for rover
        if self.use_dlio:
            rover_pose_sub = self.create_subscription(
                    Odometry,
                    rover_pose_topic,
                    lambda msg, name=rover_name: self.dlio_callback(msg, name),
                    10,
                )
            self.rover_pose_subscriptions[rover_name] = rover_pose_sub
        else:
            rover_pose_sub = self.create_subscription(
                PoseStamped,
                rover_pose_topic,
                lambda msg, name=rover_name: self.vicon_callback(msg, name),
                10,
            )
            self.rover_pose_subscriptions[rover_name] = rover_pose_sub

        # create publisher for rover
        server_to_rover_pub = self.create_publisher(ServerToRoverMessage, server_to_rover_topic, 10)
        self.server_to_rover_publishers[rover_name] = server_to_rover_pub

        if self.DEBUG_INIT_POSE and self.num_rovers == 1:
            self.get_logger().info("[DEBUG INIT POSE]: Starting game with one rover.")
            self.pre_start_game_utils()
            self.start_game_callback()

        # start the game if enough rovers
        if self.num_rovers == self.total_rovers:
            self.pre_start_game_utils()
            self.start_game_callback()

        response.accepted = True
        response.message = f"{rover_name} join successful."

        return response
    
    def pre_start_game_utils(self):
        inv_map = {}
        for rr_name, ctf_agent in self.rr_to_ctf_agent_map.items():
            inv_map[ctf_agent] = rr_name
        self.ctf_agent_to_rr_map = copy.deepcopy(inv_map)
        return
    
    def publish_marker(self, pose, rover_name):
        """
        Publish a sphere marker at the given pose.
        pose: np.array or list [x, y, z, yaw] or [x, y, z]
        rover_name: str, used for marker namespace
        """
        marker = Marker()
        marker.header.frame_id = "world"
        marker.header.stamp = self.get_clock().now().to_msg()
        
        marker.ns = rover_name
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        
        # Set position
        marker.pose.position.x = pose[0]
        marker.pose.position.y = pose[1]
        marker.pose.position.z = 0.5 #pose[2] if len(pose) > 2 else 0.0
        
        # No orientation for a simple sphere
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0
        
        # Scale of the sphere
        marker.scale.x = 0.3  # meters
        marker.scale.y = 0.3
        marker.scale.z = 0.3
        
        # Color (just pick a color, e.g., red)
        
        rover_team = self.rovers_info[rover_name]["team"]

        marker.color.r = 1.0 if rover_team == "RED" else 0.0
        marker.color.g = 0.0
        marker.color.b = 1.0 if rover_team == "BLUE" else 0.0
        marker.color.a = 1.0  # alpha
        
        # Lifetime 0 = forever
        marker.lifetime.sec = 0
        marker.lifetime.nanosec = 0
        
        # Publish
        self.marker_pub.publish(marker)


    def start_game_callback(self):
        # Send initial commanded poses to start the game.
        self.get_logger().info("[GAMESERVER] All rovers joined. Sending initial poses...")

        # Build a list of (ctf_agent, vicon_pos_3d, vicon_heading_rad) to send.
        if self._use_acl_mode:
            # ACL map: hard-coded Vicon spawn positions, no sim→Vicon conversion.
            spawn_list = []
            for ctf_agent, (xy, hdg) in _ACL_SPAWNS_3V3.items():
                if ctf_agent not in self.ctf_agent_to_rr_map:
                    continue  # agent not in this game session
                p_vicon_pos = np.array([xy[0], xy[1], 0.0])
                spawn_list.append((ctf_agent, p_vicon_pos, hdg))
        else:
            state = self.compute_initial_poses()
            spawn_list = []
            for ctf_agent, pose in state.items():
                sim_x, sim_y = pose
                # Blue faces +y in sim (heading=2=North), Red faces -y (heading=6=South)
                sim_heading = 2 if ctf_agent.startswith('Blue') else 6
                p_vicon_pos, p_vicon_heading = self.sim_frame_to_vicon_frame(
                    sim_x, sim_y, sim_heading
                )
                spawn_list.append((ctf_agent, p_vicon_pos, p_vicon_heading))

        for ctf_agent, p_vicon_pos, p_vicon_heading in spawn_list:
            self.get_logger().info(f"AGENT = {ctf_agent}")
            rr_name = self.ctf_agent_to_rr_map[ctf_agent]

            vel_vicon_xy = np.array([np.cos(p_vicon_heading), np.sin(p_vicon_heading)])
            eps_vel = 0.001

            goal_msg = State()
            goal_msg.pos.x = p_vicon_pos[0]
            goal_msg.pos.y = p_vicon_pos[1]
            goal_msg.pos.z = self.goal_height
            goal_msg.vel.x = eps_vel * vel_vicon_xy[0]
            goal_msg.vel.y = eps_vel * vel_vicon_xy[1]
            goal_msg.vel.z = 0.
            goal_msg.quat.x = 0.0
            goal_msg.quat.y = 0.0
            goal_msg.quat.z = 0.0
            goal_msg.quat.w = 1.0

            msg = ServerToRoverMessage()
            msg.command = 'INIT'
            msg.roster_rover_names = list(self.rr_to_ctf_agent_map.keys())
            msg.roster_ctf_agent_names = list(self.rr_to_ctf_agent_map.values())
            msg.commanded_goal = goal_msg

            self.server_to_rover_publishers[rr_name].publish(msg)
            self.publish_marker(p_vicon_pos, rr_name)
            self.get_logger().info(
                f"[GAMESERVER] Sent initial pose (vicon frame) {p_vicon_pos} to {rr_name}"
            )

        self.game_started = True
        self.get_logger().info("[GAMESERVER] Game started!")

        # start a timer to count up the time
        self.clock_text_timer_period = 0.1
        self.start_time = self.get_clock().now()
        self.clock_pub = self.create_timer(self.clock_text_timer_period, self.clock_pub_cb)
        return
    
    def compute_initial_poses(self):
        """Sample spawn positions from GraphCTF, returning {ctf_agent: [sim_x, sim_y]}."""
        self.ctf_env.reset()
        state = {}
        for ctf_agent in self.rr_to_ctf_agent_map.values():
            node = self.ctf_env.state[ctf_agent]
            sim_xy = np.array(self.ctf_env.node_pose_dict[node], dtype=float)
            state[ctf_agent] = sim_xy
        return state
    
    def publish_sim_to_vicon_tf(self, t, R):

        rot = R_scipy.from_matrix(R)
        q = rot.as_quat()

        self.tf_broadcaster = tf2_ros.StaticTransformBroadcaster(self)

        t_msg = TransformStamped()
        t_msg.header.stamp = self.get_clock().now().to_msg()
        t_msg.header.frame_id = "world"
        t_msg.child_frame_id = "sim"

        t_msg.transform.translation.x = t[0]
        t_msg.transform.translation.y = t[1]
        t_msg.transform.translation.z = t[2]

        t_msg.transform.rotation.x = q[0]
        t_msg.transform.rotation.y = q[1]
        t_msg.transform.rotation.z = q[2]
        t_msg.transform.rotation.w = q[3]

        self.tf_broadcaster.sendTransform(t_msg)

    # @staticmethod
    def sim_frame_to_vicon_frame(self, sim_x, sim_y, sim_heading=0):
        # Graph-consistent transform — matches rover_node._R_SIM_TO_VICON / _T_VICON:
        #   Axes: x_sim = -y_vicon, y_sim = +x_vicon, scale = 1 m/unit
        #   R_sim_to_vicon = [[0,1],[-1,0]] (90° CW, det=+1),  T = [5, 5] m
        #   vicon_xy = R2 @ sim_xy + T
        # Heading: θ_vicon = θ_sim - π/2  (derived from R2 applied to unit direction vector)
        R2 = np.array([[0., 1.], [-1., 0.]])
        T2 = np.array([5.0, 5.0])

        sim_xy = np.array([float(sim_x), float(sim_y)])
        vicon_xy = R2 @ sim_xy + T2
        p_vicon_pos = np.array([vicon_xy[0], vicon_xy[1], 0.0])

        p_sim_yaw = (np.pi / 4.0) * sim_heading
        p_vicon_heading = p_sim_yaw - np.pi / 2.0

        # Publish debug TF with updated transform
        R3 = np.array([[R2[0][0], R2[0][1], 0.], [R2[1][0], R2[1][1], 0.], [0., 0., 1.]])
        t3 = np.array([T2[0], T2[1], 0.0])
        self.publish_sim_to_vicon_tf(t3, R3)

        if self.DEBUG_INIT_POSE:
            self.get_logger().info(f"[DEBUG INIT POSE]: Converted pose (vicon frame) = {p_vicon_pos}")

        return (p_vicon_pos, p_vicon_heading)

    @staticmethod
    def yaw_to_quaternion(yaw):
        """
        Convert a yaw (rotation about Z) to a geometry_msgs.Quaternion
        """
        from scipy.spatial.transform import Rotation as R_scipy
        from geometry_msgs.msg import Quaternion
        rot = R_scipy.from_euler('z', yaw)  # 'z' = yaw rotation
        q = rot.as_quat()  # returns [x, y, z, w]
        return Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])

    def vicon_callback(self, msg, name):
        """Update rover poses from mocap system."""
        pose = msg.pose
        self.rovers_state[name]['pose'].append(pose)
        now = self.get_clock().now().to_msg()  # builtin_interfaces/Time
        self.rovers_state[name]['last_seen'].append(now)
        # Optional: log at low verbosity
        self.get_logger().debug(f"[GAMESERVER] Updated pose for {name} at sec={now.sec} nsec={now.nanosec}")
        self.get_logger().debug(f"[GAMESERVER] Latest pose for {name} at sec={now.sec} nsec={now.nanosec}: {pose}")
        return
    
    # ------------------------------------------------------------------
    # Synchronised game start
    # ------------------------------------------------------------------

    def handle_rover_ready(self, msg: String):
        """Collect spawn confirmations; broadcast START once all rovers have confirmed."""
        if not self.game_started:
            return
        rover_name = msg.data
        if rover_name not in self.rovers_list:
            self.get_logger().warn(f"[GAMESERVER] rover_ready from unknown rover: {rover_name}")
            return
        if rover_name in self._rovers_at_spawn:
            return  # already counted
        self._rovers_at_spawn.add(rover_name)
        self.get_logger().info(
            f"[GAMESERVER] {rover_name} at spawn "
            f"({len(self._rovers_at_spawn)}/{self.total_rovers})"
        )
        if len(self._rovers_at_spawn) == self.total_rovers:
            self._broadcast_start()

    def _broadcast_start(self):
        """Send START to every rover — signals that GNN inference may begin."""
        self._game_running = True
        self.get_logger().info("[GAMESERVER] All rovers at spawn — broadcasting START.")
        for rr_name in self.rovers_list:
            msg = ServerToRoverMessage()
            msg.command = 'START'
            self.server_to_rover_publishers[rr_name].publish(msg)
            self.get_logger().info(f"[GAMESERVER] START → {rr_name}")

    # ------------------------------------------------------------------
    # Flag discovery
    # ------------------------------------------------------------------

    def handle_flag_discovered(self, msg: String):
        try:
            team, rover_name = msg.data.split(':')
        except ValueError:
            return
        team = team.upper()
        if team in self._flag_discovered_text:
            return  # already shown

        self._flag_discovered_text[team] = rover_name
        self.get_logger().info(f"[FLAG] {team} flag discovered by {rover_name}")

        # One marker per team, stacked vertically so they don't overlap
        y_offset = 0.0 if team == "BLUE" else -1.2
        color = (0.3, 0.6, 1.0) if team == "BLUE" else (1.0, 0.3, 0.3)

        m = Marker()
        m.header.frame_id = "world"
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = "flag_discovered"
        m.id = 20 if team == "BLUE" else 21
        m.type = Marker.TEXT_VIEW_FACING
        m.action = Marker.ADD
        m.scale.z = 0.6
        m.color.r, m.color.g, m.color.b, m.color.a = color[0], color[1], color[2], 1.0
        m.pose.position.x = -8.0
        m.pose.position.y = 4.5 + y_offset
        m.pose.position.z = 1.0
        m.pose.orientation.w = 1.0
        m.text = f"{team} flag discovered! ({rover_name})"
        self.flag_text_pub.publish(m)

    # ------------------------------------------------------------------
    # Physical tag handling
    # ------------------------------------------------------------------

    def handle_tag_event(self, msg: String):
        """Validate a rover-reported tag and dispatch RESPAWN if legitimate."""
        if not self._game_running:
            return

        try:
            tagger_rr, tagged_rr = msg.data.split(':')
        except ValueError:
            self.get_logger().warn(f"[TAG] Malformed tag event: '{msg.data}'")
            return

        if tagger_rr not in self.rovers_info or tagged_rr not in self.rovers_info:
            self.get_logger().warn(f"[TAG] Unknown rover in event: {tagger_rr} → {tagged_rr}")
            return

        # Cooldown check
        now_sec = self.get_clock().now().nanoseconds / 1e9
        cooldown = self.get_parameter("tag_cooldown").value
        last_tagged = self._tag_cooldowns.get(tagged_rr, 0.0)
        if now_sec - last_tagged < cooldown:
            self.get_logger().info(
                f"[TAG] {tagger_rr} → {tagged_rr} ignored: cooldown "
                f"({now_sec - last_tagged:.1f}s < {cooldown}s)"
            )
            return

        # Need poses for both rovers
        tagger_poses = self.rovers_state[tagger_rr]['pose']
        tagged_poses = self.rovers_state[tagged_rr]['pose']
        if not tagger_poses or not tagged_poses:
            self.get_logger().warn("[TAG] Missing Vicon poses for validation.")
            return

        tagger_p = tagger_poses[-1].position
        tagged_p = tagged_poses[-1].position

        # Distance check
        dist = np.linalg.norm([tagger_p.x - tagged_p.x, tagger_p.y - tagged_p.y])
        tag_radius = self.get_parameter("tag_radius").value
        if dist > tag_radius:
            self.get_logger().info(
                f"[TAG] {tagger_rr} → {tagged_rr} invalid: dist={dist:.3f}m > radius={tag_radius}m"
            )
            return

        # Heading check: tagger must be facing tagged rover
        tagger_q = tagger_poses[-1].orientation
        tagger_yaw = R_scipy.from_quat(
            [tagger_q.x, tagger_q.y, tagger_q.z, tagger_q.w]
        ).as_euler('xyz')[2]
        bearing = np.arctan2(tagged_p.y - tagger_p.y, tagged_p.x - tagger_p.x)
        angle_diff = abs(((tagger_yaw - bearing) + np.pi) % (2 * np.pi) - np.pi)
        tag_angle = self.get_parameter("tag_angle_tolerance").value
        if angle_diff > tag_angle:
            self.get_logger().info(
                f"[TAG] {tagger_rr} → {tagged_rr} invalid: heading off by "
                f"{np.degrees(angle_diff):.1f}° (tolerance={np.degrees(tag_angle):.1f}°)"
            )
            return

        self.get_logger().info(
            f"[TAG] Valid: {tagger_rr} tagged {tagged_rr} "
            f"(dist={dist:.3f}m, heading_diff={np.degrees(angle_diff):.1f}°)"
        )
        self._tag_cooldowns[tagged_rr] = now_sec
        self._send_respawn(tagged_rr)

    def _sample_respawn_sim(self, team: str):
        """Sample a random (x, y) in sim coordinates from the team's respawn zone."""
        if team.upper().startswith('R'):
            area = self.ctf_env.red_tag_spawn_area
        else:
            area = self.ctf_env.blue_tag_spawn_area
        x = np.random.uniform(area['x_lim'][0], area['x_lim'][1])
        y = np.random.uniform(area['y_lim'][0], area['y_lim'][1])
        return float(x), float(y)

    def _send_respawn(self, tagged_rr_name: str):
        """Send a RESPAWN command to the tagged rover with a sampled respawn position."""
        team = self.rovers_info[tagged_rr_name]['team']  # 'RED' or 'BLUE'

        if self._use_acl_mode:
            # ACL map: randomly pick one of the team's 3 hard-coded spawn positions.
            if team.upper().startswith('R'):
                candidates = ['Red_0', 'Red_1', 'Red_2']
            else:
                candidates = ['Blue_0', 'Blue_1', 'Blue_2']
            chosen = candidates[np.random.randint(len(candidates))]
            xy, p_vicon_heading = _ACL_SPAWNS_3V3[chosen]
            p_vicon_pos = np.array([xy[0], xy[1], 0.0])
        else:
            sim_x, sim_y = self._sample_respawn_sim(team)
            sim_heading = 6 if team.upper().startswith('R') else 2
            p_vicon_pos, p_vicon_heading = self.sim_frame_to_vicon_frame(
                sim_x, sim_y, sim_heading
            )

        vel_vicon_xy = np.array([np.cos(p_vicon_heading), np.sin(p_vicon_heading)])
        eps_vel = 0.001

        goal_msg = State()
        goal_msg.pos.x = p_vicon_pos[0]
        goal_msg.pos.y = p_vicon_pos[1]
        goal_msg.pos.z = self.goal_height
        goal_msg.vel.x = eps_vel * vel_vicon_xy[0]
        goal_msg.vel.y = eps_vel * vel_vicon_xy[1]
        goal_msg.vel.z = 0.0
        goal_msg.quat.x = 0.0
        goal_msg.quat.y = 0.0
        goal_msg.quat.z = 0.0
        goal_msg.quat.w = 1.0

        msg = ServerToRoverMessage()
        msg.command = 'RESPAWN'
        msg.commanded_goal = goal_msg

        self.server_to_rover_publishers[tagged_rr_name].publish(msg)
        self.get_logger().info(
            f"[GAMESERVER] RESPAWN sent to {tagged_rr_name} (team={team}) "
            f"→ vicon=[{p_vicon_pos[0]:.3f}, {p_vicon_pos[1]:.3f}]"
        )

    # use dlio for poses instead of vicon
    # convert dlio pose to the vicon (global) frame and save it for each rover
    def dlio_callback(self, msg, name):

        # X_world_map = self.rovers_info[name]["X_world_map"]

        p = msg.pose.pose.position
        q = msg.pose.pose.orientation

        # local_point = np.array([p.x, p.y, p.z, 1.0])
        # q_local = np.array([q.x, q.y, q.z, q.w])

        # global_point = self.X_world_map @ local_point #local pt is dlio

        # R_world_map = X_world_map[:3, :3]
        # q_world_map = R_scipy.from_matrix(R_world_map).as_quat()  

        # # get rotations
        # r_world_map = R_scipy.from_quat(q_world_map)
        # r_local = R_scipy.from_quat(q_local)

        # r_global = r_world_map * r_local
        # global_orient = r_global.as_quat()

        # rover_pose = PoseStamped()

        # rover_pose.pose.position.x = global_point[0]
        # rover_pose.pose.position.y = global_point[1]
        # rover_pose.pose.position.z = global_point[2]

        # rover_pose.pose.orientation.x = global_orient[0]
        # rover_pose.pose.orientation.y = global_orient[1]
        # rover_pose.pose.orientation.z = global_orient[2]
        # rover_pose.pose.orientation.w = global_orient[3]

        rover_pose = PoseStamped()

        rover_pose.pose.position.x = p.x #global_point[0]
        rover_pose.pose.position.y = p.y #global_point[1]
        rover_pose.pose.position.z = p.z #global_point[2]

        rover_pose.pose.orientation.x = q.x #global_orient[0]
        rover_pose.pose.orientation.y = q.y #global_orient[1]
        rover_pose.pose.orientation.z = q.z #global_orient[2]
        rover_pose.pose.orientation.w = q.w #global_orient[3]

        
    
def main(args=None):
    rclpy.init(args=args)
    node = GameServer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    pass