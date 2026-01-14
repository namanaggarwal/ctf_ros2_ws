import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from dynus_interfaces.msg import State

from ctf_msgs.msg import JoinGameMessage, ServerToRoverMessage
#from ctf_msgs.srv import RequestGameState
from geometry_msgs.msg import TransformStamped
from scipy.spatial.transform import Rotation as R_scipy
import tf2_ros
from visualization_msgs.msg import Marker

import copy
import functools
import random
import copy
import numpy as np
# import torch
from gymnasium.utils import seeding

# import tf.transformations as tft

def make_seeded_rngs(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    #torch.manual_seed(seed)
    
    """
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    """
    np_random, actual_seed = seeding.np_random(seed)
    torch_rng = None #torch.Generator().manual_seed(actual_seed)
    return {
        "np_random": np_random,
        "torch_rng": None,
        "actual_seed": actual_seed
    }

class GameServer(Node):
    def __init__(self, **kwargs):
        super().__init__("ctf")
        self.grid_size = 10
        self.ctf_player_config = kwargs.get('ctf_player_config', '2v2')
        self.ctf_env = None # !? --> move CTF custom environment to within the same directory and import CustomCTF_v1.
        self.blue_init_spawn_y_lim = 2 # Measured from bottom of the grid.
        self.possible_init_headings_blue_team = [0, 1, 2, 3] # possible init headings = (0, 45, 90, 135)
        self.red_init_spawn_y_lim = 2 # Measured from top of the grid.
        self.possible_init_headings_red_team = [0, 7, 6, 5] # possible init headings = (0, -45, -90, -135)

        # Subscriptions for each rover pose from VICON
        # (You can generate these dynamically)
        """
        Rover_names: Updated from rover subscriptions, subscribing and participating in the game.
        """
        self.num_rovers = 0
        self.total_rovers = 4
        self.rovers_list = []
        self.rovers_info = {}
        self.rovers_state = {}
        self.rover_pose_subscriptions = {}
        self.server_to_rover_publishers = {}

        self.blue_agents = []
        self.red_agents = []
        self.num_agents_blue_team = 0
        self.num_agents_red_team = 0
        
        self._seed = kwargs.get('seed', 3758)
        self.seed(seed=self._seed)

        self.ctf_red_agents = ['Red_0', 'Red_1']
        self.ctf_blue_agents = ['Blue_0', 'Blue_1']
        self.agents = copy.deepcopy(self.ctf_blue_agents) + copy.deepcopy(self.ctf_red_agents)
        self.rr_to_ctf_agent_map = {}
        self.state = {}

        self.join_game_topic = "/ctf/join"
        self.subscriber_join_game_topic = self.create_subscription(
                JoinGameMessage,
                self.join_game_topic,
                self.join_game_callback,
                10,
            )
        """Pending: rover_name hardware to rover_name in sim mapping: for eg. rro3:red_01, rr06:blue_02"""
        self.get_logger().info("GameServer started, waiting for rovers to join...")
        
        for rover in self.rovers_list:
            server_to_rover_channel = self.rovers_info[rover]['server_to_rover_channel']
            self.publisher_serve_to_rover = self.create_publisher(
                    ServerToRoverMessage,
                    server_to_rover_channel,
                )
        
        self.game_started = False

        # debug iniital pose flag, tests code with just one rover
        self.DEBUG_INIT_POSE = False

        self.marker_pub = self.create_publisher(Marker, 'initial_pose_marker', 10)

        """
        # Service for rover to request game state
        self.srv = self.create_service(
            RequestGameState,
            "ctf/get_state",
            self.handle_get_state
        ) 
        """

    def join_game_callback(self, msg: JoinGameMessage):
        if self.num_rovers == self.total_rovers:
            return
        
        rover_name = msg.rover_name
        rover_team_name = msg.rover_team_name
        assert rover_team_name.startswith('R') or rover_team_name.startswith('B')

        rover_pose_topic = "/{}/world".format(rover_name)
        server_to_rover_topic = "/{}/server_to_rover".format(rover_name)
        if rover_name in self.rovers_list:
            self.get_logger().warn(f"[GAMESERVER]: ROVER {rover_name} already joined.")
            return
        
        self.rr_to_ctf_agent_map[rover_name] = self.ctf_red_agents[self.num_agents_red_team] if rover_team_name.startswith('R') else self.ctf_blue_agents[self.num_agents_blue_team]
        self.num_rovers += 1
        if rover_team_name.startswith('R'): self.num_agents_red_team += 1
        elif rover_team_name.startswith('B'): self.num_agents_blue_team += 1

        self.rovers_list.append(rover_name)
        self.rovers_info[rover_name] = {
            'team': rover_team_name,
            'pose_topic': rover_pose_topic,
            'server_to_rover_topic': server_to_rover_topic,
            'ctf_agent_name': self.rr_to_ctf_agent_map[rover_name]
        }
        self.rovers_state[rover_name] = {
            'pose': [],
            'last_seen': []
        }
        self.get_logger().info("[GAMESERVER]: ROVER {} JOINED FROM TEAM {} ".format(rover_name, rover_team_name))

        rover_pose_sub = self.create_subscription(
                PoseStamped,
                rover_pose_topic,
                lambda msg, name=rover_name: self.vicon_callback(msg, name),
                10,
            )
        self.rover_pose_subscriptions[rover_name] = rover_pose_sub

        server_to_rover_pub = self.create_publisher(ServerToRoverMessage, server_to_rover_topic, 10)
        self.server_to_rover_publishers[rover_name] = server_to_rover_pub

        if self.DEBUG_INIT_POSE and self.num_rovers == 1:
            self.get_logger().info("[DEBUG INIT POSE]: Starting game with one rover.")
            self.pre_start_game_utils()
            self.start_game_callback()

        if self.num_rovers == self.total_rovers:
            self.pre_start_game_utils()
            self.start_game_callback()
        return

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
        marker.pose.position.z = pose[2] if len(pose) > 2 else 0.0
        
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
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 1.0  # alpha
        
        # Lifetime 0 = forever
        marker.lifetime.sec = 0
        marker.lifetime.nanosec = 0
        
        # Publish
        self.marker_pub.publish(marker)


    def start_game_callback(self):
        # Send initial commanded poses to start the game.
        # Wait for response.
        self.get_logger().info("[GAMESERVER] All rovers joined. Sending initial poses...")
        state = self.compute_initial_poses()  # method to define starting positions

        for ctf_agent, pose in state.items():
            self.get_logger().info(f"AGENT = {ctf_agent}")
            rr_name = self.ctf_agent_to_rr_map[ctf_agent]
            msg = ServerToRoverMessage()
            msg.command = 'INIT'

            if self.DEBUG_INIT_POSE:
                self.get_logger().info(f"[DEBUG INIT POSE] Received init pose for {ctf_agent}: {pose}")

            discrete_x, discrete_y, discrete_heading = pose
            p_vicon_pos, p_vicon_heading = self.sim_frame_to_vicon_frame(discrete_x, discrete_y, discrete_heading)
            q = GameServer.yaw_to_quaternion(p_vicon_heading)

            vel_vicon_xy = np.array([np.cos(p_vicon_heading), np.sin(p_vicon_heading)])
            
            goal_msg = State()
            # set position
            goal_msg.pos.x = p_vicon_pos[0]
            goal_msg.pos.y = p_vicon_pos[1]
            goal_msg.pos.z = p_vicon_pos[2]

            eps_vel = 0.001
            # set velocity
            goal_msg.vel.x = eps_vel*vel_vicon_xy[0]
            goal_msg.vel.y = eps_vel*vel_vicon_xy[1]
            goal_msg.vel.z = 0.

            # identity
            goal_msg.quat.x = 0.0
            goal_msg.quat.y = 0.0
            goal_msg.quat.z = 0.0
            goal_msg.quat.w = 1.0

            msg.commanded_goal = goal_msg
            """
            pose_msg = PoseStamped()
            pose_msg.header.stamp = self.get_clock().now().to_msg()
            pose_msg.header.frame_id = "world"

            pose_msg.pose.position.x = p_vicon_pos[0]
            pose_msg.pose.position.y = p_vicon_pos[1]
            pose_msg.pose.position.z = p_vicon_pos[2] # equal to 0.

            pose_msg.pose.orientation.x = q[0]
            pose_msg.pose.orientation.y = q[1]
            pose_msg.pose.orientation.z = q[2]
            pose_msg.pose.orientation.w = q[3]
            
            msg.commanded_pose = pose_msg
            """
            self.server_to_rover_publishers[rr_name].publish(msg)
            self.publish_marker(p_vicon_pos, rr_name)
            self.get_logger().info(f"[GAMESERVER] Sent initial pose (vicon frame) {p_vicon_pos} to {rr_name}")


        # Wait for confirmation from rovers or add a short delay
        # Then mark game as started
        self.game_started = True
        self.get_logger().info("[GAMESERVER] Game started!")
        return
    
    def compute_initial_poses(self):
        self.reset()
        """
        RR03_init_pose = (+2.8, -1.4, 0.)
        RR06_init_pose = (+0., +1.4, 0.)
        pose_dict = {'RR03': RR03_init_pose, 'RR06': RR06_init_pose}
        """
        state = copy.deepcopy(self.state)
        return state
    
    def _sample_init_heading(self, agent_team, x, y):
        assert agent_team == "Blue" or agent_team == "Red"
        if agent_team == "Blue":
            possible_init_headings = copy.deepcopy(self.possible_init_headings_blue_team)
            if x == 0: possible_init_headings.remove(3)
            if x == self.grid_size - 1:
                possible_init_headings.remove(0)
                possible_init_headings.remove(1)
            heading = self.np_random.choice(possible_init_headings) #np.random.choice(possible_init_headings)
        elif agent_team == "Red":
            possible_init_headings = copy.deepcopy(self.possible_init_headings_red_team)
            if x == 0: possible_init_headings.remove(5)
            if x == 1:
                possible_init_headings.remove(0)
                possible_init_headings.remove(7)
            heading = self.np_random.choice(possible_init_headings)
        return heading

    @staticmethod
    def _heading_to_direction_vector(heading):
        vec = None
        if heading == 0:
            vec = np.array([ +1, 0 ])
        elif heading == 1:
            vec = np.array([ +1, +1 ])
        elif heading == 2:
            vec = np.array([ 0, +1 ])
        elif heading == 3:
            vec = np.array([ -1, +1 ])
        elif heading == 4:
            vec = np.array([ -1, 0 ])
        elif heading == 5:
            vec = np.array([ -1, -1 ])
        elif heading == 6:
            vec = np.array([ 0, -1 ])
        elif heading == 7:
            vec = np.array([ +1, -1 ])
        return vec

    def reset(self):

        blue_team_init_xys = []
        for agent_iter, blue_agent_num in enumerate(range(self.num_agents_blue_team)):
            if agent_iter == 0:
                rand_x = self.np_random.integers(0, self.grid_size)
                rand_y = self.np_random.integers(0, self.blue_init_spawn_y_lim + 1)
                rand_theta = self._sample_init_heading(agent_team="Blue", x=rand_x, y=rand_y)
                self.state["Blue_{}".format(blue_agent_num)] = np.array([rand_x, rand_y, rand_theta])
                blue_team_init_xys.append((rand_x, rand_y))
                continue
            # A while loop to break ties in the x-y location to prevent collision during agent initialization.
            while (rand_x, rand_y) in blue_team_init_xys:
                rand_x = self.np_random.integers(0, self.grid_size)
                rand_y = self.np_random.integers(0, self.blue_init_spawn_y_lim + 1)
            blue_team_init_xys.append((rand_x, rand_y))
            rand_theta = self._sample_init_heading(agent_team="Blue", x=rand_x, y=rand_y)
            self.state["Blue_{}".format(blue_agent_num)] = np.array([rand_x, rand_y, rand_theta])

        red_team_init_xys = []
        for agent_iter, red_agent_num in enumerate(range(self.num_agents_red_team)):

            # set agent initial positions to a random x, y, theta position
            if agent_iter == 0:
                rand_x = self.np_random.integers(0, self.grid_size)
                rand_y = self.np_random.integers(self.grid_size - self.red_init_spawn_y_lim - 1, self.grid_size)
                rand_theta = self._sample_init_heading(agent_team="Red", x=rand_x, y=rand_y)

                if self.DEBUG_INIT_POSE:
                    rand_x, rand_y, rand_theta = -3., -1., 0.
                    self.get_logger().info(f"[DEBUG INIT POSE]: Red agent init pose (game frame) set to ({rand_x}, {rand_y}, {rand_theta}).")

                self.state["Red_{}".format(red_agent_num)] = np.array([rand_x, rand_y, rand_theta])
                red_team_init_xys.append((rand_x, rand_y))
                continue
            # A while loop to break ties in the x-y location to prevent collision during agent initialization.
            while (rand_x, rand_y) in red_team_init_xys:
                rand_x = self.np_random.integers(0, self.grid_size)
                rand_y = self.np_random.integers(self.grid_size - self.red_init_spawn_y_lim - 1, self.grid_size)
            red_team_init_xys.append((rand_x, rand_y))
            rand_theta = self._sample_init_heading(agent_team="Red", x=rand_x, y=rand_y)
            self.state["Red_{}".format(red_agent_num)] = np.array([rand_x, rand_y, rand_theta])

        #obs = self._observations() # !!!!!! --> method self._observations() not copied from CustomCTF_v0 to GameServer class.
        obs = {agent: {} for agent in self.agents}
        info = {agent: {} for agent in self.agents}
        return obs, info

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
    def sim_frame_to_vicon_frame(self, discrete_x, discrete_y, discrete_heading):
        a = 0.762 # meters
        delta_x = a / 2. # meters
        delta_y = a / 2. # meters

        import numpy as np
        yaw = -np.pi/2
        R = np.array([
            [ np.cos(yaw), -np.sin(yaw), 0],
            [ np.sin(yaw),  np.cos(yaw), 0],
            [          0,           0, 1]
        ])
        tx, ty, tz = 4*a + delta_y, -(4*a + delta_x), 0. # t_from_sim_to_vicon is position of sim origin in Vicon frame:
        t = np.array([tx, ty, tz])

        self.publish_sim_to_vicon_tf(t, R)

        p_sim_pos = np.array([a*discrete_x, a*discrete_y, 0.])

        p_sim_yaw = (+np.pi/4.) * discrete_heading

        p_pos_sim_to_vicon = R@p_sim_pos + t
        p_yaw_sim_to_vicon = p_sim_yaw - yaw

        p_vicon_pos = p_pos_sim_to_vicon
        p_vicon_heading = p_yaw_sim_to_vicon

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

    def discrete_grid_abstraction_to_highbay_coordinates(self, discrete_x, discrete_y, heading):

        x, y = a*discrete_x, a*discrete_y # game ctf discrete frame

        x_vicon, y_vicon = 4*a + delta_x, 4*a + delta_y # vicon coords in game ctf discrete frame

        # game ctf -> vicon frame: 90 degrees counter-clockwise
        # !? : Convert above to quaternion rotation and build tf from this rotation and x_vicon, y_vicon.
        # !?2: Then convert x, y from game_ctf_frame to vicon_frame.

        heading_in_radians = heading * np.pi / 4. # game_ctf_frame
        # !? convert above to vicon frame.self._seed

        q = tft.quaternion_from_euler(0.0, 0.0, -1.57079632679)

        from geometry_msgs.msg import TransformStamped

        t = TransformStamped()
        t.header.frame_id = "sim_frame"      # parent
        t.child_frame_id  = "vicon_frame"    # child

        # translation of vicon origin expressed in sim_frame
        t.transform.translation.x = tx
        t.transform.translation.y = ty
        t.transform.translation.z = tz

        # rotation: +90° clockwise (−π/2 yaw)
        t.transform.rotation.x = 0.0
        t.transform.rotation.y = 0.0
        t.transform.rotation.z = -0.70710678
        t.transform.rotation.w =  0.70710678

        return

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
    
    def seed(self, seed=None):
        self.rngs = make_seeded_rngs(seed)
        self._seed = self.rngs["actual_seed"]
        self.np_random = self.rngs["np_random"]
        self.torch_rng = self.rngs["torch_rng"]
        return [self._seed]

def main(args=None):
    rclpy.init(args=args)
    node = GameServer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    pass