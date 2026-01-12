import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped

from ctf_msgs.msg import JoinGameMessage, ServerToRoverMessage
#from ctf_msgs.srv import RequestGameState

import functools
import random
import copy
import numpy as np
# import torch
# from gymnasium.utils import seeding

def make_seeded_rngs(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    np_random, actual_seed = seeding.np_random(seed)
    torch_rng = torch.Generator().manual_seed(actual_seed)
    return {
        "np_random": np_random,
        "torch_rng": torch_rng,
        "actual_seed": actual_seed
    }

class GameServer(Node):
    def __init__(self, **kwargs):
        super().__init__("ctf")
        self.grid_size = 10
        self.ctf_player_config = kwargs.get('ctf_player_config', '2v2')
        self.ctf_env = None # !? --> move CTF custom environment to within the same directory and import CustomCTF_v1.

        # Subscriptions for each rover pose from VICON
        # (You can generate these dynamically)
        """
        Rover_names: Updated from rover subscriptions, subscribing and participating in the game.
        """
        self.num_rovers = 0
        self.rovers_list = []
        self.rovers_info = {}
        self.rovers_state = {}
        self.rover_pose_subscriptions = {}
        self.server_to_rover_publishers = {}

        self.blue_agents = []
        self.red_agents = []
        self.num_agents_blue_team = None
        self.num_agents_red_team = None

        self.join_game_topic = "/ctf/join"
        self.subscriber_join_game_topic = self.create_subscription(
                JoinGameMessage,
                self.join_game_topic,
                self.join_game_callback,
                10,
            )
        self.get_logger().info("GameServer started, waiting for rovers to join...")
        
        for rover in self.rovers_list:
            server_to_rover_channel = self.rovers_info[rover]['server_to_rover_channel']
            self.publisher_serve_to_rover = self.create_publisher(
                    ServerToRoverMessage,
                    server_to_rover_channel,
                )
        
        self.game_started = False

        """
        # Service for rover to request game state
        self.srv = self.create_service(
            RequestGameState,
            "ctf/get_state",
            self.handle_get_state
        ) 
        """

    def join_game_callback(self, msg: JoinGameMessage):
        if self.num_rovers == 4:
            return
        
        rover_name = msg.rover_name
        rover_team_name = msg.rover_team_name
        rover_pose_topic = "/{}/world".format(rover_name)
        server_to_rover_topic = "/{}/server_to_rover".format(rover_name)
        if rover_name in self.rovers_list:
            self.get_logger().warn(f"[GAMESERVER]: ROVER {rover_name} already joined.")
            return
        self.num_rovers += 1

        self.rovers_list.append(rover_name)
        self.rovers_info[rover_name] = {
            'team': rover_team_name,
            'pose_topic': rover_pose_topic,
            'server_to_rover_topic': server_to_rover_topic
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

        if self.num_rovers == 4:
            self.blue_agents = [agent for agent in self.rovers_list if self.rovers_info[agent]['team'] == 'BLUE']
            self.red_agents = [agent for agent in self.rovers_list if self.rovers_info[agent]['team'] == 'RED']
            self.num_agents_blue_team = len(self.blue_agents)
            self.num_agents_red_team = len(self.red_agents)
            
            self.start_game_callback()
        return

    def start_game_callback(self):
        # Send initial commanded poses to start the game.
        # Wait for response.
        self.get_logger().info("[GAMESERVER] All rovers joined. Sending initial poses...")
        initial_poses = self.compute_initial_poses()  # method to define starting positions

        for rover_name, pose in initial_poses.items():
            msg = ServerToRoverMessage()
            msg.command = 'INIT'

            pose_msg = PoseStamped()
            pose_msg.header.stamp = self.get_clock().now().to_msg()
            pose_msg.header.frame_id = "world"

            pose_msg.pose.position.x = pose[0]
            pose_msg.pose.position.y = pose[1]
            pose_msg.pose.position.z = pose[2]

            pose_msg.pose.orientation.x = 0.0
            pose_msg.pose.orientation.y = 0.0
            pose_msg.pose.orientation.z = 0.0
            pose_msg.pose.orientation.w = 1.0

            msg.commanded_pose = pose_msg
            self.server_to_rover_publishers[rover_name].publish(msg)
            self.get_logger().info(f"[GAMESERVER] Sent initial pose to {rover_name}")

        # Wait for confirmation from rovers or add a short delay
        # Then mark game as started
        self.game_started = True
        self.get_logger().info("[GAMESERVER] Game started!")
        return
    
    def compute_initial_poses(self):
        RR03_init_pose = (+2.8, -1.4, 0)
        RR06_init_pose = (+0, +1.4, 0)
        pose_dict = {'RR03': RR03_init_pose, 'RR06': RR06_init_pose}
        return pose_dict #self.reset()
    
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
            if agent_iter == 0:
                rand_x = self.np_random.integers(0, self.grid_size)
                rand_y = self.np_random.integers(self.grid_size - self.red_init_spawn_y_lim - 1, self.grid_size)
                rand_theta = self._sample_init_heading(agent_team="Red", x=rand_x, y=rand_y)
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

        #global_state = [self.state[agent] for agent in self.agents]
        #obs = {agent: global_state for agent in self.agents} # global_state dict. The prisoner guard example on ParallelEnv has this as a tuple, and you have this as a List currently.
        obs = self._observations()
        info = {agent: {} for agent in self.agents}
        #print("self.agents: {}".format(self.agents))
        return obs, info
    
    def discrete_grid_abstraction_to_highbay_coordinates(self, discrete_x, discrete_y, heading):
        assert discrete_x in range(self.grid_size)
        assert discrete_y in range(self.grid_size)
        assert heading in range(8)
        a = 0.762 # meters
        delta_x = a / 2. # meters
        delta_y = a / 2. # meters

        x, y = a*discrete_x, a*discrete_y # game ctf discrete frame

        x_vicon, y_vicon = 4*a + delta_x, 4*a + delta_y # vicon coords in game ctf discrete frame

        # game ctf -> vicon frame: 90 degrees counter-clockwise
        # !? : Convert above to quaternion rotation and build tf from this rotation and x_vicon, y_vicon.
        # !?2: Then convert x, y from game_ctf_frame to vicon_frame.

        heading_in_radians = heading * np.pi / 4. # game_ctf_frame
        # !? convert above to vicon frame.

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