import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped

from ctf_msgs.msg import RoverStatus, FlagState, GameState, JoinGameMessage
from ctf_msgs.srv import RequestGameState

import functools
import random
import copy
import numpy as np
import torch
from gymnasium.utils import seeding

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

        self.join_game_topic = "/ctf/join"
        self.subscriber_join_game_topic = self.create_subscription(
                JoinGameMessage,
                self.join_game_topic,
                self.join_game_callback,
                10,
            )
        self.get_logger().info("GameServer started, waiting for rovers to join...")

        rover_names = self.declare_parameter(
            "rover_names", ["rover1", "rover2", "rover3", "rover4"]
        ).value # RR03, RR04, RR05, RR06
        
        """
        for name in rover_names:
            self.create_subscription(
                PoseStamped,
                f"/{name}/world",
                lambda msg, name=name: self.vicon_callback(msg, name),
                10,
            )
            
            self.create_subscription(
                PoseStamped,
                f"/vicon/{name}/pose",
                lambda msg, name=name: self.vicon_callback(msg, name),
                10,
            )
        """

        # Service for rover to request game state
        self.srv = self.create_service(
            RequestGameState,
            "ctf/get_state",
            self.handle_get_state
        )

    def join_game_callback(self, msg: JoinGameMessage):
        if self.num_rovers == 4:
            return
        
        rover_name = msg.rover_name
        rover_team_name = msg.rover_team_name
        rover_pose_topic = "/{}/world".format(rover_name)
        if rover_name in self.rovers_list:
            self.get_logger().warn(f"[GAMESERVER]: ROVER {rover_name} already joined.")
            return
        self.num_rovers += 1

        self.rovers_list.append(rover_name)
        self.rovers_info[rover_name] = {
            'team': rover_team_name,
            'pose_topic': rover_pose_topic
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
        return

    def seed(self, seed=None):
        self.rngs = make_seeded_rngs(seed)
        self._seed = self.rngs["actual_seed"]
        self.np_random = self.rngs["np_random"]
        self.torch_rng = self.rngs["torch_rng"]
        return [self._seed]

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

def main(args=None):
    rclpy.init(args=args)
    node = GameServer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    pass