import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped

from ctf_msgs.msg import RoverStatus, FlagState, GameState, JoinGameMessage, ServerToRoverMessage
from ctf_msgs.srv import RequestGameState

class RoverNode(Node):
    def __init__(self, **kwargs):
        super().__init__("ctf")
        self.grid_size = 10
        self.ctf_player_config = kwargs.get('ctf_player_config', '2v2')

        self.rover_name = 'RR03' # Read from a config file / YAML file.
        self.rover_team_name = 'RED' # Read from a config file / YAML file.

        self.join_game_topic = "/ctf/join"
        self.publisher_join_game_topic = self.create_publisher(
                JoinGameMessage,
                self.join_game_topic,
            )
        self.get_logger().info("Rover node for ROVER {} started, waiting for rover to join...".format(self.rover_name))

        self.server_to_rover_topic = "{}/server_to_rover".format(self.rover_name)
        self.subscriber_server_to_rover_topic = self.create_subscription(
                ServerToRoverMessage,
                self.server_to_rover_topic,
                self.server_to_rover_callback,
                10,
            )
        
        self.local_dynus_pub_goal_topic = '/{}/term_goal'.format(self.rover_name)
        self.publisher_local_dynus_command_goal = self.create_publisher(
                PoseStamped,
                self.local_dynus_pub_goal_topic,
            )
        
        """
        # Service for rover to request game state
        self.srv = self.create_service(
            RequestGameState,
            "ctf/get_state",
            self.handle_get_state
        )
        """

    def server_to_rover_callback(self, msg: ServerToRoverMessage):
        command = msg.command
        commanded_pose = msg.commanded_pose

        if command == 'INIT':
            local_planner_msg = commanded_pose
            self.publisher_local_dynus_command_goal.publish(local_planner_msg)        
        return


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
    node = RoverNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    pass