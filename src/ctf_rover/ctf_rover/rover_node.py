import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped

from ctf_msgs.msg import JoinGameMessage, ServerToRoverMessage
#from ctf_msgs.srv import RequestGameState

class RoverNode(Node):
    def __init__(self, **kwargs):
        super().__init__("ctf")
        self.grid_size = 10
        self.ctf_player_config = kwargs.get('ctf_player_config', '2v2')

        # Pending: declare_parameters for rover_name and rover_team_name.
        self.rover_name = 'RR03' # Read from a config file / YAML file.
        self.rover_team_name = 'RED' # Read from a config file / YAML file.

        self.join_game_topic = "/ctf/join"
        self.publisher_join_game_topic = self.create_publisher(
                JoinGameMessage,
                self.join_game_topic,
                10
            )
        
        msg = JoinGameMessage()
        msg.rover_name = self.rover_name
        msg.rover_team_name = self.rover_team_name
        self.publisher_join_game_topic.publish(msg)

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
                10
            )
        
        """
        # Service for rover to request game state
        self.srv = self.create_service(
            RequestGameState,
            "ctf/get_state",
            self.handle_get_state
        )
        """
        # INIT: Reach goal and pose.
        # START GAME: Generate waypoint and publish to goal term. Use a VICON callback that generates a new waypoint once the previous goal is reached: compare current VICON position to the current published goal_term.
        # Game Server declares Game Termination on flag reaching or truncation. For example, play for 15 seconds and see if flag reached, if not truncate and kill and restart a new instance of the game.

    def server_to_rover_callback(self, msg: ServerToRoverMessage):
        command = msg.command
        commanded_pose = msg.commanded_pose

        if command == 'INIT':
            local_planner_msg = commanded_pose
            self.publisher_local_dynus_command_goal.publish(local_planner_msg)
            self.game_play_callback() # While goal is in progress, go to goal and then replan on seeing the world state.
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