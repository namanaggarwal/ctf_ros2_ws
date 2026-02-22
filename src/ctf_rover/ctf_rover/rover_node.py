import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, TransformStamped
from dynus_interfaces.msg import State
from scipy.spatial.transform import Rotation as R

from ctf_msgs.msg import JoinGameMessage, ServerToRoverMessage
#from ctf_msgs.srv import RequestGameState

import tf2_ros
from tf2_ros import TransformException
from tf2_ros import TransformBroadcaster
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster
import os
import numpy as np

class RoverNode(Node):
    def __init__(self, **kwargs):
        super().__init__("ctf")
        self.grid_size = 10
        self.ctf_player_config = kwargs.get('ctf_player_config', '2v2')

        self.goal_height = -0.01

        # Pending: declare_parameters for rover_name and rover_team_name.
        # self.rover_name = 'RR03' # Read from a config file / YAML file.
        vehtype = os.getenv("VEHTYPE")
        vehnum = os.getenv("VEHNUM")
        self.rover_name = vehtype + vehnum

        self.declare_parameter("team", "RED")

        self.rover_team_name = self.get_parameter("team").value

        self.get_logger().info(f"TEAM = {self.rover_team_name}")
        # self.rover_team_name = 'RED' # Read from a config file / YAML file.

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
        
        self.local_dynus_pub_goal_topic = '/{}/term_goal'.format(self.rover_name) #'/{}/term_goal'.format(self.rover_name)
        self.publisher_local_dynus_command_goal = self.create_publisher(
                State,
                self.local_dynus_pub_goal_topic,
                10
            )

        # create tf buffer for global goal
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # create tf broadcaster for map --> init pose (world --> RR0X/map)
        self.world_map_broadcaster = StaticTransformBroadcaster(self)

        # init global goal variables
        self.X_map_world = None

        self.init_timer = self.create_timer(0.2, self.initialize_tf) # timer to init tf
        self.global_frame = "world"
        self.initial_frame = self.rover_name
        self.local_frame = f"{self.rover_name}/map"

        self.get_logger().info("ROVER NODE INITIALIZED")
        
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
        # Maintain a queue of waypoints: pop and add.
        ### Idea: Receeding horizon prediction for fast planning and then replanning every now and then depending on the game being played: with slow adversaries, can be faster. With fast adversaries, need to be strategic.
        ### VICON Callback: either waypoint reach within goal tolerance or if collision avoidance activated with an adversary agent, respawn.        
    
    # initialize tf from world to map
    def initialize_tf(self):

        # return if already initialized
        if self.X_map_world is not None:
            return

        # try to get the transform
        try:

            # get transform from world to init pose
            # X_world_RR
            tf = self.tf_buffer.lookup_transform(
                self.global_frame,             
                self.initial_frame,              
                rclpy.time.Time()
            )
        except TransformException:
            self.get_logger().info("Waiting for initial TF")
            return 

        self.get_logger().info("Successfully got TF!")

        # Convert TF to 4x4 matrix
        t = tf.transform.translation
        q = tf.transform.rotation
        r = R.from_quat([q.x, q.y, q.z, q.w])

        X_world_rr = np.eye(4)
        X_world_rr[:3, :3] = r.as_matrix()
        X_world_rr[:3, 3] = np.array([t.x, t.y, t.z])

        # publish transform for map --> init pose (world --> RR0X/map)
        T_world_map = TransformStamped()

        T_world_map.header.stamp = self.get_clock().now().to_msg()
        T_world_map.header.frame_id = 'world'
        T_world_map.child_frame_id = self.local_frame

        T_world_map.transform.translation.x = t.x
        T_world_map.transform.translation.y = t.y
        T_world_map.transform.translation.z = t.z # 0.0?

        T_world_map.transform.rotation.x = q.x
        T_world_map.transform.rotation.y = q.y
        T_world_map.transform.rotation.z = q.z
        T_world_map.transform.rotation.w = q.w

        self.world_map_broadcaster.sendTransform(T_world_map)

        # X world wrt map = inverse(X RR wrt world)
        self.X_map_world = np.linalg.inv(X_world_rr)

        # stop the timer
        self.init_timer.cancel()
        self.get_logger().info("Fixed world to map transform saved.")
    
    def server_to_rover_callback(self, msg: ServerToRoverMessage):
        command = msg.command
        commanded_goal = msg.commanded_goal        
        self.get_logger().info("ENTERING server_to_rover_callback ...")

        if command == 'INIT':
            # if self.X_map_world is None:
            #     self.get_logger().warn("world to map not initialized yet. Ignoring goal.")
            #     return

            self.initialize_tf()

            self.get_logger().info("PUBLISHING GOAL ...")
            # local_planner_msg = commanded_goal
            global_planner_msg = commanded_goal

            # get global pose
            p = global_planner_msg.pos

            self.get_logger().info(f"[ROVER] Converting Global Goal [{p.x}, {p.y}, {p.z}] to local frame")

            global_point = np.array([p.x, p.y, p.z, 1.0])

            # transform into map (local) frame
            # X_localFrame_localGoal = X_localFrame_globalFrame @ X_globalFrame_globalGoal
            # where X_localFrame_globalFrame = X_map_world
            # and X_globalFrame_globalGoal = global_point
            local_point = self.X_map_world @ global_point

            # get global velocity
            v = global_planner_msg.vel
            global_vel = np.array([v.x, v.y, v.z])

            R_map_world = self.X_map_world[:3, :3]  # rotation only
            local_vel = R_map_world @ global_vel

            # create local goal
            local_goal = State()
            local_goal.header.stamp = global_planner_msg.header.stamp
            local_goal.header.frame_id = self.local_frame

            # set local position
            local_goal.pos.x = local_point[0]
            local_goal.pos.y = local_point[1]
            local_goal.pos.z = self.goal_height #local_point[2]

            # set local velocity
            local_goal.vel.x = local_vel[0]
            local_goal.vel.y = local_vel[1]
            local_goal.vel.z = local_vel[2]

            # ignore yaw (already set with velocity vector)
            local_goal.quat.x = 0.0
            local_goal.quat.y = 0.0
            local_goal.quat.z = 0.0
            local_goal.quat.w = 1.0

            self.get_logger().info(f"[ROVER] Publishing local goal [{local_point[0]}, {local_point[1]}, {local_point[2]}]")

            self.publisher_local_dynus_command_goal.publish(local_goal)
            # self.game_play_callback() # While goal is in progress, go to goal and then replan on seeing the world state.
        return

    def seed(self, seed=None):
        self.rngs = make_seeded_rngs(seed)
        self._seed = self.rngs["actual_seed"]
        self.np_random = self.rngs["np_random"]
        self.torch_rng = self.rngs["torch_rng"]
        return [self._seed]
    
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