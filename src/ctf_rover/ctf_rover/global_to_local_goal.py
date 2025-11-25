#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

from geometry_msgs.msg import PoseStamped
import numpy as np
from scipy.spatial.transform import Rotation as R

import tf2_ros
from tf2_ros import TransformException
import os

# Use the initial pose of the robot to get the transform to publish
# the received global goal in the local frame
# X_A_B = B wrt A
class GlobalToLocalGoal(Node):

    def __init__(self):
        super().__init__("global_to_local_goal")

        # create tf listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.X_map_world = None

        # timer to init tf
        self.init_timer = self.create_timer(0.2, self.initialize_tf)

        # get rover name
        vehtype = os.getenv("VEHTYPE")
        vehnum = os.getenv("VEHNUM")
        self.rover_name = vehtype + vehnum

        # create subscriber to global goal topic
        self.global_goal_sub = self.create_subscription(
            PoseStamped,
            f"/{self.rover_name}/global_term_goal",
            self.global_goal_callback,
            10
        )

        # create publisher to local goal topic for dynus
        self.local_goal_pub = self.create_publisher(
            PoseStamped,
            "/RR04/term_goal",
            10
        )

        # initialize frame names
        self.global_frame = "world"
        self.initial_frame = self.rover_name
        self.local_frame = f"{self.rover_name}/map"

    # initialize tf from world to map
    def initialize_tf(self):

        # return if already initialized
        if self.X_map_world is not None:
            return

        # try to get the transform
        try:

            # get transform from world to init pose
            # X_world_RR04
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

        X_world_rr04 = np.eye(4)
        X_world_rr04[:3, :3] = r.as_matrix()
        X_world_rr04[:3, 3] = np.array([t.x, t.y, t.z])

        # X world wrt map = inverse(X RR04 wrt world)
        self.X_map_world = np.linalg.inv(X_world_rr04)

        # stop the timer
        self.init_timer.cancel()
        self.get_logger().info("Fixed world to map transform saved.")


    # convert global goal to local goal
    def global_goal_callback(self, msg: PoseStamped):
        if self.X_map_world is None:
            self.get_logger().warn("world to map not initialized yet. Ignoring goal.")
            return

        # get global pose
        p = msg.pose.position
        global_point = np.array([p.x, p.y, p.z, 1.0])

        # transform into map (local) frame
        # X_localFrame_localGoal = X_localFrame_globalFrame @ X_globalFrame_globalGoal
        # where X_localFrame_globalFrame = X_map_world
        # and X_globalFrame_globalGoal = global_point
        local_point = self.X_map_world @ global_point

        # create local goal
        local_goal = PoseStamped()
        local_goal.header.stamp = msg.header.stamp
        local_goal.header.frame_id = self.local_frame

        local_goal.pose.position.x = local_point[0]
        local_goal.pose.position.y = local_point[1]
        local_goal.pose.position.z = local_point[2]

        local_goal.pose.orientation.x = 0.0
        local_goal.pose.orientation.y = 0.0
        local_goal.pose.orientation.z = 0.0
        local_goal.pose.orientation.w = 1.0

        self.local_goal_pub.publish(local_goal)
        self.get_logger().info(f"Published local goal {local_point}")


def main(args=None):
    rclpy.init(args=args)
    node = GlobalToLocalGoal()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
