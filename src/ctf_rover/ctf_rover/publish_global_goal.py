#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
import os

# Publish the goal at self.goal_pose in the global (world) frame
class PublishGlobalGoal(Node):

    def __init__(self):
        super().__init__("publish_global_goal")

        # get rover name
        vehtype = os.getenv("VEHTYPE")
        vehnum = os.getenv("VEHNUM")
        self.rover_name = vehtype + vehnum

        # create publisher to terminal global goal topic
        self.publisher = self.create_publisher(
            PoseStamped,
            f"/{self.rover_name}/global_term_goal",
            10
        )

        # goal pose in global frame
        self.goal_pose = [2.0, 0.0, 0.0]

        # create timer
        self.timer = self.create_timer(2.0, self.publish_goal)

        self.get_logger().info(f"Global Goal Publisher Initialized")


    def publish_goal(self):

        # create goal message
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "world"

        msg.pose.position.x = self.goal_pose[0]
        msg.pose.position.y = self.goal_pose[1]
        msg.pose.position.z = self.goal_pose[2]

        msg.pose.orientation.x = 0.0
        msg.pose.orientation.y = 0.0
        msg.pose.orientation.z = 0.0
        msg.pose.orientation.w = 1.0

        self.publisher.publish(msg)
        self.get_logger().info(f"Published global goal: {self.goal_pose}")


def main(args=None):
    rclpy.init(args=args)
    node = PublishGlobalGoal()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
