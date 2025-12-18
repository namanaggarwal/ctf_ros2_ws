#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from dynus_interfaces.msg import State
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
            State,
            f"/{self.rover_name}/global_term_goal",
            10
        )

        # goal pose in global frame
        self.goal_pose = [2.0, 0.0, 0.0]
        self.goal_vel = [0.0, 0.0, 0.0]

        # create timer
        self.timer = self.create_timer(2.0, self.publish_goal)

        self.get_logger().info(f"Global Goal Publisher Initialized")


    def publish_goal(self):

        # create goal message
        msg = State()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "world"

        # set position
        msg.pos.x = self.goal_pose[0]
        msg.pos.y = self.goal_pose[1]
        msg.pos.z = self.goal_pose[2]

        # set velocity
        msg.vel.x = self.goal_vel[0]
        msg.vel.y = self.goal_vel[1]
        msg.vel.z = self.goal_vel[2]

        # identity
        msg.quat.x = 0.0
        msg.quat.y = 0.0
        msg.quat.z = 0.0
        msg.quat.w = 1.0

        # publish goal
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
