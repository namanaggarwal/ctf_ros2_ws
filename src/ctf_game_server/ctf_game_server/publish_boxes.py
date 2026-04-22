import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point, PoseStamped
import math

class PublishBoxes(Node):

    def __init__(self):
        super().__init__('publish_boxes')

        self.marker_pub = self.create_publisher(Marker, 'box_edges', 10)
        self.big_marker_pub = self.create_publisher(Marker, 'big_box_edges', 10)

        ### SMALL BOXES ###
        # edge length
        self.small_a = 0.31

        # store poses: { "BOX0": PoseStamped, ... }
        self.box_poses = {}

        # number of boxes
        self.num_boxes = 6

        # create subscribers
        for i in range(self.num_boxes):
            topic = f'/BOX{i}/world'
            self.create_subscription(
                PoseStamped,
                topic,
                lambda msg, i=i: self.pose_callback(msg, i),
                10
            )
        
        ### BIG BOXES ###
        # edge length
        self.big_a = 0.6

        # store poses: { "BOX0": PoseStamped, ... }
        self.big_box_poses = {}

        # number of boxes
        self.num_big_boxes = 2

        # create subscribers
        for i in range(self.num_big_boxes):
            topic = f'/BIGBOX{i}/world'
            self.create_subscription(
                PoseStamped,
                topic,
                lambda big_msg, i=i: self.big_pose_callback(big_msg, i),
                10
            )

        self.timer = self.create_timer(0.1, self.publish_boxes)


    def pose_callback(self, msg, i):
        self.box_poses[i] = msg
    
    def big_pose_callback(self, msg, i):
        self.big_box_poses[i] = msg
    
    def publish_boxes(self):

        marker = Marker()
        marker.header.frame_id = "world"
        marker.header.stamp = self.get_clock().now().to_msg()

        marker.ns = "boxes"
        marker.id = 0
        marker.type = Marker.LINE_LIST
        marker.action = Marker.ADD

        marker.scale.x = 0.05

        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 1.0

        half = self.small_a / 2.0

        for i, pose_msg in self.box_poses.items():

            cx = pose_msg.pose.position.x
            cy = pose_msg.pose.position.y
            cz = 0.0

            q = pose_msg.pose.orientation

            # quat to yaw
            yaw = math.atan2(
                2.0 * (q.w * q.z + q.x * q.y),
                1.0 - 2.0 * (q.y * q.y + q.z * q.z)
            )

            cos_y = math.cos(yaw)
            sin_y = math.sin(yaw)

            # square centered
            local_corners = [
                (-half, -half),
                ( half, -half),
                ( half,  half),
                (-half,  half),
            ]

            # rotate & translate
            world_corners = []
            for x_local, y_local in local_corners:
                x_rot = cos_y * x_local - sin_y * y_local
                y_rot = sin_y * x_local + cos_y * y_local

                world_corners.append((cx + x_rot, cy + y_rot))

            # get edges
            for j in range(4):
                x1, y1 = world_corners[j]
                x2, y2 = world_corners[(j + 1) % 4]

                p1 = Point()
                p1.x = x1
                p1.y = y1
                p1.z = cz

                p2 = Point()
                p2.x = x2
                p2.y = y2
                p2.z = cz

                marker.points.append(p1)
                marker.points.append(p2)

        self.marker_pub.publish(marker)


        ### BIG BOX MARKER ###
        big_marker = Marker()
        big_marker.header.frame_id = "world"
        big_marker.header.stamp = self.get_clock().now().to_msg()

        big_marker.ns = "big_boxes"
        big_marker.id = 0
        big_marker.type = Marker.LINE_LIST
        big_marker.action = Marker.ADD

        big_marker.scale.x = 0.05

        big_marker.color.r = 1.0
        big_marker.color.g = 1.0
        big_marker.color.b = 0.0
        big_marker.color.a = 1.0

        half = self.big_a / 2.0

        for i, pose_msg in self.big_box_poses.items():

            cx = pose_msg.pose.position.x
            cy = pose_msg.pose.position.y
            cz = 0.0

            q = pose_msg.pose.orientation

            # quat to yaw
            yaw = math.atan2(
                2.0 * (q.w * q.z + q.x * q.y),
                1.0 - 2.0 * (q.y * q.y + q.z * q.z)
            )

            cos_y = math.cos(yaw)
            sin_y = math.sin(yaw)

            # square centered
            local_corners = [
                (-half, -half),
                ( half, -half),
                ( half,  half),
                (-half,  half),
            ]

            # rotate & translate
            world_corners = []
            for x_local, y_local in local_corners:
                x_rot = cos_y * x_local - sin_y * y_local
                y_rot = sin_y * x_local + cos_y * y_local

                world_corners.append((cx + x_rot, cy + y_rot))

            # get edges
            for j in range(4):
                x1, y1 = world_corners[j]
                x2, y2 = world_corners[(j + 1) % 4]

                p1 = Point()
                p1.x = x1
                p1.y = y1
                p1.z = cz

                p2 = Point()
                p2.x = x2
                p2.y = y2
                p2.z = cz

                big_marker.points.append(p1)
                big_marker.points.append(p2)

        self.big_marker_pub.publish(big_marker)


def main(args=None):
    rclpy.init(args=args)
    node = PublishBoxes()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()