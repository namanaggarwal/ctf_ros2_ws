import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point


class PublishObs(Node):

    def __init__(self):
        super().__init__('publish_obs')
        self.graph_pub = self.create_publisher(Marker, 'obs_graph', 10)
        self.timer = self.create_timer(0.1, self.publish_obs)

        # obs corners
        a = 0.61
        self.corners = [(0., -a), (0., a), (-3 * a, a), (-3 * a, 0.), (-2 * a, 0.), (-2 * a, -a)]

        self.frame_id = "world"
    
    def publish_obs(self):

        # create node markers
        node_marker = Marker()
        node_marker.header.frame_id = self.frame_id
        node_marker.header.stamp = self.get_clock().now().to_msg()

        node_marker.ns = "obs_nodes"
        node_marker.id = 0
        node_marker.type = Marker.SPHERE_LIST
        node_marker.action = Marker.ADD

        node_marker.scale.x = 0.1
        node_marker.scale.y = 0.1
        node_marker.scale.z = 0.1

        node_marker.color.r = 1.0
        node_marker.color.g = 1.0
        node_marker.color.b = 0.0
        node_marker.color.a = 1.0

        for x, y in self.corners:
            p = Point()
            p.x = float(x)
            p.y = float(y)
            p.z = 0.0
            node_marker.points.append(p)

        self.graph_pub.publish(node_marker)

        # create edge markers
        edge_marker = Marker()
        edge_marker.header.frame_id = self.frame_id
        edge_marker.header.stamp = self.get_clock().now().to_msg()

        edge_marker.ns = "obs_edges"
        edge_marker.id = 1
        edge_marker.type = Marker.LINE_LIST
        edge_marker.action = Marker.ADD

        edge_marker.scale.x = 0.05

        edge_marker.color.r = 1.0
        edge_marker.color.g = 1.0
        edge_marker.color.b = 0.0
        edge_marker.color.a = 1.0

        n = len(self.corners)

        for i in range(n):
            x1, y1 = self.corners[i]
            x2, y2 = self.corners[(i + 1) % n]  # wrap around

            p1 = Point()
            p1.x = float(x1)
            p1.y = float(y1)
            p1.z = 0.0

            p2 = Point()
            p2.x = float(x2)
            p2.y = float(y2)
            p2.z = 0.0

            edge_marker.points.append(p1)
            edge_marker.points.append(p2)

        self.graph_pub.publish(edge_marker)


def main(args=None):
    rclpy.init(args=args)
    publish_obs = PublishObs()  
    rclpy.spin(publish_obs)  
    publish_obs.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()