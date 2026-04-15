import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point


class PublishGraph(Node):

    def __init__(self):
        super().__init__('publish_graph')
        self.graph_pub = self.create_publisher(Marker, 'ctf_graph', 10)
        self.timer = self.create_timer(0.1, self.publish_graph)

        # self.nodes = [(0, 0), (1, 1), (2, 0), (3, 1)]
        # self.nodes = []
        self.nodes = [(x, y) for x in range(-5, 6) for y in range(-5, 6)]
        # self.edges = [(0, 1), (1, 2), (2, 3)]
        self.edges = [
            (i, j)
            for i, (x1, y1) in enumerate(self.nodes)
            for j, (x2, y2) in enumerate(self.nodes)
            if i < j and abs(x1 - x2) + abs(y1 - y2) == 1
        ]

    def publish_graph(self):

        # create node markers
        node_marker = Marker()
        node_marker.header.frame_id = "world"
        node_marker.header.stamp = self.get_clock().now().to_msg()

        node_marker.ns = "ctf_graph_nodes"
        node_marker.id = 0
        node_marker.type = Marker.SPHERE_LIST
        node_marker.action = Marker.ADD

        node_marker.scale.x = 0.1  # point width
        node_marker.scale.y = 0.1
        node_marker.scale.z = 0.1

        node_marker.color.r = 1.0
        node_marker.color.g = 1.0
        node_marker.color.b = 0.0
        node_marker.color.a = 1.0

        # iterate through graph and publish nodes
        for x, y in self.nodes:

            # point for each node
            p = Point()
            p.x = float(x)
            p.y = float(y)
            p.z = 0.0
            node_marker.points.append(p)

        # publish nodes
        self.graph_pub.publish(node_marker)

        # create edge markers
        edge_marker = Marker()
        edge_marker.header.frame_id = "world"
        edge_marker.header.stamp = self.get_clock().now().to_msg()

        edge_marker.ns = "ctf_graph_nodes"
        edge_marker.id = 1
        edge_marker.type = Marker.LINE_LIST
        edge_marker.action = Marker.ADD

        edge_marker.scale.x = 0.05

        edge_marker.color.r = 1.0
        edge_marker.color.g = 1.0
        edge_marker.color.b = 0.0
        edge_marker.color.a = 0.1

        for i, j in self.edges:
            p1 = Point()
            p1.x = float(self.nodes[i][0])
            p1.y = float(self.nodes[i][1])
            p1.z = 0.0

            p2 = Point()
            p2.x = float(self.nodes[j][0])
            p2.y = float(self.nodes[j][1])
            p2.z = 0.0

            edge_marker.points.append(p1)
            edge_marker.points.append(p2)

        # publish edges
        self.graph_pub.publish(edge_marker)

def main(args=None):
    rclpy.init(args=args)
    publish_graph = PublishGraph()  
    rclpy.spin(publish_graph)  
    publish_graph.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()