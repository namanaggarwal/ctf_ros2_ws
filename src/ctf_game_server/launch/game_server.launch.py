from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():

    params_file = PathJoinSubstitution([
        FindPackageShare("ctf_game_server"),
        "config",
        "params.yaml"
    ])

    return LaunchDescription([
        Node(
            package="ctf_game_server",
            executable="game_server",
            name="game_server",
            output="screen",
            parameters=[params_file],
        )
    ])