# Server to Rover message

# Command type: INIT, START, STOP, MOVE, RESPAWN, TAG etc.
string command

# Commanded pose for the rover
# geometry_msgs/PoseStamped commanded_pose
dynus_interfaces/State commanded_goal