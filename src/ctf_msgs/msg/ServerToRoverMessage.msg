# Server to Rover message

# Command type: INIT, START, STOP, MOVE, RESPAWN, TAG etc.
string command

# Commanded pose for the rover
# geometry_msgs/PoseStamped commanded_pose
dynus_interfaces/State commanded_goal

# Full roster: parallel arrays mapping physical rover names to CTF agent names.
# Populated by the server on INIT so each rover can build rr_to_ctf locally.
string[] roster_rover_names
string[] roster_ctf_agent_names