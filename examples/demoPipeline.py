import numpy as np
from ur_ikfast import ur_kinematics
import json

# Initialize the UR3e Kinematics
ur3e_arm = ur_kinematics.URKinematics('ur3e')

def compute_inverse_kinematics(position, quaternion, last_joint=None):
    """Compute all inverse kinematics solutions for a given position and quaternion orientation."""
    ik_input = np.array([*position, *quaternion])
    if last_joint is not None:
        joint_solutions = ur3e_arm.inverse(ik_input, q_guess=last_joint)
    else:
        joint_solutions = ur3e_arm.inverse(ik_input)
    return joint_solutions

def generate_trajectory_file(data, filename="trajectory.json"):
    modTraj = []
    #time_step = 1e6   # 1e6 ns = 1 ms
    time_step = 1
    time = 4
    
    for arr in data:
        positions = [round(float(x), 4) if abs(x) >= 1e-4 else 0.0 for x in arr]
        velocities = [0.0] * 6  # Vélocités à zéro
        modTraj.append({
            "positions": positions,
            "velocities": velocities,
            "time_from_start": [time, 0]
        })
        time += time_step
    
    with open(filename, "w") as f:
        json.dump({"modTraj": modTraj}, f, indent=4)
    
    print(f"Trajectory file '{filename}' generated successfully.")

def load_points_cloud(filename="paths.json"):
    """Load the points cloud from the JSON file."""
    with open(filename) as f:
        paths = json.load(f)
    
    return paths

def transform_coordinates_to_joint_angles(coordinates, orientations):
    """Transform path coordinates into joint angles using inverse kinematics, with specific quaternion orientation per point."""
    joint_trajectories = []
    last_joint = None
    
    for position, quaternion in zip(coordinates, orientations):
        solution = compute_inverse_kinematics(position, quaternion, last_joint)
        if solution is not None:
            last_joint = solution
            joint_trajectories.append(solution)
    
    return joint_trajectories

# hardcoded, position of the tapped cube near the robot
# juts for tests
offset = [84/1000.0, 323/1000.0, 293/1000.0/2.0]

paths = {}

path = load_points_cloud()[0][1]
orientations = [p[3:] for p in path]
orientations = [[o[0] - 0.001, o[1] - 0.001, o[2] - 0.001, o[3] - 0.001] for o in orientations]
path = [p[:3] for p in path]

path = [[p[0] + offset[0], p[1] + offset[1], p[2] + offset[2]] for p in path]

# take only one ine 20 points for faster simulation
# path = path[::20]

# Compute all joint solutions for each path point
joint_trajectory = transform_coordinates_to_joint_angles(path, orientations)
print(f"Computed {len(joint_trajectory)} joint solutions for the trajectory.")

print(joint_trajectory)

# Generate trajectory file
generate_trajectory_file(joint_trajectory, filename="3dmodel_trajectory.json")
