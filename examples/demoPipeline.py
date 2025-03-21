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
    time_step = 1e6 * 1000   # 1e6 ns = 1 ms
    #time_step = 3
    ns_time = 0
    s_time = 0
    
    for arr in data:
        ns_time += time_step

        if ns_time >= 1e9:
            s_time += 1
            ns_time = 0

        positions = [round(float(x), 4) if abs(x) >= 1e-4 else 0.0 for x in arr]
        velocities = [0.0] * 6  # Vélocités à zéro
        modTraj.append({
            "positions": positions,
            "velocities": velocities,
            "time_from_start": [int(s_time), int(ns_time)]
        })
    
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
offset = [250/1000.0, 223/1000.0, 293/1000.0]

paths = {}


pp = load_points_cloud()
i = 0

for ppp in pp:
    i += 1
    path = ppp[1]
    path = [(*p[0], *p[1]) for p in path]

    orientations = [p[3:] for p in path]
    path = [p[:3] for p in path]

    path = [[p[0]*1.0 + offset[0], p[1]*1.0 + offset[1], p[2]*1.0 + offset[2]] for p in path]

    # Take only every 50th point for faster simulation 
    # path = path[::50]

    # Compute all joint solutions for each path point
    joint_trajectory = transform_coordinates_to_joint_angles(path, orientations)
    print(f"Computed {len(joint_trajectory)} joint solutions for the trajectory.")

    if len(joint_trajectory) != len(path):
        print(f"Warning: The number of joint solutions ({len(joint_trajectory)}) does not match the number of path points ({len(path)}).")

    # Generate trajectory file
    generate_trajectory_file(joint_trajectory, filename=f"3dmodel_trajectory__{i}.json")
