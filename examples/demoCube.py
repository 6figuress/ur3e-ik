import numpy as np
from scipy.spatial.transform import Rotation as R
from ur_ikfast import ur_kinematics
import pandas as pd
import matplotlib.pyplot as plt
import json


if __name__ == "__main__":
    # Initialize the UR3e Kinematics
    ur3e_arm = ur_kinematics.URKinematics("ur3e")

def compute_inverse_kinematics(position, roll, pitch, yaw, last_joint=None):
    # Convert roll, pitch, yaw to quaternion
    quaternion = R.from_euler('xyz', [roll, pitch, yaw], degrees=True).as_quat()
    q_w, q_x, q_y, q_z = quaternion  # Extract quaternion elements
    
    # Construct the 7-element vector format [tx, ty, tz, qw, qx, qy, qz]
    ik_input = np.array([*position, q_w, q_x, q_y, q_z])
    
    # Compute inverse kinematics
    if last_joint is not None:
        joint_solutions = ur3e_arm.inverse(ik_input, q_guess=last_joint)
    else:
        joint_solutions = ur3e_arm.inverse(ik_input)
    
    return joint_solutions

def generate_trajectory_file(data, filename="trajectory.json"):
    modTraj = []
    time_step = 2  # Incrément du temps
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
    
    print(f"Fichier JSON '{filename}' généré avec succès.")


def generate_path_square(origin=(0, 0, 0), direction=(1, 0, 0), width=0.1, length=0.1, step=0.01, depth=0.01):
    """
    Generate a zigzag path in a square space with a defined origin and movement direction.

    Parameters:
    - origin (tuple): (x0, y0, z0) starting position.
    - direction (tuple): (dx, dy, dz) vector indicating movement direction.
    - width (float): The width of the zigzag area.
    - height (float): The height of the zigzag area.
    - step (float): The step size for movement.
    - depth (float): How much the path dips when going down.

    Returns:
    - List of (x, y, z) points representing the path.
    """
    x0, y0, z0 = origin
    dx, dy, dz = direction
    
    # Normalize the direction vector to ensure step sizes are correct
    norm = np.sqrt(dx**2 + dy**2 + dz**2)
    dx, dy, dz = dx / norm, dy / norm, dz / norm  # Unit vector

    path = []
    
    # Generate positions along the main movement direction
    steps = np.arange(0, width + step, step)  # Move along width
    for i, s in enumerate(steps):
        # Compute main path movement using direction vector
        x = x0 + s * dx
        y = y0 + s * dy
        z = z0 + s * dz  # Z can change if direction includes dz

        # Zigzag motion along height (perpendicular movement)
        if i % 2 == 0:
            y_positions = np.linspace(y, y + length, num=len(steps))
        else:
            y_positions = np.linspace(y + length, y + depth, num=len(steps))

        for y_pos in y_positions:
            path.append((x, y_pos, z))  # Store the position
    
    return path

def compute_orientation_towards_target(origin, target, offset=(0.0, 0.0, 0.0)):
    direction = np.array(target) - np.array(origin)
    direction = direction / np.linalg.norm(direction)

    # Compute yaw (rotation around z-axis)
    yaw = np.arctan2(direction[1], direction[0])
    
    # Compute pitch (rotation around y-axis)
    pitch = np.arcsin(-direction[2])   # Assuming standard convention where downward is negative
    
    # Roll is assumed to be zero (no rotation around x-axis)
    roll = 0.0 
    
    return np.degrees(roll) + offset[0], np.degrees(pitch) + offset[1] , np.degrees(yaw) + offset[2]


def transform_coordinates_to_joint_angles(coordinates, orientation=(None, None, None)):
    joint_angles = []

    last_joint = None
    
    for position in coordinates:
        # check if orientation is provided and for each three angles not None
        roll, pitch, yaw = orientation
        if roll is not None and pitch is not None and yaw is not None:
            pass

        else:
            print("Orientation is not provided. Computing orientation towards target bottom.")
            # Offset to look at the bottom of the cube
            offset=(0.0, 89.9, 0.0)
            roll, pitch, yaw = compute_orientation_towards_target(origin=position, target=[position[0], position[1], 0.0], offset=offset)
        joint = compute_inverse_kinematics(position, roll, pitch, yaw, last_joint)
        if joint is not last_joint:
            last_joint = joint
            joint_angles.append(joint)
        else:
            print("No solution found for position", position)
            pass
    return joint_angles

if __name__ == "__main__":
    # Generate and visualize the path
    # origin is the start position of the cube first corner where it will start filling the cube (and z is the height of the cube, so the arm will stay to the same altitude)
    # direction is the direction of the cube filling
    # width is the width of the cube
    # length is the length of the cube
    # step is the step of the filling (how much the arm will move in the cube)
    # depth is how much the arm will go down when it reach the end of the cube
    path = generate_path_square(
        origin=(0.2, 0.2, 0.2),
        direction=(1, 0, 0),
        width=0.15,
        length=0.15,
        step=0.05,
        depth=0.05,
    )

    print("Number of paths :", len(path))

    print("Computing joint angles...")

    # Compute joint angles for the generated path (list of (x, y, z) points). Could be used with other paths such as circle, triangle, line, etc.
    joint_trajectory = transform_coordinates_to_joint_angles(
        path, orientation=(0, 179.942, 0)
    )  # Orientation, roll, pitch and yaw, defined in degree. 180 to look to the bottom (180 degree make issue, so 179.942 is the closest value to 180 degree that works)
    print("Joint Trajectory for Square:")
    print(joint_trajectory)

    # Generate trajectory file, to be used with the UR robot (URSim or real robot)
    generate_trajectory_file(joint_trajectory, filename="square_trajectory.json")