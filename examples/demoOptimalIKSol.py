from ur_ikfast import ur_kinematics
import numpy as np
from scipy.spatial.transform import Rotation as R

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

def transform_coordinates_to_ee_poses(coordinates):
    """Transform path coordinates into joint angles using inverse kinematics."""
    ik_inputs = []
    
    for position in coordinates:
        orientation = [0, 179.942, 0]
        quaternion = R.from_euler('xyz', orientation, degrees=True).as_quat()
        ik_input = np.array([*position, *quaternion])
        ik_inputs.append(ik_input)

    return ik_inputs

def main():
    robot = ur_kinematics.URKinematics('ur3e')
    multi_kin = ur_kinematics.MultiURKinematics(robot)

    path = generate_path_square(origin=(0.2, 0.2, 0.2), width=0.15, length=0.15, step=0.05)

    ik_inputs = transform_coordinates_to_ee_poses(path)

    joint_trajectory = multi_kin.inverse_optimal(ik_inputs)

    print(joint_trajectory)


