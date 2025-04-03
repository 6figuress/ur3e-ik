import math
import numpy as np
from scipy.spatial.transform import Rotation as R
from ur_ikfast import ur_kinematics
import pandas as pd
import matplotlib.pyplot as plt
import json

# Initialize the UR3e Kinematics
#ur3e_arm = ur_kinematics.URKinematics('ur3e')
ur3e_arm = ur_kinematics.URKinematics('ur3e_pen_gripper')

def compute_inverse_kinematics(pos_raw, previous_angles):
    """
    Computes the inverse kinematics from a desired position. Also uses the angles of the previous position to help decide which
    of the returned solutions is the best (the one that generates the least joint movement).

    Parameters
    ----------
    pos_raw : list[float]
        The desired position. it is structured like this: [x, y, z, qw, qx, qy, qz] [m, m, m, -, -, -, -]
    previous_angles : list[float]
        The angles that are currently applied to the robot arm. [rad, rad, rad, rad, rad, rad]

    Returns
    -------
    numpy.ndarray
        The angles [rad] of the 6 joints of the robot that will place the endpoint of the robot in the desired location, from base to wrists (so j1 -> j6).
    """
    # Splits the pos_raw input
    position = pos_raw[0:3]
    q_w, q_x, q_y, q_z = pos_raw[3:7]

    
    # Constructs the 7-element vector format [tx, ty, tz, qx, qy, qz, qw]
    ik_input = np.array([*position, q_x, q_y, q_z, q_w])
    
    # Computes inverse kinematics
    joint_solutions = ur3e_arm.inverse(ik_input, q_guess=previous_angles)
    return joint_solutions



def compute_joint_trajectory_for_points_tab(points, home):
    """
    Computes the inverse kinematics for a list of points.

    Parameters
    ----------
    points : list[list[float]]
        All the points in the trajectory. each point has to contain a valid position, so [x, y, z, qw, qx, qy, qz] [m, m, m, -, -, -, -]
    home: list[float]
        The home position of the robot. It will considered as the current position of the robot and the one to return to inbetween actions.
        It is mainly used to chose the first inverse kinematics solution, since we use the difference of all angles for our optimization.

    Returns
    -------
    list[[list[float]]]
        A list of all the joint positions to apply to the robot arm to make the endpoints pass by all desired positions. all angles are in [rad]. The values are ordered from base to wrists (so j1 -> j6)
    """
    # variable for the previous position of the arm, the first one being the home position
    previous_point = home

    joint_trajectory = []
    for point in points:
        # for each point in the points list
        # computes the inverse kinematics
        joint_angles = compute_inverse_kinematics(point, previous_point)
        if joint_angles is not None:
            # if a solution was found
            # adds the angles to the trajectory and sets the previous point for the next iteration
            joint_trajectory.append(joint_angles)
            previous_point = joint_angles
        else:
            # if no solution was found
            print(f"[ERROR] Could not find coordinates for the point: {point}")
    return joint_trajectory


def generate_trajectory_file(data, filename="trajectory.json"):
    """
    Generates a JSON trajectory file from the computed joint trajectory data.

    Parameters
    ----------
    data : list[list[float]]
        The joint trajectory data containing joint positions.
    filename : str, optional
        The name of the output JSON file (default is "trajectory.json").

    Returns
    -------
    None
        The function writes the trajectory data into a JSON file.
    """
    modTraj = []
    time_step = 2000000000  # Incrément du temps [us]
    time = 4000000000 # Temps pour aller a la position initiale [us]
    
    for arr in data:
        positions = [round(float(x), 4) if abs(x) >= 1e-4 else 0.0 for x in arr]
        velocities = [0.0] * 6  # Vélocités à zéro
        ns_time = time % 1000000000
        s_time = int((time - ns_time)/1000000000)
        modTraj.append({
            "positions": positions,
            "velocities": velocities,
            "time_from_start": [s_time, ns_time]
        })
        time += time_step
    
    with open(filename, "w") as f:
        json.dump({"modTraj": modTraj}, f, indent=4)
    
    print(f"Fichier JSON '{filename}' généré avec succès.")



def interpolate_lines(points, threshold):
    """
    Interpolates the lines in a trajectory (list of points) if they exceed a given threshold.

    Parameters
    ----------
    points : list[list[float]]
        A list of points respecting the following format: [x, y, z, qw, qx, qy, qz] [m, m, m, -, -, -, -]
    threshold : 
        The maximum distance that can be done before doing interpolation [m].

    Returns
    -------
    list[list[float]]    
        The input trajectories with added points from interpolation
       
    """
    # the first point can't change
    new_points = [points[0]]

    for i in range(1,len(points)):
        # for each point in the points list (starting from the second element)
        # gets the two points
        p1 = points[i-1]
        p2 = points[i]

        # splits the position
        x1,y1,z1 = p1[0:3]
        x2,y2,z2 = p2[0:3]

        # gets the difference between the two points
        dif_x = x2-x1
        dif_y = y2-y1
        dif_z = z2-z1 

        # computes the distance between the 2 points
        dist = math.sqrt(dif_x * dif_x + dif_y * dif_y + dif_z * dif_z)

        if dist > threshold:
            # if the distance is bigger than the threshold
            # computes how many parts the new division will have
            n_parts = int(dist / threshold)

            for j in range(1, n_parts):
                # for each step of the interpolation to be done
                # computes the next point and adds it to the list
                new_point = [round(x1 + j*dif_x/n_parts,5), round(y1 + j*dif_y/n_parts,5), round(z1 + j*dif_z/n_parts,5), p1[3], p1[4], p1[5], p1[6]]
                new_points.append(new_point)

        # adds the last point of the interpolated line
        new_points.append(p2)
    
    return new_points


def back_up_point(point, length):
    """
    Backs up a point along his direction vector.

    Parameters
    ----------
    point : list[float]
        The starting point: [x, y, z, qw, qx, qy, qz] [m, m, m, -, -, -, -]
    length : float
        The distance to back up the point [m].

    Returns
    -------
    list[float]   
        The backed up point: [x, y, z, qw, qx, qy, qz] [m, m, m, -, -, -, -]
    """

    # splits infos from the point
    pos = point[:3]
    r = R.from_quat(point[3:7], scalar_first=True)

    # gets the direction vector 
    direction = r.apply([0, 0, 1])  # Direction vector of the end effector

    # Normalize the direction vector (just in case)
    direction = direction / np.linalg.norm(direction)

    # Compute the end position
    end_pos = pos + direction * length

    # prints results for verification
    #print(pos)
    #print(end_pos)

    # returns the new point
    return([end_pos[0], end_pos[1], end_pos[2], point[3], point[4], point[5], point[6]])




# ============================================================================================
# ===                                       2D shapes                                     ====
# ============================================================================================
# test zone de travail
#traj_points = [[0, 0.3, 0.4], [0.3, -0.1, 0.4]]
# Z shape
'''traj_points = [
    [0.064, 0.303, 0.0], 
    [0.064, 0.343, 0.0], 
    [0.10400000000000001, 0.303, 0.0], 
    [0.10400000000000001, 0.343, 0.0], 
    [0.10400000000000001, 0.343, 0.05]
]'''
# Z shape then hit table
#traj_points = [[0.104, 0.343, 0.2], [0.104, 0.343, 0.1], [0.104, 0.343, 0], [0.104, 0.343, -0.1], [0.104, 0.343, 0.3]]
# rectangle shape
#traj_points = [[0.094, 0.2931, 0.35650000000000004], [0.114, 0.2931, 0.35650000000000004], [0.114, 0.2931, 0.38650000000000007], [0.094, 0.29460000000000003, 0.38650000000000007], [0.094, 0.29460000000000003, 0.35650000000000004], [0.114, 0.29460000000000003, 0.35650000000000004], [0.114, 0.29460000000000003, 0.38650000000000007], [0.094, 0.29610000000000003, 0.38650000000000007], [0.094, 0.29610000000000003, 0.35650000000000004], [0.114, 0.29610000000000003, 0.35650000000000004], [0.114, 0.29610000000000003, 0.38650000000000007], [0.094, 0.2976, 0.38650000000000007], [0.094, 0.2976, 0.35650000000000004], [0.114, 0.2976, 0.35650000000000004], [0.114, 0.2976, 0.38650000000000007], [0.094, 0.2991, 0.38650000000000007], [0.094, 0.2991, 0.35650000000000004], [0.114, 0.2991, 0.35650000000000004], [0.114, 0.2991, 0.38650000000000007], [0.094, 0.3006, 0.38650000000000007], [0.094, 0.3006, 0.35650000000000004], [0.114, 0.3006, 0.35650000000000004], [0.114, 0.3006, 0.38650000000000007], [0.094, 0.3021, 0.38650000000000007], [0.094, 0.3021, 0.35650000000000004], [0.114, 0.3021, 0.35650000000000004], [0.114, 0.3021, 0.38650000000000007], [0.094, 0.3036, 0.38650000000000007], [0.094, 0.3036, 0.35650000000000004], [0.114, 0.3036, 0.35650000000000004], [0.114, 0.3036, 0.38650000000000007], [0.094, 0.3051, 0.38650000000000007], [0.094, 0.3051, 0.35650000000000004], [0.114, 0.3051, 0.35650000000000004], [0.114, 0.3051, 0.38650000000000007], [0.094, 0.3066, 0.38650000000000007], [0.094, 0.3066, 0.35650000000000004], [0.114, 0.3066, 0.35650000000000004], [0.114, 0.3066, 0.38650000000000007], [0.094, 0.3081, 0.38650000000000007], [0.094, 0.3081, 0.35650000000000004], [0.114, 0.3081, 0.35650000000000004], [0.114, 0.3081, 0.38650000000000007], [0.094, 0.3096, 0.38650000000000007], [0.094, 0.3096, 0.35650000000000004], [0.114, 0.3096, 0.35650000000000004], [0.114, 0.3096, 0.38650000000000007], [0.094, 0.3111, 0.38650000000000007], [0.094, 0.3111, 0.35650000000000004], [0.114, 0.3111, 0.35650000000000004], [0.114, 0.3111, 0.38650000000000007], [0.094, 0.3126, 0.38650000000000007], [0.094, 0.3126, 0.35650000000000004], [0.114, 0.3126, 0.35650000000000004], [0.114, 0.3126, 0.38650000000000007], [0.094, 0.3141, 0.38650000000000007], [0.094, 0.3141, 0.35650000000000004], [0.114, 0.3141, 0.35650000000000004], [0.114, 0.3141, 0.38650000000000007], [0.094, 0.3156, 0.38650000000000007], [0.094, 0.3156, 0.35650000000000004], [0.114, 0.3156, 0.35650000000000004], [0.114, 0.3156, 0.38650000000000007], [0.094, 0.3171, 0.38650000000000007], [0.094, 0.3171, 0.35650000000000004], [0.114, 0.3171, 0.35650000000000004], [0.114, 0.3171, 0.38650000000000007], [0.094, 0.3186, 0.38650000000000007], [0.094, 0.3186, 0.35650000000000004], [0.114, 0.3186, 0.35650000000000004], [0.114, 0.3186, 0.38650000000000007], [0.094, 0.3201, 0.38650000000000007], [0.094, 0.3201, 0.35650000000000004], [0.114, 0.3201, 0.35650000000000004], [0.114, 0.3201, 0.38650000000000007], [0.094, 0.3216, 0.38650000000000007], [0.094, 0.3216, 0.35650000000000004], [0.114, 0.3216, 0.35650000000000004], [0.114, 0.3216, 0.38650000000000007]]
# adds the angles for the 2D shape that will be drawn, 180,0,0 (looking down)
'''traj_points = [
    [244.32, 128.5, 340.5, 3.720, -0.023, 0.099],
    [-46.43, 217.58, 376.47, 2.000, -1.975, -0.091],
    [385.26, 79.67, 281.63, 2.990, 0.704, 0.006],
    #[208.78, 437.98, 401.75, 0.008, -2.904, 0.724],
    [-224.5, 337.9, 242.5, 1.739, -2.103, -0.176],
    [260.7, -69.5, 319.4, 2.844, 0.169, 0.018],
    #[472.5, 22.6, 192.0, 2.345, 1.133, -0.401],
    #[-334.8, 235.2, 249.1, 2.075, -1.724, -0.071]
]'''




joint_trajectory = []

#for i in traj_points:
#    joint_trajectory.append([i[0]/1000,i[1]/1000,i[2]/1000,math.degrees(i[3]),math.degrees(i[4]),math.degrees(i[5])])
#for i in traj_points:
    #joint_trajectory.append([i[0],i[1],i[2],0.01,1.01,0.01,0.01])
    #r = R.from_quat([0.01,0.01,1.01,0.01], scalar_first=True)
    #normal = r.as_mrp()
    #length = -0.25
    #pos = i[:3]
    #print(pos)
    #end_pos = (np.array(pos) - np.array(normal) * length)  # Ending at the correct point
    #print(end_pos)
    #joint_trajectory.append([end_pos[0], end_pos[1], end_pos[2], 0.01, 1.01, 0.01, 0.01])

#print(joint_trajectory)


# ============================================================================================
# ===                                       3D shapes                                     ====
# ============================================================================================
'''joint_trajectory = [
    [ 0.0957, 0.35  , 0.231 ,-0.2363, 0.3032, 0.8525, 0.3541],
    [ 0.0829, 0.3978, 0.231 ,-0.4271,-0.1594, 0.8485, 0.2686],
    [ 0.0478, 0.4329, 0.231 ,-0.5495,-0.5581, 0.6138, 0.0992],
    [ 5.8582e-18, 4.4567e-01, 2.3097e-01, 5.7080e-01, 7.8599e-01,-2.1114e-01,1.0877e-01],
    [-0.0478, 0.4329, 0.231 , 0.4853, 0.782 , 0.2515, 0.2995],
    [-0.0829, 0.3978, 0.231 , 0.3158, 0.5473, 0.6502, 0.4219],
    [-0.0957, 0.35  , 0.231 , 0.1079, 0.1446, 0.8781, 0.4432],
    [-0.0829, 0.3022, 0.231 ,-0.0828,-0.318 , 0.8741, 0.3577],
    [-0.0478, 0.2671, 0.231 , 0.2053, 0.7167,-0.6393,-0.1883],
    [-1.7574e-17, 2.5433e-01, 2.3097e-01, 2.2656e-01, 9.4459e-01,-2.3669e-01,1.9690e-02],
    [0.0478,0.2671,0.231 ,0.141 ,0.9406,0.226 ,0.2104],
    [ 0.0829, 0.3022, 0.231 ,-0.0284, 0.7059, 0.6246, 0.3328],
    [ 0.1733, 0.3851, 0.1768,-0.4715, 0.2003, 0.8157, 0.2687],
    [ 0.1325, 0.467 , 0.1768,-0.6316,-0.226 , 0.7254, 0.154 ],
    [ 0.0562, 0.5176, 0.1768, 0.7129, 0.5502,-0.4341, 0.0253],
    [-0.0351, 0.5233, 0.1768, 0.6937, 0.6852,-0.0197, 0.2213],
    [-0.117 , 0.4825, 0.1768, 0.579 , 0.5949, 0.4066, 0.3814],
    [-0.1676, 0.4062, 0.1768, 0.3997, 0.3036, 0.7308, 0.4627],
    [-0.1733, 0.3149, 0.1768, 0.2037,-0.1108, 0.8658, 0.4434],
    [-0.1325, 0.233 , 0.1768, 0.0436,-0.5371, 0.7756, 0.3288],
    [-0.0562, 0.1824, 0.1768, 0.0377, 0.8613,-0.4842,-0.1494],
    [ 0.0351, 0.1767, 0.1768, 0.0184, 0.9963,-0.0698, 0.0465],
    [ 0.117 , 0.2175, 0.1768,-0.0962, 0.9061, 0.3565, 0.2066],
    [ 0.1676, 0.2938, 0.1768,-0.2756, 0.6147, 0.6807, 0.2879],
    [ 0.2127, 0.4399, 0.0957,-0.6719, 0.115 , 0.7132, 0.1633],
    [ 0.1393, 0.5343, 0.0957, 0.7926, 0.2449,-0.5574,-0.0335],
    [ 0.0285, 0.5792, 0.0957, 0.8323, 0.4787,-0.2424, 0.1392],
    [-0.0899, 0.5627, 0.0957, 0.7802, 0.5237, 0.1472, 0.3086],
    [-0.1843, 0.4893, 0.0957, 0.6505, 0.3679, 0.5071, 0.4293],
    [-0.2292, 0.3785, 0.0957, 0.4778, 0.053 , 0.7409, 0.469 ],
    [-0.2127, 0.2601, 0.0957, 0.3084,-0.3367, 0.786 , 0.4169],
    [-0.1393, 0.1657, 0.0957,-0.1877, 0.6966,-0.6301,-0.2872],
    [-0.0285, 0.1208, 0.0957,-0.148 , 0.9304,-0.3152,-0.1145],
    [ 0.0899, 0.1373, 0.0957,-0.2   , 0.9754, 0.0744, 0.0549],
    [ 0.1843, 0.2107, 0.0957,-0.3298, 0.8196, 0.4344, 0.1756],
    [ 0.2292, 0.3215, 0.0957,-0.5025, 0.5046, 0.6682, 0.2153]
]'''
# points of the trajectory
'''joint_trajectory_raw = [
    [ 0.0957, 0.35  , 0.231 ,-0.2363, 0.3032, 0.8525, 0.3541],
    [ 0.0829, 0.3978, 0.231 ,-0.4271,-0.1594, 0.8485, 0.2686],
    [-0.0829, 0.3978, 0.231 , 0.3158, 0.5473, 0.6502, 0.4219]
]'''

# similar points with different orientations
'''joint_trajectory = [
    [ 0.01, 0.35, 0.0, -0.346543784287115, -0.296221799766215, 0.88919633413857, -0.0385996148913219],
    [ 0.01, 0.35, 0.0, -0.288390842651529, 0.353087341828493, 0.868185927289021, 0.195992975824476],
    [ 0.01, 0.35, 0.0, 0.346543784287114, 0.296221799766215, 0.889196334138571, 0.0385996148913219],
]'''

# test martin pipeline
joint_trajectory = [
    [ 0.0148, 0.3101, 0.1981, 0.00003, 0.9999, 0.0039, -0.0036],
]

'''joint_trajectory = [
    [ 0.0, 0.35, 0.0, 0.01, 1.01, 0.01, 0.01],]
[ 0.01, 0.35, 0.215, 0.2, 0.0, 1, 0.0],
    [ 0.01, 0.35, 0.210, 0.2, 0.0, 1, 0.0],
    [ 0.01, 0.35, 0.205, 0.2, 0.0, 1, 0.0],
    [ 0.01, 0.35, 0.200, 0.2, 0.0, 1, 0.0],
    [ 0.01, 0.35, 0.195, 0.2, 0.0, 1, 0.0],
    [ 0.01, 0.35, 0.190, 0.2, 0.0, 1, 0.0],
    [ 0.01, 0.35, 0.185, 0.2, 0.0, 1, 0.0],
    [ 0.01, 0.35, 0.180, 0.2, 0.0, 1, 0.0],
    [ 0.01, 0.35, 0.195, 0.2, 0.0, 1, 0.0],
    [ 0.01, 0.35, 0.190, 0.2, 0.0, 1, 0.0],
    [ 0.01, 0.35, 0.185, 0.2, 0.0, 1, 0.0],
    [ 0.01, 0.35, 0.180, 0.2, 0.0, 1, 0.0],
]'''

# test pour guillaume porte stylo
'''joint_trajectory = [
    [0.1664148371899884, 0.3001690082761688, 0.18520303487360162, 0.0008, 0.9999, 0, 0], 
    [0.1513499618294426, 0.30212194444001716, 0.12551891421236722, 0.0008, 0.9999, 0, 0], 
    [0.1664148371899884, 0.3001690082761688, 0.18520303487360162, 0.0008, 0.9999, 0, 0], 
    [0.11181405488384043, 0.24194206538072482, 0.1841540860427493, 0.0008, 0.9999, 0, 0], 
    [0.09674917952329463, 0.24389500154457322, 0.1244699653815149, 0.0008, 0.9999, 0, 0], 
    [0.11181405488384043, 0.24194206538072482, 0.1841540860427493, 0.0008, 0.9999, 0, 0]
]'''

'''joint_trajectory = []
for i in joint_trajectory_raw:
    # for each point in the trajectory
    joint_trajectory.append(i)

    # Compute the back up point
    length = -0.1
    back_up = back_up_point(i, length)
    joint_trajectory.append(back_up)
'''
#print("==================================================================================================")
#print(f"Voila les points demandés en entrée: ({len(joint_trajectory)} points)")
#print(joint_trajectory)
#print("==================================================================================================")

# interpolates the trajectory
interpolated_traj = interpolate_lines(joint_trajectory, 0.6)

#print("==================================================================================================")
#print(f"Voila les points demandés en entrée avec des intermediaires pour raccourcir la distance: ({len(interpolated_traj)} points)")
#print(interpolated_traj)
#print("==================================================================================================")

# computes the joint trajectory for all the points we need it to go through
home = [0.950855376, -1.66225158, 0.635299848, -0.597600736, -1.57219259, 0]
traj_0 = compute_joint_trajectory_for_points_tab(interpolated_traj, home)

# adds home position at start and finish
#traj = [home] + traj + [home]
traj = [home]
for i in traj_0:
    traj = traj + [i] + [home]

print(interpolated_traj)
# Generate trajectory file
generate_trajectory_file(traj, filename="traj.json")