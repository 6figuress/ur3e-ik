import json
import math
import numpy as np
from ur_ikfast.best_trajectory import TrajectoryPlanner

from security import ValidateRobotPosition

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
    time_step = int(2000000000)  # Incrément du temps [us]
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


def quaternion_from_matrix(matrix):
    """Return quaternion from rotation matrix.
    >>> R = rotation_matrix(0.123, (1, 2, 3))
    >>> q = quaternion_from_matrix(R)
    >>> numpy.allclose(q, [0.0164262, 0.0328524, 0.0492786, 0.9981095])
    True
    """
    q = np.empty((4, ), dtype=np.float64)
    M = np.array(matrix, dtype=np.float64, copy=False)[:4, :4]
    t = np.trace(M)
    if t > M[3, 3]:
        q[3] = t
        q[2] = M[1, 0] - M[0, 1]
        q[1] = M[0, 2] - M[2, 0]
        q[0] = M[2, 1] - M[1, 2]
    else:
        i, j, k = 0, 1, 2
        if M[1, 1] > M[0, 0]:
            i, j, k = 1, 2, 0
        if M[2, 2] > M[i, i]:
            i, j, k = 2, 0, 1
        t = M[i, i] - (M[j, j] + M[k, k]) + M[3, 3]
        q[i] = t
        q[j] = M[i, j] + M[j, i]
        q[k] = M[k, i] + M[i, k]
        q[3] = M[k, j] - M[j, k]
    q *= 0.5 / math.sqrt(t * M[3, 3])
    return q

def pose_quaternion_from_matrix(matrix):
    """Return translation + quaternion(x,y,z,w)
    """
    if matrix.shape == (3, 4):
        matrix = np.concatenate((matrix, [[0, 0, 0, 1]]), axis=0)

    pose = matrix[:3, 3]
    quat = quaternion_from_matrix(matrix)
    return np.concatenate((pose, quat), axis=0)

class URKinematics():

    def __init__(self, robot_name):
        if robot_name == 'ur3e':
            import ur3e_ikfast as ur_ikfast
        elif robot_name == 'ur3e_pen_21':
            import ur3e_pen_21_ikfast as ur_ikfast
        elif robot_name == 'ur3e_pen_150':
            import ur3e_pen_150_ikfast as ur_ikfast
        elif robot_name == 'ur3e_pen_165':
            import ur3e_pen_165_ikfast as ur_ikfast
        elif robot_name == 'ur3e_pen_203':
            import ur3e_pen_203_ikfast as ur_ikfast
        elif robot_name == 'ur3e_pen_final':
            import ur3e_pen_final_ikfast as ur_ikfast
        else:
            raise Exception("Unsupported robot")

        self.kinematics = ur_ikfast.PyKinematics()
        self.n_joints = self.kinematics.getDOF()

    def forward(self, joint_angles, rotation_type='quaternion'):
        """
            Compute robot's forward kinematics for the specified robot
            joint_angles: list
            rotation_type: 'quaternion' or 'matrix'
            :return: if 'quaternion' then return a list of [x, y, z, qx, qy, qz, w]
                     if 'matrix' then a list of 12 values the 3x3 rotation matrix and 
                     the 3 translational values
        """
        if isinstance(joint_angles, np.ndarray):
            joint_angles = joint_angles.tolist()

        ee_pose = self.kinematics.forward(joint_angles)
        ee_pose = np.asarray(ee_pose).reshape(3, 4)

        if rotation_type == 'matrix':
            return ee_pose
        elif rotation_type == 'quaternion':
            return pose_quaternion_from_matrix(ee_pose)

    def inverse(self, ee_pose, all_solutions=False, q_guess=np.zeros(6), max_retries=5, pertubation=1e-3):
        """ Compute robot's inverse kinematics for the specified robot
            ee_pose: list of 7 if quaternion [x, y, z, qx, qy, qz, w]
                     list of 12 if rotation matrix + translational values
            all_solutions: whether to return all the solutions found or just the best one
            q_guess:  if just one solution is request, this set of joint values will be use
                      to find the closest solution to this
            :return: list of joint angles
                     list of best joint angles if found
                     q_guess if no solution is found
        """
        pose = None
        if len(ee_pose) == 7:
            rot = np.roll(ee_pose[3:], 1)
            pose = np.concatenate((ee_pose[:3], rot), axis=0)
        else:
            pose = ee_pose

        for attempt in range(max_retries):
            joint_configs = self.kinematics.inverse(pose.reshape(-1).tolist())
            n_solutions = int(len(joint_configs)/self.n_joints)
            joint_configs = np.asarray(joint_configs).reshape(n_solutions, self.n_joints)

            if n_solutions > 0:
                if all_solutions:
                    return joint_configs
                return best_ik_sol(joint_configs, q_guess)
            
            # Pertubate the pose and try again if no solution is found
            # print(f"Failed to find a solution, pertubating the pose by {pertubation} for attempt {attempt+1}")
            if len(ee_pose) == 7:
                pose[3:] += np.random.uniform(-pertubation, pertubation, 4)
                pose[3:] /= np.linalg.norm(pose[3:])
            else:
                pose[3:12] += np.random.uniform(-pertubation, pertubation, 9)

        # print("Failed to find a solution")
        return None
    
def best_ik_sol(sols, q_guess, weights=np.ones(6)):
    """
    Get best IK solution

    Parameters:
        sols: list of joint angles
        q_guess: set of joint values to find the closest solution to
        weights: weights for each joint
    Returns:
        best solution
    """
    valid_sols = []
    for sol in sols:
        test_sol = np.ones(6) * 9999.
        for i in range(6):
            for add_ang in [-2. * np.pi, 0, 2. * np.pi]:
                test_ang = sol[i] + add_ang
                if (abs(test_ang) <= 2. * np.pi
                        and abs(test_ang - q_guess[i]) <
                        abs(test_sol[i] - q_guess[i])):
                    test_sol[i] = test_ang
        if np.all(test_sol != 9999.):
            valid_sols.append(test_sol)
    if not valid_sols:
        return None
    best_sol_ind = np.argmin(
        np.sum((weights * (valid_sols - np.array(q_guess)))**2, 1))
    return valid_sols[best_sol_ind]

class MultiURKinematics():
    """
    This class is used to compute the forward and inverse kinematics for a list of positions.
    """
    def __init__(self, kinematics: URKinematics):
        """
        Initialize the MultiURKinematics object.
        Parameters:
            kinematics (URKinematics): URKinematics object to use for the computations
        """
        self.kinematics = kinematics
        self.planner = TrajectoryPlanner()

    def inverse(self, ee_poses, all_solutions=False, q_guess=np.zeros(6), max_retries=5, pertubation=1e-3):
        """
        Compute the inverse kinematics for a list of end-effector poses.
        
        Parameters:
            ee_poses (list): List of end-effector poses to compute the inverse kinematics for
            all_solutions (bool): Whether to return all the solutions found or just the best one
            q_guess (np.ndarray): Set of joint values to find the closest solution to
            max_retries (int): Maximum number of retries to find a solution
            pertubation (float): Pertubation to apply to the pose if no solution is found
            
        Returns:
            List of joint angles for each end-effector pose (list of lists)
        """
        solutions = []

        for ee_pose in ee_poses:
            sol = self.kinematics.inverse(ee_pose=ee_pose, all_solutions=all_solutions, q_guess=q_guess, max_retries=max_retries, pertubation=pertubation)
            if sol is not None:
                solutions.append(sol.tolist())
        return solutions
    
    def number_valid_solutions(self, ee_poses, logs=False):
        """
        Check if there is a valid IK solution for the given end-effector poses.
        
        Parameters:
            ee_poses (list): List of end-effector poses to check
            
        Returns:
            list(tuple(ee_pose, float)): List of tuples containing the end-effector pose and the number of corresponding solutions
        """

        valid_sols = []
        for ee_pose in ee_poses:
            solutions = self.kinematics.inverse(ee_pose=ee_pose, all_solutions=True)

            if solutions is None:
                if logs:
                    print(f"No solutions found for pose {ee_pose}")
                valid_sols.append((ee_pose, 0))
                continue

            number_of_solutions = len(solutions)

            secure_solution = ValidateRobotPosition([solutions], logs=False).finalPositions

            number_of_secure_solutions = len(secure_solution)

            if logs:
                print(f"Pose {ee_pose} has {number_of_solutions} solutions, {number_of_secure_solutions} of which are secure")

            if secure_solution[0] is not None:
                valid_sols.append((ee_pose, len(secure_solution)))
            else:
                valid_sols.append((ee_pose, 0))
        return valid_sols

    def inverse_optimal(
        self, ee_poses, q_guess=np.zeros(6), max_retries=5, pertubation=1e-3, logs=True
    ):
        """
        Compute the optimal joint trajectory for a list of end-effector poses.

        Parameters:
            ee_poses (list): List of end-effector poses to compute the inverse kinematics for
            q_guess (np.ndarray): Set of joint values to find the closest solution to
            max_retries (int): Maximum number of retries to find a solution
            pertubation (float): Pertubation to apply to the pose if no solution is found

        Returns:
            List of joint angles for the optimal trajectory
        """
        solutions = self.inverse(
            ee_poses=ee_poses,
            all_solutions=True,
            q_guess=q_guess,
            max_retries=max_retries,
            pertubation=pertubation,
        )

        def checkForSolutions(l) -> bool:
            if not l:
                if logs:
                    print("No solutions found")
                return False
            valid = True
            for i, s in enumerate(solutions):
                if len(s) == 0:
                    if logs:
                        print(f"No solutions found for point : {i}")
                    valid = False

            return valid

        if not checkForSolutions(solutions):
            return None

        secure_solution = ValidateRobotPosition(solutions, logs=logs).finalPositions

        if not checkForSolutions(secure_solution):
            return None

        best_trajectory = self.planner.best_first_search(secure_solution)

        return best_trajectory
    
    def forward(self, joint_angles, rotation_type='quaternion'):
        """
        Compute the forward kinematics for a list of joint angles.

        Parameters:
            joint_angles (list): List of joint angles to compute the forward kinematics for
            rotation_type (str): Type of rotation to return ('quaternion' or 'matrix')

        Returns:
            List of end-effector poses for each joint angle (list of lists)
        """
        
        poses = []
        for joint_angle in joint_angles:
            pose = self.kinematics.forward(joint_angle, rotation_type)
            poses.append(pose)
        return poses