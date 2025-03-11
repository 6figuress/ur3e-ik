import numpy as np
from ur_ikfast import ur_kinematics
from itertools import combinations, product
import copy
import time


sol_pos_1 = [
    [2, 40, 55],
    [2, 60, 65],
    [2, 15, 45],
    [7, 80, 90],
    [3, 22, 55],
    [9, 45, 78],
]
sol_pos_2 = [
    [6, 20, 95],
    [6, 5, 88],
    [6, 8, 82],
    [6, 10, 75],
    [6, 3, 78],
    [7, 12, 85],
]
sol_pos_3 = [
    [9, 35, 79],
    [9, 2, 85],
    [8, 18, 92],
    [7, 24, 88],
    [10, 10, 90],
]
sol_pos_4 = [
    [11, 32, 50],
    [9, 6, 60],
    [12, 14, 65],
    [10, 8, 70],
    [13, 11, 75],
    [14, 9, 80],
]
sol_pos_5 = [
    [10, 11, 45],
    [10, 14, 44],
    [11, 20, 52],
    [12, 18, 25],
    [14, 15, 71],
]
sol_pos_6 = [
    [60, 25, 16],
    [18, 2, 15],
    [19, 5, 20],
    [20, 7, 75],
    [21, 9, 42],
    [22, 12, 34],
]

#default joint weights
JOINT_WEIGHT_ARR = [1,1,1,1,1,1]

# A position is an array with 6 (rad) angles, corresponding to each joint

# A solution is composed of a suite of postition

# A x, y, z coordinate can be attained by multiple position

ur3e_arm = ur_kinematics.URKinematics("ur3e")

# Class to store a full trajectory
class TrajectoryClass:
    trajectory: list[list[float]] = []
    weight: float = 0.0

    def __init__(self, traj: list[list[float]], joint_weights = JOINT_WEIGHT_ARR):
        self.trajectory = traj
        self.joint_weights = joint_weights

    def __lt__(self, other):
         return self.weight < other.weight
    
    def __str__(self):
       return f"Trajectory: {str(self.trajectory)}, Weight: {str(self.weight)}"

    def eval_weight(self):
        self.weight = 0
        if len(self.trajectory) > 1:
            for i in range(len(self.trajectory)-1):
                for j in range(len(self.trajectory[i])):
                    self.weight += abs(self.trajectory[i][j] - self.trajectory[i+1][j]) * self.joint_weights[j]

    def add_point(self, pos: list[float]):
        return TrajectoryClass(self.trajectory + (pos))


# Function which lets us delete trajectories that are suboptimal
# If two path lead to the same joint state, the one with less cost is ALWAYS better (at that point)
def kill_traj(traj: list[TrajectoryClass]):
    best_trajs = {}
    
    for i in traj:
        i.eval_weight()
        last_position = tuple(i.trajectory[-1])  # Ensure it's hashable
        
        if last_position not in best_trajs or best_trajs[last_position].weight > i.weight:
            best_trajs[last_position] = i  # Keep the trajectory with the lowest weight
    
    return list(best_trajs.values())

def naive_search(nodes: np.ndarray[np.ndarray[float]]):
    results: list[TrajectoryClass]= []
    def helper(depth: int, results: list[TrajectoryClass]):
        if depth == 0:
            for node in nodes[depth]:
                newTraj = TrajectoryClass([node])
                newTraj.eval_weight()
                results.append(newTraj)
            return helper(depth+1, results)

        elif depth != len(nodes):
            res2 = []
            for node in nodes[depth]:
                for i in results:
                    newTraj: TrajectoryClass = i.add_point([node])
                    newTraj.eval_weight()
                    res2.append(newTraj)
            results = copy.deepcopy(res2)
            return helper(depth+1, results)

        else: return results
    ah = helper(0, results)
    ah.sort()
    return ah[0]

def best_first_search(nodes):
    trajectories = []
    # Compute all possibilites for node 0 to node 1
    for i in nodes[0]:
        for j in nodes[1]:
            trajectories.append(TrajectoryClass([i, j]))
    # Dont explore suboptimal paths
    trajectories = kill_traj(trajectories)
    trajectories.sort() # Sorting so best solution is always to the start
    while len(trajectories[0].trajectory) < len(nodes):
        for i in nodes[len(trajectories[0].trajectory)]:
            # Compute possibilities for next nodes, only on best path so far
            newTraj = copy.deepcopy(trajectories[0].trajectory)
            newTraj.append(i)
            trajectories.append(TrajectoryClass(newTraj))
        # Replace the best path with the new path
        trajectories.pop(0)
        trajectories = kill_traj(trajectories)
        trajectories.sort() # Sorting so best solution is always to the start
    return trajectories[0]

def benchmark():
    sols = []

    sols.append(sol_pos_1)
    sols.append(sol_pos_2)
    sols.append(sol_pos_3)
    sols.append(sol_pos_4)
    sols.append(sol_pos_5)
    sols.append(sol_pos_6)

    t = time.time()
    res = naive_search(sols)
    print(f"Time for naive search: {(time.time() - t)* 1000}ms")
    print(res)
    t = time.time()
    res = best_first_search(sols)
    print(f"Time for best first search: {(time.time() - t)* 1000}ms")
    print(res)


class Range:
    def __init__(self, min: float, max: float, weight: float):
        self.min = min
        self.max = max
        self.weight = self.weight


class JointsPreferedRange:
    def __init__(self, prefences: dict[int, Range]):
        for i in range(0, 6):
            if i not in prefences:
                prefences[i] = False
        self.preferences = prefences


def isInPref(position: np.ndarray[float], pref: dict[int, tuple[float]]):
    for jointId in pref:
        currPref = pref[jointId]
        currAngle = position[jointId]
        if currAngle < currPref[0] or currAngle > currPref[1]:
            return False
    return True


def filterSolutions(
    positions: np.ndarray[np.ndarray[float]], pref: dict[int, list[tuple]]
):
    accepted = []

    for p in positions:
        if isInPref(p, pref):
            accepted.append(p)

    return accepted


'''def getBestSolution(
    solutions: np.ndarray[np.ndarray[np.ndarray[float]]],
    preferedRange: JointsPreferedRange,
    jointWeight: dict[int, float],
):
    best_combination = []

    costs = {}

    for i in range(0, len(solutions) - 1):
        currStart = solutions[i]
        currEnd = solutions[i + 1]
        costs[(i, i + 1)] = {}
        pairsIndices = np.array(
            list(product(range(len(currStart)), range(len(currEnd))))
        )
        pairs = np.array([(currStart[i], currEnd[j]) for i, j in pairsIndices])
        angleCost = np.sum(np.abs(np.diff(pairs, axis=1)).squeeze(1), axis=1)
        for j in range(len(angleCost)):
            tup = tuple(pairsIndices[j].tolist())
            costs[(i, i + 1)][tup] = float(angleCost[j])

    
    import ipdb

    ipdb.set_trace()
    

    return best_combination


getBestSolution(
    [sol_pos_1, sol_pos_2, sol_pos_3], JointsPreferedRange({}), {0: 1.0, 6: 0.5}
)'''

benchmark()