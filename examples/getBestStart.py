import numpy as np
from ur_ikfast import ur_kinematics
from itertools import combinations, product
import copy


sol_pos_1 = [
    [1, 50, 100],
    [1, 75, 100],
]
sol_pos_2 = [
    [4, 25, 90],
    [4, 0, 90],
]
sol_pos_3 = [
    [5, 30, 89],
    [5, 1, 90],
]
sol_pos_4 = [
    [5, 30, 89],
    [5, 1, 90],
]
sol_pos_5 = [
    [5, 30, 89],
    [5, 1, 90],
]


# A position is an array with 6 (rad) angles, corresponding to each joint

# A solution is composed of a suite of postition

# A x, y, z coordinate can be attained by multiple position

ur3e_arm = ur_kinematics.URKinematics("ur3e")

# Class to store a full trajectory
class TrajAgainstTheMachine:
    trajectory: list[list[float]] = []
    weight: float = 0.0

    def __init__(self, traj: list[list[float]]):
        self.trajectory = traj

    def __lt__(self, other):
         return self.weight < other.weight

    def eval_weight(self):
        self.weight = 0
        for i in self.trajectory:
            self.weight += i[2]

    def add_point(self, pos: list[float]):
        return TrajAgainstTheMachine(self.trajectory + (pos))


def naive_search(nodes: np.ndarray[np.ndarray[float]]):
    results: list[TrajAgainstTheMachine]= []
    def helper(depth: int, results: list[TrajAgainstTheMachine]):
        if depth == 0:
            for node in nodes[depth]:
                newTraj = TrajAgainstTheMachine([node])

                newTraj.eval_weight()
                results.append(newTraj)

            return helper(depth+1, results)

        elif depth != len(nodes):
            res2 = []
            for node in nodes[depth]:
                for i in results:
                    newTraj: TrajAgainstTheMachine = i.add_point([node])
                    newTraj.eval_weight()
                    res2.append(newTraj)
            results = copy.deepcopy(res2)
            return helper(depth+1, results)

        else: return results
    ah = helper(0, results)
    ah.sort()


def best_first_search(nodes):
    pass
    

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


def getBestSolution(
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

    '''
    import ipdb

    ipdb.set_trace()
    '''

    return best_combination


'''getBestSolution(
    [sol_pos_1, sol_pos_2, sol_pos_3], JointsPreferedRange({}), {0: 1.0, 6: 0.5}
)'''

sols = []

for i in range(10):
    sols.append([[1,2,1],[3,1,1],[3,1,2]])
print(sols)
naive_search(sols)
