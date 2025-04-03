import numpy as np
import pytest

# Assuming JOINT_WEIGHT_ARR is defined somewhere in your code
JOINT_WEIGHT_ARR = [1, 1, 1, 1, 1, 1]  

from .best_trajectory import Trajectory, TrajectoryPlanner  # Replace 'your_module' with the actual module name


def test_best_first_search():
    planner = TrajectoryPlanner(JOINT_WEIGHT_ARR)
    nodes = [
        [[0, 0, 0, 0, 0, 0]],
        [[1, 1, 1, 1, 1, 1], [2, -1, 2, 2, 2, 2], [-1, -1, -1, 3, 3, 3], [4, 4, 4, 4, 4, 4]],
        [[3, 3, 3, 3, 3, 3]]
    ]
    
    best_traj = planner.best_first_search(nodes)

    assert isinstance(best_traj, Trajectory)
    assert len(best_traj.trajectory) == 3  # Should include all steps
    assert np.array_equal(best_traj.trajectory[-1], [3, 3, 3, 3, 3, 3])
    assert np.array_equal(best_traj.trajectory[1], [1,1,1,1,1,1])

if __name__ == "__main__":
    pytest.main()
