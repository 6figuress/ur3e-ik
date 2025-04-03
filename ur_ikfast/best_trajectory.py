JOINT_WEIGHT_ARR = [10, 5, 2, 1, 1, 0]

import copy
import heapq
import numpy as np
import datetime


import numpy as np
import heapq
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass, field


# Assuming JOINT_WEIGHT_ARR is defined elsewhere
# If not, we'll define a default
import numpy as np
import heapq
from typing import List, Tuple

# Assuming DEFAULT_JOINT_WEIGHTS is defined elsewhere in your code
DEFAULT_JOINT_WEIGHTS = np.ones(
    3, dtype=np.float64
)  # Default example, adjust as needed


class Trajectory:
    """Class to store a trajectory and compute its weight using NumPy arrays"""

    def __init__(self, traj: np.ndarray, joint_weights: np.ndarray = None):
        """
        Initialize a trajectory and compute its weight

        Parameters:
        traj: np.ndarray - A 2D array of shape (n_points, n_joints)
        joint_weights: np.ndarray - A 1D array of shape (n_joints,)
        """
        # Ensure trajectory is a numpy array
        self.trajectory = np.asarray(traj, dtype=np.float64)

        # Handle empty trajectories
        if self.trajectory.size == 0:
            if isinstance(traj, list) and len(traj) == 0:
                self.trajectory = np.empty(
                    (0, len(joint_weights) if joint_weights is not None else 0)
                )
            else:
                self.trajectory = np.empty((0, 0))

        # Ensure proper shape for single point trajectories
        if len(self.trajectory.shape) == 1:
            self.trajectory = self.trajectory.reshape(1, -1)

        # Set joint weights
        if joint_weights is None:
            if self.trajectory.shape[1] > 0:
                self.joint_weights = np.ones(self.trajectory.shape[1], dtype=np.float64)
            else:
                self.joint_weights = DEFAULT_JOINT_WEIGHTS
        else:
            self.joint_weights = np.asarray(joint_weights, dtype=np.float64)

        self.weight = 0.0
        self.eval_weight()

    def __lt__(self, other):
        """For sorting trajectories by weight"""
        return self.weight < other.weight

    def __str__(self):
        """String representation of the trajectory and its weight"""
        return f"Trajectory: {self.trajectory}, Weight: {self.weight}"

    def __iter__(self):
        """Iterator for the trajectory"""
        return iter(self.trajectory)

    def eval_weight(self):
        """Compute trajectory weight using vectorized operations"""
        if len(self.trajectory) > 1:
            # Calculate differences between consecutive points
            diffs = np.diff(self.trajectory, axis=0)
            # Apply weights to the absolute differences and sum
            self.weight = np.sum(np.abs(diffs) * self.joint_weights)

    def add_point(self, pos: np.ndarray) -> "Trajectory":
        """
        Add a point to the trajectory and return a new trajectory object

        Parameters:
        pos: np.ndarray - A 1D array of joint positions

        Returns:
        Trajectory - A new trajectory object with the added point
        """
        pos_array = np.asarray(pos, dtype=np.float64).reshape(1, -1)
        new_trajectory = (
            np.vstack((self.trajectory, pos_array))
            if self.trajectory.size > 0
            else pos_array
        )

        # Create new trajectory
        result = Trajectory(new_trajectory, self.joint_weights)

        # Optimize: incrementally update weight instead of recomputing
        if len(self.trajectory) > 0:
            last_point = self.trajectory[-1]
            increment = np.sum(np.abs(pos_array - last_point) * self.joint_weights)
            result.weight = self.weight + increment

        return result

    def get_last_position_tuple(self) -> Tuple[float, ...]:
        """Return the last position as a hashable tuple for dictionary keys"""
        if self.trajectory.size > 0:
            return tuple(self.trajectory[-1])
        return tuple()


class TrajectoryPlanner:
    def __init__(self, joint_weights: np.ndarray = None):
        """
        Initialize the trajectory planner with optional joint weights

        Parameters:
        joint_weights: np.ndarray - A 1D array of weights for each joint
        """
        self.joint_weights = np.asarray(
            joint_weights if joint_weights is not None else DEFAULT_JOINT_WEIGHTS,
            dtype=np.float64,
        )

    def kill_traj(self, trajectories: List[Trajectory]) -> List[Trajectory]:
        """
        Remove suboptimal trajectories using a dictionary lookup

        Parameters:
        trajectories: List[Trajectory] - List of trajectory objects to filter

        Returns:
        List[Trajectory] - List of optimal trajectories
        """
        best_trajs = {}

        for traj in trajectories:
            last_position = traj.get_last_position_tuple()
            if (
                last_position not in best_trajs
                or best_trajs[last_position].weight > traj.weight
            ):
                best_trajs[last_position] = traj

        return list(best_trajs.values())

    def best_first_search(self, nodes) -> Trajectory:
        """
        Perform a best-first search using a priority queue for efficiency.
        Modified to handle variable number of options per layer.

        Parameters:
        nodes: List[List[np.ndarray]] or similar structure where:
              - nodes[i] is the list of options for layer i
              - each option is a numpy array of joint positions

        Returns:
        Trajectory - The optimal trajectory
        """
        # Check if nodes is empty
        if not nodes or len(nodes) == 0:
            return Trajectory(
                np.empty((0, len(self.joint_weights))), self.joint_weights
            )

        # Handle case with only one layer
        if len(nodes) == 1:
            if len(nodes[0]) > 0:
                return Trajectory(nodes[0][0].reshape(1, -1), self.joint_weights)
            else:
                return Trajectory(
                    np.empty((0, len(self.joint_weights))), self.joint_weights
                )

        # Convert input to a consistent format (list of lists of numpy arrays)
        processed_nodes = []
        for layer in nodes:
            if isinstance(layer, np.ndarray):
                # Handle numpy arrays: could be 2D (set of options) or 3D (layer of options)
                if len(layer.shape) == 3:  # [layer_idx, option_idx, joint_idx]
                    processed_nodes.append([option for option in layer])
                elif len(layer.shape) == 2:  # [option_idx, joint_idx]
                    processed_nodes.append([option for option in layer])
                else:  # Single option [joint_idx]
                    processed_nodes.append([layer])
            else:  # Already a list of options
                processed_nodes.append(layer)

        # Priority queue for best-first search
        queue = []

        # Initialize with trajectories from first to second node
        for start_point in processed_nodes[0]:
            start_point = np.asarray(start_point, dtype=np.float64)

            # Handle different dimensions for start points
            if len(start_point.shape) == 0:  # scalar
                start_point = np.array([start_point])
            elif len(start_point.shape) == 2:  # already 2D
                start_point = start_point[0]  # Take first option if multiple

            for next_point in processed_nodes[1]:
                next_point = np.asarray(next_point, dtype=np.float64)

                # Handle different dimensions for next points
                if len(next_point.shape) == 0:  # scalar
                    next_point = np.array([next_point])
                elif len(next_point.shape) == 2:  # already 2D
                    next_point = next_point[0]  # Take first option if multiple

                # Create a trajectory with two points
                traj_points = np.vstack(
                    (start_point.reshape(1, -1), next_point.reshape(1, -1))
                )
                traj = Trajectory(traj_points, self.joint_weights)

                # Use a tuple of (weight, unique_id, trajectory) for the heap
                heapq.heappush(queue, (traj.weight, id(traj), traj))

        # Track best paths to each state
        best_paths = {}

        while queue:
            _, _, current_traj = heapq.heappop(queue)
            current_layer = len(current_traj.trajectory) - 1

            # If we've reached the final layer, we're done
            if current_layer == len(processed_nodes) - 1:
                return current_traj

            # Skip if we already have a better path to this state
            last_pos_tuple = current_traj.get_last_position_tuple()
            if (
                last_pos_tuple in best_paths
                and best_paths[last_pos_tuple].weight < current_traj.weight
            ):
                continue

            # Mark this as the best path to this state
            best_paths[last_pos_tuple] = current_traj

            # Expand to next layer
            next_layer = current_layer + 1
            if next_layer < len(processed_nodes):
                for next_point in processed_nodes[next_layer]:
                    next_point = np.asarray(next_point, dtype=np.float64)

                    # Handle different dimensions
                    if len(next_point.shape) == 0:  # scalar
                        next_point = np.array([next_point])
                    elif len(next_point.shape) == 2:  # already 2D
                        next_point = next_point[0]  # Take first option if multiple

                    new_traj = current_traj.add_point(next_point)
                    next_pos_tuple = new_traj.get_last_position_tuple()

                    # Only add if it's better than what we've seen
                    if (
                        next_pos_tuple not in best_paths
                        or best_paths[next_pos_tuple].weight > new_traj.weight
                    ):
                        heapq.heappush(queue, (new_traj.weight, id(new_traj), new_traj))

        # If queue is empty and we haven't returned, return the best path we've found
        if best_paths:
            return min(best_paths.values(), key=lambda t: t.weight)
        return Trajectory(np.empty((0, len(self.joint_weights))), self.joint_weights)


if __name__ == "__main__":
    import numpy as np

    # Define the shape
    shape = (5000, 5, 6)

    # Generate random numbers between -pi and pi
    random_numbers = np.random.uniform(-np.pi, np.pi, shape)

    # Display the shape of the generated array to confirm
    random_numbers.shape

    toCompute = random_numbers.tolist()

    planner = TrajectoryPlanner(JOINT_WEIGHT_ARR)

    then = datetime.datetime.now()

    print("Starting")

    best = planner.best_first_search(toCompute)

    now = datetime.datetime.now()

    print("Took : ", now - then)

    import ipdb

    ipdb.set_trace()