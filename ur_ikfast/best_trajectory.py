import copy

JOINT_WEIGHT_ARR = [10, 5, 2, 1, 1, 0]

class Trajectory:
    def __init__(self, traj: list[list[float]], joint_weights=JOINT_WEIGHT_ARR):
        """
        Class to store a trajectory and compute its weight
        traj: A list of joint positions (each a list of 6 floats)
        joint_weights: A list of weights for each joint (default: all ones)
        """
        self.trajectory = traj
        self.joint_weights = joint_weights
        self.weight = 0.0
        self.eval_weight()

    def __lt__(self, other):
        """ 
        This is useful for sorting the trajectories by weight
        """
        return self.weight < other.weight

    def __str__(self):
        """ Return a string representation of the trajectory and its weight """
        return f"Trajectory: {str(self.trajectory)}, Weight: {str(self.weight)}"
    
    def __iter__(self):
        """ Return an iterator for the trajectory """
        return iter(self.trajectory)

    def eval_weight(self):
        """ Computes the weight of the trajectory """
        self.weight = 0
        if len(self.trajectory) > 1:
            for i in range(len(self.trajectory) - 1):
                for j in range(len(self.trajectory[i])):
                    self.weight += abs(self.trajectory[i][j] - self.trajectory[i+1][j]) * self.joint_weights[j]

    def add_point(self, pos: list[float]):
        """ Add a point to the trajectory and return a new trajectory object """
        new_trajectory = self.trajectory + [pos]
        return Trajectory(new_trajectory, self.joint_weights)

class TrajectoryPlanner:
    def __init__(self, joint_weights=JOINT_WEIGHT_ARR):
        """ Initialize the trajectory planner with optional joint weights """
        self.joint_weights = joint_weights

    def kill_traj(self, traj: list[Trajectory]):
        """
        Remove suboptimal trajectories (i.e., if two paths lead to the same joint state, 
        the one with less cost is always better)
        """
        best_trajs = {}
        for i in traj:
            last_position = tuple(i.trajectory[-1])
            if last_position not in best_trajs or best_trajs[last_position].weight > i.weight:
                best_trajs[last_position] = i  # Keep the trajectory with the lowest weight
        return list(best_trajs.values())

    def best_first_search(self, nodes: list[list[list[float]]]):
        """
        Perform a best-first search to find the optimal trajectory
        nodes: List of possible joint states at each step
        """
        trajectories = []

        # If there are no nodes, return an empty trajectory
        if len(nodes) == 0:
            return Trajectory([], self.joint_weights)
        
        # Compute all possibilities for node 0 to node 1
        for i in nodes[0]:
            for j in nodes[1]:
                trajectories.append(Trajectory([i, j], self.joint_weights))

        # Don't explore suboptimal paths
        trajectories = self.kill_traj(trajectories)
        trajectories.sort()  # Sorting so best solution is always at the start
        
        while len(trajectories[0].trajectory) < len(nodes):
            for i in nodes[len(trajectories[0].trajectory)]:
                # Compute possibilities for next nodes, only on the best path so far
                newTraj = copy.deepcopy(trajectories[0].trajectory)
                newTraj.append(i)
                trajectories.append(Trajectory(newTraj, self.joint_weights))

            # Replace the best path with the new path
            trajectories.pop(0)
            trajectories = self.kill_traj(trajectories)
            trajectories.sort()  # Sorting so best solution is always at the start
        
        return trajectories[0]

def main():
    joints = [
        [[0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
         [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]],
        [[0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
         [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]]
    ]

    planner = TrajectoryPlanner()

    # Perform a best-first search to find the optimal trajectory
    best_trajectory = planner.best_first_search(joints)

    print(f"Best Trajectory: {best_trajectory}")

if __name__ == "__main__":
    main()
