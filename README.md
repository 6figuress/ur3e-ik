# IKFast for Universal Robots

IKFast libraries for UR3, UR5, and UR10 both the CB and e-series.

Inspired by https://github.com/andyzeng/ikfastpy

## Forward/Inverse Kinematics
This library computes the forward and inverse for Universal Robots from the `base_link` frame to the `tool0` frame. 
This only applies to the ideal robot model defined by their URDF. The real robots have small differences in each joint that are not considered here, so expect to see position/orientation errors from submillimeter errors to a few centimeters.

## Installation
Easy to install and use:
1. install dependencies
```shell
sudo apt-get install libblas-dev liblapack-dev
pip install --user numpy Cython
```

3. install using pip </br>

```shell
git clone https://github.com/cambel/ur_ikfast.git
cd ur_ikfast
pip install -e .
```

or directly from the git url :

```shell
pip install git+https://github.com/6figuress/ur3e-ik.git
```


It takes a few minutes to compile the IKfast libraries.

## Example
```python
from ur_ikfast import ur_kinematics

ur3e_arm = ur_kinematics.URKinematics('ur3e')

joint_angles = [-3.1, -1.6, 1.6, -1.6, -1.6, 0.]  # in radians
print("joint angles", joint_angles)

pose_quat = ur3e_arm.forward(joint_angles)
pose_matrix = ur3e_arm.forward(joint_angles, 'matrix')

print("forward() quaternion \n", pose_quat)
print("forward() matrix \n", pose_matrix)

# print("inverse() all", ur3e_arm.inverse(pose_quat, True))
print("inverse() one from quat", ur3e_arm.inverse(pose_quat, False, q_guess=joint_angles))

print("inverse() one from matrix", ur3e_arm.inverse(pose_matrix, False, q_guess=joint_angles))
```

## To create the files for a new robot, you can follow these steps (or ask Zeb)

### Install ros:
sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'

sudo apt install curl\
curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | sudo apt-key add -

sudo apt update\
sudo apt install ros-noetic-desktop-full

sudo rosdep init\
rosdep update

echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc\
source ~/.bashrc

rosversion -d \
(should return "noetic")


### Get a docker image for openrave (don't try to install it in local trust me)
docker pull hamzamerzic/openrave


### File creation pipeline
1. Modifiy or create a .urdf file
2. Run the command in local: rosrun collada_urdf urdf_to_collada [robot_name].urdf [robot_name].dae \
Ex. rosrun collada_urdf urdf_to_collada ur3e.urdf ur3e.dae
3. Make an xml wrapper called [robot_name].wrapper.xml (example in the ur3e directory)
4. Pass these files to a running instance of the docker image (manually or with a volume in a docker compose)
5. Run the command in the docker: openrave.py --database inversekinematics --robot=[wrapper_name].xml --iktype=transform6d --iktests=100
6. Copy and modify an existing .pyx file for cython (there should be one per robot directory
7. Add your robot to the setup.py in the same format as the others
8. Run the command in local in the ur3e-ik directory: python setup.py build_ext --inplace
9. Run the command in local: pip install -e .
10. Add your robot to the init in ur_ikfast/ur_kinematics.py at line 45
11. You should be able to call this: new_arm = ur_kinematics.URKinematics('[robot_name]')
