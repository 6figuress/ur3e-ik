import pytest
import math
import numpy as np
from IKOn2DShape import compute_inverse_kinematics, back_up_point, interpolate_lines
from ur_ikfast import ur_kinematics

ur3e_arm = ur_kinematics.URKinematics("ur3e_pen")

@pytest.fixture
def ik_result():
	"""test fixture for the compute_inverse_kinematics function"""
	pos_raw = [0.2, 0.2, 0.5, 1, 0.1, 0.2, 0.3]
	previous_angles = [0, 0, 0, 0, 0, 0]
	result = compute_inverse_kinematics(pos_raw, previous_angles)
	return result, pos_raw

def test_compute_inverse_kinematics_result_type(ik_result):
	"""Test if the result of the compute_inverse_kinematics function is a numpy array of the right size"""
	result, _ = ik_result
	assert isinstance(result, np.ndarray), "The result should be a numpy array"
	assert result.shape == (6,), "The result should be a 6-element array"

def test_compute_inverse_kinematics_forward(ik_result):
	"""Test if the result is correct by comparing the input to the forward kinematics of the output"""
	result, pos_raw = ik_result
	pose_quat = ur3e_arm.forward(result)
	assert np.allclose(pose_quat[:3], pos_raw[:3], atol=1e-3), "The position should be the same"
	assert np.allclose(pose_quat[3:6], pos_raw[4:7], atol=5e-2), "The orientation (qx, qy, qz) should be the same"
	assert np.allclose(pose_quat[6], pos_raw[3], atol=1e-1), "The orientation (w) should be the same"

def test_interpolate_lines():
	"""test if the line interpolation function works as expected"""
	pose1 = np.array([0, 0, 0, 1, 0, 0, 0])
	pose2 = np.array([0, 0, 1, 1, 0, 0, 0])
	poses = interpolate_lines([pose1, pose2], 0.05)
	assert len(poses) == 21, "The result should be a list of 21 poses"
	for i in range(1, len(poses)):
		dist = math.sqrt(math.pow(poses[i][0] - poses[i-1][0], 2) + math.pow(poses[i][1] - poses[i-1][1], 2) + math.pow(poses[i][2] - poses[i-1][2], 2))
		assert dist < 0.5, "The poses should be less than 0.5 apart"
		
	assert np.allclose(poses[0], pose1), "The first pose should be the same as the first input"
	assert np.allclose(poses[-1], pose2), "The last pose should be the same as the second input"

def test_back_up_point():
	"""test the function back_up_point in all three directions (one at a time)"""
	pose = np.array([0, 0, 0, 1, 0, 0, 0])
	pose_back = back_up_point(pose, 0.1)
	assert np.allclose(pose_back, np.array([0, 0, 0.1, 1, 0, 0, 0])), f"[{pose_back}] The back up point should be 0.1 in the z direction"

	pose = np.array([0, 0, 0, 0.7073, -0.7073, 0, 0])
	pose_back = back_up_point(pose, 0.1)
	assert np.allclose(pose_back, np.array([0, 0.1, 0, 0.7073, -0.7073, 0, 0])), f"[{pose_back}] The back up point should be 0.1 in the y direction"

	pose = np.array([0, 0, 0, 1, 0, 1, 0])
	pose_back = back_up_point(pose, 0.1)
	assert np.allclose(pose_back, np.array([0.1, 0, 0, 1, 0, 1, 0])), f"[{pose_back}] The back up point should be 0.1 in the x direction"

	"""test the function back_up_point in all three directions (all at the same time)"""
	pose = np.array([0.0957, 0.35, 0.231, -0.2363, 0.3032, 0.8525, 0.3541])
	pose_back = back_up_point(pose, -0.1)
	assert np.allclose(pose_back, np.array([0.11451820100807172, 0.2752900680689588, 0.2947518735542651, -0.2363, 0.3032, 0.8525, 0.3541])), f"[{pose_back}] The back up point should be -0.1 away in the direction of the axis"

	pose = np.array([0.0829, 0.3978, 0.231, -0.4271, -0.1594, 0.8485, 0.2686])
	pose_back = back_up_point(pose, -0.1)
	assert np.allclose(pose_back, np.array([0.163948242432117, 0.3658320018887892, 0.2800839026099843, -0.4271, -0.1594, 0.8485, 0.2686])), f"[{pose_back}] The back up point should be -0.1 away in the direction of the axis"
	