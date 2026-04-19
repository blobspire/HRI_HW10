import pybullet as p
import pybullet_data
import numpy as np
import os
import time
from robot import Panda
from objects import objects
import pickle
from tqdm import tqdm

# parameters
control_dt = 1. / 240.

# create simulation and place camera
physicsClient = p.connect(p.GUI)
p.setGravity(0, 0, -9.81)
# disable keyboard shortcuts so they do not interfere with keyboard control
p.configureDebugVisualizer(p.COV_ENABLE_KEYBOARD_SHORTCUTS, 0)
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
p.resetDebugVisualizerCamera(cameraDistance=1.0, 
                                cameraYaw=40.0,
                                cameraPitch=-40.0, 
                                cameraTargetPosition=[0.5, 0.0, 0.2])

# load the objects
urdfRootPath = pybullet_data.getDataPath()
plane = p.loadURDF(os.path.join(urdfRootPath, "plane.urdf"), basePosition=[0, 0, -0.625])
table = p.loadURDF(os.path.join(urdfRootPath, "table/table.urdf"), basePosition=[0.5, 0, -0.625])
cylinder = objects.SimpleObject("cylinder.urdf", basePosition=[0.6, 0, 0.1])

# load the robot
jointStartPositions = [0.0, 0.0, 0.0, -2*np.pi/4, 0.0, np.pi/2, np.pi/4, 0.0, 0.0, 0.04, 0.04]
panda = Panda(basePosition=[0, 0, 0],
                baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
                jointStartPositions=jointStartPositions)


# main loop
offset1 = np.array([0.0, 0.0, 0.2])      # reach above the cylinder
offset2 = np.array([0.0, 0.0, 0.0])      # grab the cylinder
offset3 = np.array([-0.3, 0.0, 0.1])     # bring it forward
offset4 = np.array([0.3, 0.0, 0.1])      # bring it back
offset = [offset1, offset2, offset3, offset4] # Robot will randomly choose 3 or 4
# offset = [offset1, offset2, offset3]
timesteps = [401, 201, 401, 401]
# timesteps = [401, 201, 401]
dataset = []
number_of_trajectories = 50 # Initial: 10
for idx in tqdm(range(number_of_trajectories)):

    # reset the robot
    panda.reset(jointStartPositions)
    panda.open_gripper()
    cylinder_position = np.random.uniform([0.6, -0.2, 0.1], [0.6, +0.2, 0.1])
    p.resetBasePositionAndOrientation(cylinder.object, cylinder_position, p.getQuaternionFromEuler([0, 0, 0]))
    gripper_state = [-1.0]

    # perform the expert behavior
    trajectory = []
    for stage in range(3):
        post_move = np.random.choice([2, 3]) # Decide to move end effector forward or back after picking up the cylinder
        for idx in range(1, timesteps[stage]):
            # close the gripper
            if stage == 2 and idx == 1:
                panda.close_gripper()
                gripper_state = [+1.0]
                for _ in range(100):
                    p.stepSimulation()
                    time.sleep(control_dt)
            # move to the goal position
            panda.move_to_pose(cylinder_position + offset[post_move if stage == 2 else stage], 
                                ee_quaternion=p.getQuaternionFromEuler([np.pi/2, np.pi/2, np.pi]), positionGain=0.01)
            p.stepSimulation()
            time.sleep(control_dt)
            if idx % 10 == 0:
                robot_state = panda.get_state()
                robot_pos = np.array(robot_state["joint-position"])
                trajectory.append(robot_pos.tolist() + gripper_state + cylinder_position.tolist()) # Robot = 11 dims, gripper = 1 dim, cylinder = 3 dims. Total = 15 dims.
    
    # calculate the actions
    trajectory = np.array(trajectory)
    forward_actions = trajectory[1:-1, 0:9] - trajectory[:-2, 0:9]
    reverse_actions = trajectory[1:-1, 0:9] - trajectory[2:, 0:9]
    trajectory_actions1 = np.concatenate((trajectory[1:-1,:], forward_actions), axis=1)
    trajectory_actions2 = np.concatenate((trajectory[1:-1,:], reverse_actions), axis=1)
    dataset += trajectory_actions1.tolist() + trajectory_actions2.tolist()

# save the dataset of demonstrations
pickle.dump(dataset, open("dataset_bidirectional.pkl", "wb"))
print("dataset has this many state-action pairs:", len(dataset))