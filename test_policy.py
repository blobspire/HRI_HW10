import pybullet as p
import pybullet_data
import numpy as np
import os
import time
from robot import Panda
from objects import objects
from models import Autoencoder
from teleop import KeyboardController
import torch

# parameters
control_dt = 1. / 240.
action_scale = 0.1
latent_scale = 500.0
test_position_gain = 0.1
panda_lower_limits = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
panda_upper_limits = np.array([+2.8973, +1.7628, +2.8973, -0.0698, +2.8973, +3.7525, +2.8973])

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
cylinder_position = np.random.uniform([0.6, -0.2, 0.1], [0.6, +0.2, 0.1])
cylinder = objects.SimpleObject("cylinder.urdf", basePosition=cylinder_position)

# load the robot
jointStartPositions = [0.0, 0.0, 0.0, -2*np.pi/4, 0.0, np.pi/2, np.pi/4, 0.0, 0.0, 0.04, 0.04]
panda = Panda(basePosition=[0, 0, 0],
                baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
                jointStartPositions=jointStartPositions)

# load the trained model
model = Autoencoder(state_dim=15, hidden_dim=256, action_dim=9, latent_dim=1)
model.load_state_dict(torch.load('model_weights_bidirectional_simple', map_location="cpu"))
model.eval()

# teleoperation interface
teleop = KeyboardController()

# main loop
gripper_state = [-1.0]
while True:
    # get the robot's position
    robot_state = panda.get_state()
    robot_pos = np.array(robot_state["joint-position"])

    # get user input
    teleop_action = teleop.get_action()

    # open or close the gripper
    if teleop_action[6] == +1:
        panda.open_gripper()
        gripper_state = [-1.0]
    elif teleop_action[6] == -1:
        panda.close_gripper()
        gripper_state = [+1.0]

    # exit current simulation
    if teleop_action[7]:
        break

    # get the full state
    state = robot_pos.tolist() + gripper_state + cylinder_position.tolist()

    # use the learned policy to output an action
    latent_action = np.clip(latent_scale * teleop_action[0], -1.0, 1.0)
    with torch.no_grad():
        action = action_scale * model.decoder(
            torch.FloatTensor([state]),
            torch.FloatTensor([[latent_action]])
        ).numpy()[0]

    # move the robot with action
    target_positions = robot_pos[0:9] + action
    target_positions[0:7] = np.clip(target_positions[0:7], panda_lower_limits, panda_upper_limits)
    target_positions[7:9] = robot_pos[7:9]
    panda.move_to_joint(target_positions, positionGain=test_position_gain)
    p.stepSimulation()
    time.sleep(control_dt)
