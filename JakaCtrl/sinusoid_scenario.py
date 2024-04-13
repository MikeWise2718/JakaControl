import math
import numpy as np
import os

from pxr import UsdPhysics

from omni.isaac.core.utils.stage import add_reference_to_stage,  get_current_stage

from omni.isaac.core.articulations import Articulation
from omni.isaac.core.objects.cuboid import FixedCuboid
from omni.isaac.core.objects import GroundPlane

from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.core.world import World

from .senut import add_sphere_light_to_stage
from .scenario_base import ScenarioBase

# Copyright (c) 2022-2023, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

"""
This scenario takes in a robot Articulation and makes it move through its joint DOFs.
Additionally, it adds a cuboid prim to the stage that moves in a circle around the robot.

The particular framework under which this scenario operates should not be taken as a direct
recomendation to the user about how to structure their code.  In the simple example put together
in this template, this particular structure served to improve code readability and separate
the logic that runs the example from the UI design.
"""

class SinusoidJointScenario(ScenarioBase):
    def __init__(self):
        super().__init__()
        self._scenario_name = "sinusoid-joint"
        self._scenario_description = ScenarioBase.get_scenario_desc(self._scenario_name)

        self._running_scenario = False

        self._time = 0.0  # s

        self._object_radius = 0.5  # m
        self._object_height = 0.5  # m
        self._object_frequency = 0.25  # Hz

        self._joint_index = 0
        self._max_joint_speed = 4  # rad/sec
        self._nrobots = 1

        self._joint_time = 0
        self._path_duration = 0
        self._calculate_position = lambda t, x: 0
        self._calculate_velocity = lambda t, x: 0

    def load_scenario(self, robot_name, ground_opt):
        super().load_scenario(robot_name, ground_opt)

        self.add_light("sphere_light")
        self.add_ground(ground_opt)

        self.create_robot_config(robot_name,"/World/roborg", ground_opt)
        self.load_robot_into_scene()

        # mode specific initialization
        self._cuboid = FixedCuboid(
            "/Scenario/cuboid", position=np.array([0.3, 0.3, 0.15]), size=0.05, color=np.array([128, 0, 128])
        )

        # Add user-loaded objects to the World
        world = World.instance()
        world.scene.add(self._cuboid)

        self._object = self._cuboid
        print("load_scenario done - self._object", self._object)

    def post_load_scenario(self):
        print("setup_scenario - self._object", self._object)

        self._initial_object_position = self._object.get_world_pose()[0]
        self._initial_object_phase = np.arctan2(self._initial_object_position[1], self._initial_object_position[0])
        self._object_radius = np.linalg.norm(self._initial_object_position[:2])

        self._running_scenario = True

        self._joint_index = 0

        self.register_robot_articulations()
        self.teleport_robots_to_zeropos()

        self._derive_sinusoid_params(0)

    def teardown_scenario(self):
        self._time = 0.0
        self._running_scenario = False

        self._joint_index = 0

        self._joint_time = 0
        self._path_duration = 0
        self._calculate_position = lambda t, x: 0
        self._calculate_velocity = lambda t, x: 0

    def update_scenario(self, step: float):
        if not self._running_scenario:
            return

        self._time += step

        x = self._object_radius * np.cos(self._initial_object_phase + self._time * self._object_frequency * 2 * np.pi)
        y = self._object_radius * np.sin(self._initial_object_phase + self._time * self._object_frequency * 2 * np.pi)
        z = self._initial_object_position[2]

        self._object.set_world_pose(np.array([x, y, z]))

        self._update_sinusoidal_joint_path(step)

    def _derive_sinusoid_params(self, joint_index: int):
        # Derive the parameters of the joint target sinusoids for joint {joint_index}
        rcfg = self.get_robot_config(0)
        start_position = rcfg.lower_dof_lim[joint_index]
        start_position = 0
        llim = rcfg.lower_dof_lim[joint_index]
        ulim = rcfg.upper_dof_lim[joint_index]
        mjs = self._max_joint_speed

        print(f"jaka - jidx:{joint_index} start_position:{start_position:.3f} llim:{llim:.3f} ulim:{ulim:.3f}")

        P_max = rcfg.upper_dof_lim[joint_index] - start_position
        V_max = self._max_joint_speed
        T = P_max * np.pi / V_max
        print(f"jaka - P_max:{P_max:.3f} V_max:{V_max:.3f} path_duration (T):{T:.3f}")

        # T is the expected time of the joint path
        self._path_duration = T
        self._path_duration = 10
        self._calculate_position = (
            lambda time, path_duration: start_position
                                        - (P_max / 2 * np.cos(time * 2 * np.pi / path_duration))
                                        + (P_max / 2)
        )
        self.lastprint_time = 0
        self._calculate_velocity = lambda time, path_duration: V_max * np.sin(2 * np.pi * time / path_duration)

    def _update_sinusoidal_joint_path(self, step):
        # Update the target for the robot joints
        self._joint_time += step
        rcfg = self.get_robot_config(0)

        if self._joint_time > self._path_duration:
            self._joint_time = 0
            self._joint_index = (self._joint_index + 1) % rcfg._articulation.num_dof
            print(f"Changing to joint {self._joint_index} at time {self._time:.3f}")
            self._derive_sinusoid_params(self._joint_index)

        joint_position_target = self._calculate_position(self._joint_time, self._path_duration)
        joint_velocity_target = self._calculate_velocity(self._joint_time, self._path_duration)

        if self._joint_time - self.lastprint_time > 1:
           self.lastprint_time = self._joint_time
           print(f"jaka - idx:{self._joint_index} joint time: {self._joint_time:.3f} path duration: {self._path_duration:.3f}")
           print(f"jaka - joint_position_target:{joint_position_target:.3f} joint_velocity_target:{joint_velocity_target:.3f}")

        action = ArticulationAction(
            np.array([joint_position_target]),
            np.array([joint_velocity_target]),
            joint_indices=np.array([self._joint_index])
        )
        rcfg._articulation.apply_action(action)
