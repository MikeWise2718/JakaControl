import math
import numpy as np
import os

from pxr import UsdPhysics

from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.stage import add_reference_to_stage,  get_current_stage

from omni.isaac.core.articulations import Articulation
from omni.isaac.core.objects.cuboid import DynamicCuboid
from omni.isaac.core.objects import GroundPlane
from omni.isaac.core.prims import XFormPrim
from omni.isaac.franka.controllers import PickPlaceController

from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.core.world import World

from omni.isaac.core.utils.extensions import get_extension_path_from_name

from omni.isaac.core.utils.numpy.rotations import euler_angles_to_quats

from omni.isaac.motion_generation import RmpFlow, ArticulationMotionPolicy

from .senut import add_light_to_stage, get_robot_params, get_robot_rmp_params
from .senut import ScenarioTemplate

# Copyright (c) 2022-2023, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#


class PickAndPlaceScenario(ScenarioTemplate):
    _running_scenario = False
    def __init__(self):
        pass

    def load_scenario(self, robot_name, ground_opt):

        self._robot_name = robot_name
        self._ground_opt = ground_opt

        add_light_to_stage()

       # print("Assets root path: ", get_assets_root_path())
        need_to_add_articulation = False
        self._robot_name = robot_name
        self._ground_opt = ground_opt
        (ok, robot_prim_path, artpath, path_to_robot_usd) = get_robot_params(self._robot_name)
        if not ok:
            print(f"Unknown robot name {self._robot_name}")
            return

        if path_to_robot_usd is not None:
            add_reference_to_stage(path_to_robot_usd, robot_prim_path)

        if need_to_add_articulation:
            prim = get_current_stage().GetPrimAtPath(artpath)
            UsdPhysics.ArticulationRootAPI.Apply(prim)

        if self._robot_name == "fancy_franka":
            from omni.isaac.franka import Franka
            self._fancy_robot = Franka(prim_path="/World/Fancy_Franka", name="fancy_franka")
            self._articulation = self._fancy_robot
        else:
            self._articulation = Articulation(artpath)


        # mode specific initialization
        self._cuboid = DynamicCuboid(
            "/Scenario/cuboid", position=np.array([0.3, 0.3, 0.15]), size=0.05, color=np.array([128, 0, 128])
        )


        # Add user-loaded objects to the World
        world = World.instance()
        if self._articulation is not None:
            world.scene.add(self._articulation)
        if self._cuboid is not None:
            world.scene.add(self._cuboid)

        if self._ground_opt == "default":
            world.scene.add_default_ground_plane()

        elif self._ground_opt == "groundplane":
            ground = GroundPlane(prim_path="/World/groundPlane", size=10, color=np.array([0.5, 0.5, 0.5]))
            world.scene.add(ground)

        elif self._ground_opt == "groundplane-blue":
            ground = GroundPlane(prim_path="/World/groundPlane", size=10, color=np.array([0.0, 0.0, 0.5]))
            world.scene.add(ground)

        self._object = self._cuboid
        self._fancy_cube = self._cuboid
        self._world = world
        print("load_scenario done - self._object", self._object)

    def post_load_scenario(self):
        self._franka = self._world.scene.get_object("fancy_franka")
        print("self._franka", self._franka)
        print("self._franka.gripper", self._franka.gripper)
        self._controller = PickPlaceController(
            name="pick_place_controller",
            gripper=self._franka.gripper,
            robot_articulation=self._franka,
        )
        print("gripper.joint_opened_positions",self._franka.gripper.joint_opened_positions)

        self._world.add_physics_callback("sim_step", callback_fn=self.physics_step)
        print("self._franka.gripper.set_joint_positions",self._franka.gripper.set_joint_positions)
        self._franka.gripper.set_joint_positions(self._franka.gripper.joint_opened_positions)

    def reset_scenario(self):
        self._franka = self._world.scene.get_object("fancy_franka")
        self._controller = PickPlaceController(
            name="pick_place_controller",
            gripper=self._franka.gripper,
            robot_articulation=self._franka,
        )
        self._franka.gripper.set_joint_positions(self._franka.gripper.joint_opened_positions)


    def physics_step(self, step_size):
        cube_position, _ = self._fancy_cube.get_world_pose()
        goal_position = np.array([-0.3, -0.3, 0.0515 / 2.0])
        current_joint_positions = self._franka.get_joint_positions()
        actions = self._controller.forward(
            picking_position=cube_position,
            placing_position=goal_position,
            current_joint_positions=current_joint_positions,
        )
        self._franka.apply_action(actions)
        # Only for the pick and place controller, indicating if the state
        # machine reached the final state.
        if self._controller.is_done():
            self._world.pause()
        return

    def setup_scenario(self):
        pass

    def teardown_scenario(self):
        pass

    def update_scenario(self, step: float):
        if not self._running_scenario:
            return
