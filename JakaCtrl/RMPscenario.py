import math
import numpy as np
import os

from pxr import UsdPhysics

from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.stage import add_reference_to_stage

from omni.isaac.core.articulations import Articulation
from omni.isaac.core.objects.cuboid import FixedCuboid
from omni.isaac.core.objects import GroundPlane
from omni.isaac.core.prims import XFormPrim
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


class RMPflowScenario(ScenarioTemplate):
    _running_scenario = False
    _show_collsion_bounds = False

    def __init__(self):
        pass

    def load_scenario(self, robot_name, ground_opt):

        self._robot_name = robot_name
        self._ground_opt = ground_opt

        add_light_to_stage()

        world = World.instance()
        if self._ground_opt == "default":
            world.scene.add_default_ground_plane()

        elif self._ground_opt == "groundplane":
            ground = GroundPlane(prim_path="/World/groundPlane", size=10, color=np.array([0.5, 0.5, 0.5]))
            world.scene.add(ground)

        elif self._ground_opt == "groundplane-blue":
            ground = GroundPlane(prim_path="/World/groundPlane", size=10, color=np.array([0.0, 0.0, 0.5]))
            world.scene.add(ground)


        # Setup Robot ARm
        (ok, robot_prim_path, artpath, path_to_robot_usd) = get_robot_params(self._robot_name)
        if not ok:
            print(f"Unknown robot name {self._robot_name}")
            return

        if path_to_robot_usd is not None:
            add_reference_to_stage(path_to_robot_usd, robot_prim_path)

        self._articulation = Articulation(artpath)

        if self._articulation is not None:
            world.scene.add(self._articulation)


        # add a cube for franka to pick up
        # world.scene.add(
        #     DynamicCuboid(
        #         prim_path="/World/random_cube",
        #         name="fancy_cube",
        #         position=np.array([0.3, 0.3, 0.3]),
        #         scale=np.array([0.0515, 0.0515, 0.0515]),
        #         color=np.array([0, 0, 1.0]),
        #     )
        # )

        add_reference_to_stage(get_assets_root_path() + "/Isaac/Props/UIElements/frame_prim.usd", "/World/target")
        self._target = XFormPrim("/World/target", scale=[.04,.04,.04])
        self._object =  self._target

        self._world = world

    def setup_scenario(self):
        print("RMPflow setup_scenario")

        self._initial_object_position = self._object.get_world_pose()[0]
        self._initial_object_phase = np.arctan2(self._initial_object_position[1], self._initial_object_position[0])
        self._object_radius = np.linalg.norm(self._initial_object_position[:2])

        self._running_scenario = True

        self._joint_index = 0
        self._lower_joint_limits = self._articulation.dof_properties["lower"]
        self._upper_joint_limits = self._articulation.dof_properties["upper"]
        self._zeros = np.zeros(len(self._lower_joint_limits))
        self._njoints = len(self._lower_joint_limits)
        print(f"jaka - njoints:{self._njoints} lower:{self._lower_joint_limits} upper:{self._upper_joint_limits}")

        # teleport robot to lower joint range
        epsilon = 0.001
        # articulation.set_joint_positions(self._lower_joint_limits + epsilon)
        self._articulation.set_joint_positions(self._zeros + epsilon)

        self._obstacle = FixedCuboid("/World/obstacle",size=.05,position=np.array([0.4, 0.0, 0.65]),color=np.array([0.,0.,1.]))


        # RMPflow config files for supported robots are stored in the motion_generation extension under "/motion_policy_configs"
        mg_extension_path = get_extension_path_from_name("omni.isaac.motion_generation")
        rmp_config_dir = os.path.join(mg_extension_path, "motion_policy_configs")
        print("rmp_config_dir",rmp_config_dir)

        (ok, rdf_path, urdf_path, rmp_config_path, eeframe_name, max_step_size) = get_robot_rmp_params(self._robot_name)
        print("rdf_path:",rdf_path)
        print("urdf_path:",urdf_path)
        print("rmp_config_path:",rmp_config_path)
        print("eeframe_name:",eeframe_name)
        print("max_step_size:",max_step_size)
        # Initialize an RmpFlow object
        self._rmpflow = RmpFlow(
            robot_description_path = rdf_path,
            urdf_path = urdf_path,
            rmpflow_config_path = rmp_config_path,
            end_effector_frame_name = eeframe_name,
            maximum_substep_size = max_step_size
        )
        self._rmpflow.add_obstacle(self._obstacle)

        if self._show_collsion_bounds:
            self._rmpflow.set_ignore_state_updates(True)
            self._rmpflow.visualize_collision_spheres()

            # Set the robot gains to be deliberately poor
            bad_proportional_gains = self._articulation.get_articulation_controller().get_gains()[0]/50
            self._articulation.get_articulation_controller().set_gains(kps = bad_proportional_gains)

        print("Created _rmpflow object")

        self._articulation_rmpflow = ArticulationMotionPolicy(self._articulation,self._rmpflow)

        self._target.set_world_pose(np.array([.5,0,.7]),euler_angles_to_quats([0,np.pi,0]))


    def post_load_scenario(self):
        self._rmpflow.add_obstacle(self._obstacle)

        self._world.add_physics_callback("sim_step", callback_fn=self.physics_step)
        pass

    def reset_scenario(self):
        self._target.set_world_pose(np.array([.5,0,.7]),euler_angles_to_quats([0,np.pi,0]))
        if self._show_collsion_bounds:
            self._rmpflow.reset()
            self._rmpflow.visualize_collision_spheres()




    def physics_step(self, step_size):
        target_position, target_orientation = self._target.get_world_pose()

        self._rmpflow.update_world()


        self._rmpflow.set_end_effector_target(
            target_position, target_orientation
        )

        action = self._articulation_rmpflow.get_next_articulation_action(step_size)
        self._articulation.apply_action(action)

    def teardown_scenario(self):
        pass

    def update_scenario(self, step: float):
        if not self._running_scenario:
            return
