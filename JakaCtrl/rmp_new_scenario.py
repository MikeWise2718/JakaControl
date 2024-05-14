import numpy as np
import carb

from pxr import UsdPhysics, Usd, UsdGeom, Gf

from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.stage import add_reference_to_stage

from omni.isaac.core.objects.cuboid import FixedCuboid
from omni.isaac.core.prims import XFormPrim
from omni.isaac.core.world import World

from omni.isaac.core.utils.numpy.rotations import euler_angles_to_quats, rot_matrices_to_quats

from omni.isaac.motion_generation import RmpFlow, ArticulationMotionPolicy
from omni.isaac.motion_generation import ArticulationKinematicsSolver

from .scenario_base import ScenarioBase

from omni.isaac.core.utils.stage import add_reference_to_stage

# Copyright (c) 2022-2023, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#


class RMPflowNewScenario(ScenarioBase):
    _running_scenario = False
    _colorScheme = ""
    _enable_obstacle = False

    def __init__(self, uibuilder=None):
        super().__init__()
        self._scenario_name = "rmpflow-new"
        self._scenario_desc = ScenarioBase.get_scenario_desc(self._scenario_name)
        self._nrobots = 1
        self.uibuilder = uibuilder

    def load_scenario(self, robot_name, ground_opt):
        # Here we do object loading and simple initialization
        super().load_scenario(robot_name, ground_opt)

        self.create_robot_config(robot_name, "/World/roborg", ground_opt)

        self.add_light("sphere_light")
        self.add_ground(ground_opt)

        # self._robcfg = self.create_robot_config(robot_name, ground_opt)

        self._robot_name = robot_name
        self._ground_opt = ground_opt

        # self._target_start_pos = np.array([0.5, 0.0, 0.7])
        self._target_start_pos = np.array([0.4, 0.0, 0.6])
        self._target_start_rot = euler_angles_to_quats([0, np.pi, 0])
        self._obstacle_start_pos = np.array([0.4, 0.0, 0.65])
        self._obstacle_start_rot = euler_angles_to_quats([0, np.pi, 0])

        pos0 = Gf.Vec3d([0, 0, 1.1])
        rot0 = [180, 0, 0]
        if self._robot_name == "ur10-suction-short":
            pos0 = Gf.Vec3d([0, 0, 0.4])
            rot0 = [180, 0, 0]

        self.load_robot_into_scene(0, pos0, rot0)

        # Add a target to the stage
        add_reference_to_stage(get_assets_root_path() + "/Isaac/Props/UIElements/frame_prim.usd", "/World/target")
        sz = 0.04
        self._target = XFormPrim("/World/target", scale=[sz, sz, sz])

        # self._world = world

        if self._enable_obstacle:
            self._obstacle = FixedCuboid("/World/obstacle", size=.05, color=np.array([0.,0.,1.]))
            self._rmpflow.add_obstacle(self._obstacle)

    def post_load_scenario(self):
        print("post_load_scenario")
        # Here we do multi-object initialization - things that needs to be done after all objects are loaded

        self.register_robot_articulations()
        self.teleport_robots_to_zeropos()

        # Initialize an RmpFlow object
        rcfg = self.get_robot_config()
        self._rmpflow = RmpFlow(
            robot_description_path=rcfg.rdf_path,
            urdf_path=rcfg.urdf_path,
            rmpflow_config_path=rcfg.rmp_config_path,
            end_effector_frame_name=rcfg.eeframe_name,
            maximum_substep_size=rcfg.max_step_size
        )

        self._articulation_rmpflow = ArticulationMotionPolicy(rcfg._articulation, self._rmpflow)
        self._kinematics_solver = self._rmpflow.get_kinematics_solver()

        self._articulation_kinematics_solver = ArticulationKinematicsSolver(rcfg._articulation,self._kinematics_solver, rcfg.eeframe_name)
        ee_pos, ee_rot_mat = self._articulation_kinematics_solver.compute_end_effector_pose()

        self._ee_pos = ee_pos
        self._ee_rot = ee_rot_mat

        print("post_load_scenario done")

    def reset_scenario(self):
        # teleport robot to its zero position
        self.teleport_robots_to_zeropos()

        self._target.set_world_pose(self._target_start_pos,self._target_start_rot)

        if self._enable_obstacle:
            self._obstacle.set_world_pose(self._obstacle_start_pos,self._obstacle_start_rot)

        self._rmpflow.reset()
        self.realize_rmptarg_vis(self._show_rmp_target_opt)
        if self._show_collision_bounds:
            self._rmpflow.visualize_collision_spheres()
            self.realize_collider_vis_opt(self._show_collision_bounds_opt)
        if self._show_endeffector_box:
            self._rmpflow.visualize_end_effector_position()

    def physics_step(self, step_size):

        rcfg = self.get_robot_config()
        robot_base_translation, robot_base_orientation = rcfg._articulation.get_world_pose()
        self._rmpflow.set_robot_base_pose(robot_base_translation, robot_base_orientation)

        target_pos, target_ori = self._target.get_world_pose()

        self._rmpflow.update_world()

        self._rmpflow.set_end_effector_target( target_pos, target_ori )

        action = self._articulation_rmpflow.get_next_articulation_action(step_size)
        rcfg._articulation.apply_action(action)

        ee_pos, ee_rot_mat = self._articulation_kinematics_solver.compute_end_effector_pose()

        self._ee_pos = ee_pos
        self._ee_rot = ee_rot_mat

    def teardown_scenario(self):
        pass

    def update_scenario(self, step: float):
        if not self._running_scenario:
            return
        self.physics_step(step)

    def get_scenario_actions(self):
        rv = ["Move Target to EE"]
        return rv

    def scenario_action(self, actionname, mouse_button=0):
        print("InvkinScenario action:",actionname, "   mouse_button:", mouse_button)
        if actionname == "Move Target to EE":
            # self._target.set_world_pose(np.array([0.0,-0.006,0.7668]),euler_angles_to_quats([0,0,0]))
            self._target.set_world_pose(self._ee_pos, rot_matrices_to_quats(self._ee_rot))
        else:
            print(f"Unknown actionname: {actionname}")
