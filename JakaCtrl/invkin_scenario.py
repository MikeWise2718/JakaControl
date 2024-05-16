import time
import numpy as np
import carb

from pxr import UsdPhysics, Usd, UsdGeom, Gf

from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.prims import XFormPrim

from omni.isaac.core.utils.numpy.rotations import euler_angles_to_quats, rot_matrices_to_quats

from .senut import add_sphere_light_to_stage
from .scenario_base import ScenarioBase

from omni.isaac.motion_generation import ArticulationKinematicsSolver, LulaKinematicsSolver

from omni.isaac.core.utils.stage import add_reference_to_stage,  get_current_stage


# Copyright (c) 2022-2023, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

class InvkinScenario(ScenarioBase):
    _running_scenario = False
    _show_collision_bounds = True

    def __init__(self, uibuilder=None):
        super().__init__()
        self._scenario_name = "inverse-kinematics"
        self._scenario_description = ScenarioBase.get_scenario_desc(self._scenario_name)
        self._kinematics_solver = None
        self._articulation_kinematics_solver = None

        self._articulation = None
        self._target = None
        self._nrobots = 1
        self.uibuilder = uibuilder

    def load_scenario(self, robot_name, ground_opt):
        super().load_scenario(robot_name, ground_opt)

        # self._ro bcfg = self.create_robot_config(robot_name, ground_opt)

        self.add_light("sphere_light")
        self.add_ground(ground_opt)

        self.create_robot_config(robot_name, "/World/roborg", ground_opt)
        # self._robcfg = self.get_robot_config()
        self.load_robot_into_scene()

        self.phystep = 0
        self.ikerrs = 0

        self._robot_name = robot_name
        self._ground_opt = ground_opt
        self._stage = get_current_stage()

        self.add_light("sphere_light")
        self.add_ground(ground_opt)

        add_reference_to_stage(get_assets_root_path() + "/Isaac/Props/UIElements/frame_prim.usd", "/World/target")
        sz = 0.04
        self._target = XFormPrim("/World/target", scale=[sz, sz, sz])

    def post_load_scenario(self):
        print("InvKin post_load_scenario")

        self.register_robot_articulations()
        self.teleport_robots_to_zeropos()

        # RMPflow config files for supported robots are stored in the motion_generation extension under "/motion_policy_configs"

        rcfg = self.get_robot_config()
        rcfg._kinematics_solver = LulaKinematicsSolver(
            robot_description_path = rcfg.rdf_path,
            urdf_path = rcfg.urdf_path
        )

        eename = rcfg.eeframe_name
        rcfg._articulation_kinematics_solver = ArticulationKinematicsSolver(rcfg._articulation, rcfg._kinematics_solver, eename)
        ee_position, ee_rot_mat = rcfg._articulation_kinematics_solver.compute_end_effector_pose()
        self._ee_pos = ee_position
        self._ee_rot = ee_rot_mat

        print("Valid frame names at which to compute kinematics:", rcfg._kinematics_solver.get_all_frame_names())

    def reset_scenario(self):
        self.teleport_robots_to_zeropos()

        rcfg = self.get_robot_config()

        ee_position, ee_rot_mat = rcfg._articulation_kinematics_solver.compute_end_effector_pose()
        self._ee_pos = ee_position
        self._ee_rot = ee_rot_mat

        self._target.set_world_pose(self._ee_pos, rot_matrices_to_quats(self._ee_rot))

    phystep = 0
    ikerrs = 0
    ik_solving_active = True
    msggap = 1
    last_msg_time = 0

    def physics_step(self, step_size):
        rcfg = self.get_robot_config()

        if self.ik_solving_active:
            target_position, target_orientation = self._target.get_world_pose()

            # Track any movements of the robot base
            robot_base_translation, robot_base_orientation = rcfg._articulation.get_world_pose()
            rcfg._kinematics_solver.set_robot_base_pose(robot_base_translation, robot_base_orientation)

            action, success = rcfg._articulation_kinematics_solver.compute_inverse_kinematics(target_position, target_orientation)

            if success:
                # print(f"step:{self.phystep}")
                # print(f"action:{action}")
                rcfg._articulation.apply_action(action)
                pass
            else:
                msg =f"IK did not converge to a solution.  No action is being taken - phystep: {self.phystep} ikerrs: {self.ikerrs}"
                if self.ikerrs == 0:
                    action, success = rcfg._articulation_kinematics_solver.compute_inverse_kinematics(target_position, target_orientation)
                    carb.log_info(msg)
                self.ikerrs += 1
                curtime = time.time()
                elap = curtime - self.last_msg_time
                if elap > self.msggap:
                    self.last_msg_time = curtime
                    carb.log_warn(msg)
                    print(msg)
        self.phystep += 1

        ee_position, ee_rot_mat = rcfg._articulation_kinematics_solver.compute_end_effector_pose()
        self._ee_pos = ee_position
        self._ee_rot = ee_rot_mat

    def teardown_scenario(self):
        pass

    def update_scenario(self, step: float):
        if not self._running_scenario:
            return
        self.physics_step(step)

    def scenario_action(self, actionname, mouse_button=0 ):
        print("InvkinScenario action:", actionname, "   mouse_button:", mouse_button)
        if actionname == "Toggle IkSolving":
            self.ik_solving_active = not self.ik_solving_active
        elif actionname == "Move Target to EE":
            # self._target.set_world_pose(np.array([0.0,-0.006,0.7668]),euler_angles_to_quats([0,0,0]))
            self._target.set_world_pose(self._ee_pos, rot_matrices_to_quats(self._ee_rot))
        else:
            print(f"Unknown actionname: {actionname}")

    def get_scenario_actions(self):
        rv = ["Move Target to EE", "Toggle IkSolving"]
        return rv
