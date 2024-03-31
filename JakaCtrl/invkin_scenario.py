import time
import numpy as np
import carb

from pxr import UsdPhysics, Usd, UsdGeom, Gf


from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.stage import add_reference_to_stage

from omni.isaac.core.articulations import Articulation, ArticulationSubset
from omni.isaac.core.utils.types import ArticulationAction

from omni.isaac.core.objects.cuboid import FixedCuboid
from omni.isaac.core.objects import GroundPlane
from omni.isaac.core.prims import XFormPrim
from omni.isaac.core.world import World


from omni.isaac.core.utils.numpy.rotations import euler_angles_to_quats, rot_matrices_to_quats
from .senut import apply_convex_decomposition_to_mesh_and_children, apply_material_to_prim_and_children


from .senut import add_light_to_stage
from .senut import adjust_joint_values, set_stiffness_for_joints, set_damping_for_joints
from .scenario_base import ScenarioBase

from omni.isaac.motion_generation import ArticulationKinematicsSolver, LulaKinematicsSolver
from omni.isaac.motion_generation import interface_config_loader

from omni.isaac.core.utils.stage import add_reference_to_stage,  get_current_stage
from omni.isaac.motion_generation.lula.interface_helper import LulaInterfaceHelper


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


    def __init__(self):
        self._kinematics_solver = None
        self._articulation_kinematics_solver = None

        self._articulation = None
        self._target = None


    def load_scenario(self, robot_name, ground_opt):
        super().load_scenario(robot_name, ground_opt)

        self._robcfg = self.get_robcfg(robot_name, ground_opt)


        #  self.get_robot_config(robot_name, ground_opt)
        self.phystep = 0
        self.ikerrs = 0
        self.tot_damping_factor = 1.0
        self.tot_stiffness_factor = 1.0

        self._robot_name = robot_name
        self._ground_opt = ground_opt
        self._stage = get_current_stage()

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

        self._start_robot_pos = Gf.Vec3d([0, 0, 0])
        self._start_robot_rot = [0, 0, 0]
        if self._robot_name == "ur10-suction-short":
            self._start_robot_pos = Gf.Vec3d([0, 0, 0.4])
            self._start_robot_rot = [180, 0, 0]

        stage = get_current_stage()
        roborg = UsdGeom.Xform.Define(stage, "/World/roborg")
        roborg.AddTranslateOp().Set(self._start_robot_pos)
        roborg.AddRotateXOp().Set(self._start_robot_rot[0])


        # Setup Robot Arm
        add_reference_to_stage(self._robcfg.robot_usd_file_path, self._robcfg.robot_prim_path)
        apply_convex_decomposition_to_mesh_and_children(self._stage, self._robcfg.robot_prim_path)

        self._articulation = Articulation(self._robcfg.artpath)
        world.scene.add(self._articulation)

        add_reference_to_stage(get_assets_root_path() + "/Isaac/Props/UIElements/frame_prim.usd", "/World/target")
        self._target = XFormPrim("/World/target", scale=[.04,.04,.04])

        self._world = world

    def post_load_scenario(self):
        print("InvKin post_load_scenario")

        self.register_articulation(self._articulation) # this has to happen in post_load_scenario

        # teleport robot to zeros
        self._articulation.set_joint_positions(self._robcfg.joint_zero_pos)

        # RMPflow config files for supported robots are stored in the motion_generation extension under "/motion_policy_configs"

        self._kinematics_solver = LulaKinematicsSolver(
            robot_description_path = self._robcfg.rdf_path,
            urdf_path = self._robcfg.urdf_path
        )
        self.lulaHelper = LulaInterfaceHelper(self._kinematics_solver._robot_description)

        # if self._robot_name in ["jaka-minicobo-0","jaka-minicobo-1","minicobo-rg2-high"]:
        #     # self.set_stiffness_for_all_joints(10000000.0 / 200) # 1e8 or 10 million seems too high
        #     # self.set_damping_for_all_joints(100000.0 / 20) # 1e5 or 100 thousand seems too high
        #     self.set_stiffness_for_all_joints(400.0) # 1e8 or 10 million seems too high
        #     self.set_damping_for_all_joints(40) # 1e5 or 100 thousand seems too high

        if self._robcfg.stiffness>0:
            self.set_stiffness_for_all_joints(self._robcfg.stiffness) # 1e8 or 10 million seems too high

        if self._robcfg.damping>0:
            self.set_damping_for_all_joints(self._robcfg.damping) # 1e5 or 100 thousand seems too high

        end_effector_name = self._robcfg.eeframe_name
        self._articulation_kinematics_solver = ArticulationKinematicsSolver(self._articulation,self._kinematics_solver, end_effector_name)
        ee_position,ee_rot_mat = self._articulation_kinematics_solver.compute_end_effector_pose()
        self._ee_pos = ee_position
        self._ee_rot = ee_rot_mat



        print("Valid frame names at which to compute kinematics:", self._kinematics_solver.get_all_frame_names())


    def reset_scenario(self):
        # self._target.set_world_pose(np.array([0.2,0.2,0.6]),euler_angles_to_quats([0,np.pi,0]))
        self._articulation.set_joint_positions(self._robcfg.joint_zero_pos)
        ee_position,ee_rot_mat = self._articulation_kinematics_solver.compute_end_effector_pose()
        self._ee_pos = ee_position
        self._ee_rot = ee_rot_mat

        self._target.set_world_pose(self._ee_pos, rot_matrices_to_quats(self._ee_rot))
        # if self._show_collision_bounds:
        #     self._rmpflow.reset()
        #     self._rmpflow.visualize_collision_spheres()
        #     self._rmpflow.visualize_end_effector_position()

    phystep = 0
    ikerrs = 0
    ik_solving_active = True
    msggap = 1
    last_msg_time = 0
    def physics_step(self, step_size):

        if self.ik_solving_active:
            target_position, target_orientation = self._target.get_world_pose()

            #Track any movements of the robot base
            robot_base_translation,robot_base_orientation = self._articulation.get_world_pose()
            self._kinematics_solver.set_robot_base_pose(robot_base_translation,robot_base_orientation)

            action, success = self._articulation_kinematics_solver.compute_inverse_kinematics(target_position, target_orientation)

            if success:
                self._articulation.apply_action(action)
                pass
            else:
                msg =f"IK did not converge to a solution.  No action is being taken - phystep: {self.phystep} ikerrs: {self.ikerrs}"
                if self.ikerrs == 0:
                    action, success = self._articulation_kinematics_solver.compute_inverse_kinematics(target_position, target_orientation)
                    carb.log_info(msg)
                self.ikerrs += 1
                curtime = time.time()
                elap = curtime - self.last_msg_time
                if elap > self.msggap:
                    self.last_msg_time = curtime
                    carb.log_warn(msg)
                    print(msg)
        self.phystep += 1

        ee_position,ee_rot_mat = self._articulation_kinematics_solver.compute_end_effector_pose()
        self._ee_pos = ee_position
        self._ee_rot = ee_rot_mat


    def teardown_scenario(self):
        pass

    def update_scenario(self, step: float):
        if not self._running_scenario:
            return
        self.physics_step(step)


    def set_stiffness_for_all_joints(self, stiffness):
        joint_names = self.lulaHelper.get_active_joints()
        set_stiffness_for_joints(joint_names, stiffness)

    def set_damping_for_all_joints(self, damping):
        joint_names = self.lulaHelper.get_active_joints()
        set_damping_for_joints(joint_names, damping)

    def adjust_stiffness_for_all_joints(self,fak):
        joint_names = self.lulaHelper.get_active_joints()
        print(f"joint_names:{joint_names} fak:{fak:.2f} tot_stiffness:{self.tot_stiffness_factor:.4e}")
        adjust_joint_values(joint_names,"stiffness",fak)
        self.tot_stiffness_factor = self.tot_stiffness_factor * fak

    def adjust_damping_for_all_joints(self,fak):
        joint_names = self.lulaHelper.get_active_joints()
        print(f"joint_names:{joint_names} fak:{fak:.2f} tot_damping:{self.tot_damping_factor:.4e}")
        adjust_joint_values(joint_names,"damping",fak)
        self.tot_damping_factor = self.tot_damping_factor * fak

    def scenario_action(self, actionname, mouse_button=0 ):
        print("InvkinScenario action:",actionname, "   mouse_button:",mouse_button)
        if actionname == "Toggle IkSolving":
            self.ik_solving_active = not self.ik_solving_active
        elif actionname == "Move Target to EE":
            # self._target.set_world_pose(np.array([0.0,-0.006,0.7668]),euler_angles_to_quats([0,0,0]))
            self._target.set_world_pose(self._ee_pos, rot_matrices_to_quats(self._ee_rot))
        elif actionname == "Adjust Stiffness - All Joints":
            if mouse_button>0:
                self.adjust_stiffness_for_all_joints(1.1)
            else:
                self.adjust_stiffness_for_all_joints(1/1.1)
        elif actionname == "Adjust Damping - All Joints":
            if mouse_button>0:
                self.adjust_damping_for_all_joints(1.1)
            else:
                self.adjust_damping_for_all_joints(1/1.1)
        else:
            print(f"Unknown actionname: {actionname}")

    def get_scenario_actions(self):
        rv = ["Move Target to EE","Adjust Stiffness - All Joints","Adjust Damping - All Joints","Toggle IkSolving" ]
        return rv
