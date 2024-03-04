import math
import time
import numpy as np
import os
import carb

from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.stage import add_reference_to_stage

from omni.isaac.core.articulations import Articulation, ArticulationSubset
from omni.isaac.core.utils.types import ArticulationAction

from omni.isaac.core.objects.cuboid import FixedCuboid
from omni.isaac.core.objects import GroundPlane
from omni.isaac.core.prims import XFormPrim
from omni.isaac.core.world import World

from pxr import UsdPhysics, Usd, UsdGeom, Gf

from omni.isaac.core.utils.extensions import get_extension_path_from_name

from omni.isaac.core.utils.numpy.rotations import euler_angles_to_quats, rot_matrices_to_quats

from omni.isaac.motion_generation import RmpFlow, ArticulationMotionPolicy

from .senut import add_light_to_stage, get_robot_params, get_robot_rmp_params
from .senut import adjust_joint_value, adjust_joint_values, set_stiffness_for_joints, set_damping_for_joints
from .senut import ScenarioTemplate

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


class InvkinScenario(ScenarioTemplate):
    _running_scenario = False
    _show_collsion_bounds = False


    def __init__(self):
        self._kinematics_solver = None
        self._articulation_kinematics_solver = None

        self._articulation = None
        self._target = None

    def load_scenario(self, robot_name, ground_opt):
        self.phystep = 0
        self.ikerrs = 0
        self.tot_damping_factor = 1.0
        self.tot_stiffness_factor = 1.0

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
        (ok, robot_prim_path, artpath, path_to_robot_usd, mopo_robot_name) = get_robot_params(self._robot_name)
        print(f"robot_prim_path:{robot_prim_path}")
        print(f"artpath:{artpath}")
        print(f"path_to_robot_usd:{path_to_robot_usd}")
        print(f"mopo_robot_name:{mopo_robot_name}")
        if not ok:
            print(f"Unknown robot name {self._robot_name}")
            return

        if path_to_robot_usd is not None:
            add_reference_to_stage(path_to_robot_usd, robot_prim_path)

        self._articulation = Articulation(artpath)

        if self._articulation is not None:
            world.scene.add(self._articulation)




        add_reference_to_stage(get_assets_root_path() + "/Isaac/Props/UIElements/frame_prim.usd", "/World/target")
        self._target = XFormPrim("/World/target", scale=[.04,.04,.04])
        self._object =  self._target

        self._world = world

    def setup_scenario(self):
        print("InvKin setup_scenario")

        self.phystep = 0
        self.ikerrs = 0
        self.tot_damping_factor = 1.0
        self.tot_stiffness_factor = 1.0

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
        kinematics_config_dir = os.path.join(mg_extension_path, "motion_policy_configs")
        self._kinematics_solver = LulaKinematicsSolver(
            robot_description_path = rdf_path,
            urdf_path = urdf_path
        )
        self.lulaHelper = LulaInterfaceHelper(self._kinematics_solver._robot_description)
        # self._kinematics_solver = LulaKinematicsSolver(
        #     robot_description_path = kinematics_config_dir + "/franka/rmpflow/robot_descriptor.yaml",
        #     urdf_path = kinematics_config_dir + "/franka/lula_franka_gen.urdf"
        # )

        if self._robot_name in ["jaka-minicobo","jaka-minicobo-1"]:
            self.set_stiffness_for_all_joints(10000000.0 / 200) # 1e8 or 10 million seems too high
            self.set_damping_for_all_joints(100000.0 / 20) # 1e5 or 100 thousand seems too high

        end_effector_name = eeframe_name
        self._articulation_kinematics_solver = ArticulationKinematicsSolver(self._articulation,self._kinematics_solver, end_effector_name)
        ee_position,ee_rot_mat = self._articulation_kinematics_solver.compute_end_effector_pose()
        self._ee_pos = ee_position
        self._ee_rot = ee_rot_mat



        print("Valid frame names at which to compute kinematics:", self._kinematics_solver.get_all_frame_names())

        self._articulation.apply_action(ArticulationAction(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0])))

        # if self._show_collsion_bounds:
        #     self._rmpflow.set_ignore_state_updates(True)
        #     self._rmpflow.visualize_collision_spheres()
        #     self._rmpflow.visualize_end_effector_position()





    def post_load_scenario(self):

        self._world.add_physics_callback("sim_step", callback_fn=self.physics_step)
        pass

    def reset_scenario(self):
        # self._target.set_world_pose(np.array([0.2,0.2,0.6]),euler_angles_to_quats([0,np.pi,0]))
        self._target.set_world_pose(self._ee_pos, rot_matrices_to_quats(self._ee_rot))
        # if self._show_collsion_bounds:
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

    def action(self, actionname, mouse_button=0 ):
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

    def get_actions(self):
        rv = ["Move Target to EE","Adjust Stiffness - All Joints","Adjust Damping - All Joints","Toggle IkSolving" ]
        return rv
