import math
import numpy as np
import os

from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.stage import add_reference_to_stage

from omni.isaac.core.articulations import Articulation
from omni.isaac.core.objects.cuboid import FixedCuboid
from omni.isaac.core.objects import GroundPlane
from omni.isaac.core.prims import XFormPrim
from omni.isaac.core.world import World

from pxr import UsdPhysics, Usd, UsdGeom, Gf

from omni.isaac.core.utils.extensions import get_extension_path_from_name

from omni.isaac.core.utils.numpy.rotations import euler_angles_to_quats, rot_matrices_to_quats

from omni.isaac.motion_generation import RmpFlow, ArticulationMotionPolicy

from .senut import add_light_to_stage, get_robot_params, get_robot_rmp_params
from .senut import ScenarioTemplate


from omni.isaac.core.utils.stage import add_reference_to_stage,  get_current_stage
from omni.isaac.motion_generation.lula.interface_helper import LulaInterfaceHelper
from omni.isaac.motion_generation import ArticulationKinematicsSolver

from .senut import adjust_joint_value, adjust_joint_values, set_stiffness_for_joints, set_damping_for_joints


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
        if not ok:
            print(f"Unknown robot name {self._robot_name}")
            return

        if path_to_robot_usd is not None:
                # if self._robot_name == "franka":
                #     stage = get_current_stage()
                #     roborg = UsdGeom.Xform.Define(stage, "/World/roborg")
                #     pos = Gf.Vec3d([0, 0.0, 1.6])
                #     roborg.AddTranslateOp().Set(pos)
                #     roborg.AddRotateXOp().Set(180)
                add_reference_to_stage(path_to_robot_usd, robot_prim_path)


        # if path_to_robot_usd is not None:
        #     robprim = add_reference_to_stage(path_to_robot_usd, robot_prim_path)
            # if robot_name == "ur10-suction-short":
            #     gripper_base =
            #     robprim.set_visibility(False)

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

        self.lulaHelper = LulaInterfaceHelper(self._rmpflow._robot_description)

        if self._robot_name in ["jaka-minicobo","jaka-minicobo-1"]:
            self.set_stiffness_for_all_joints(10000000.0 / 200) # 1e8 or 10 million seems too high
            self.set_damping_for_all_joints(100000.0 / 20) # 1e5 or 100 thousand seems too high

        if self._show_collsion_bounds:
            self._rmpflow.set_ignore_state_updates(True)
            self._rmpflow.visualize_collision_spheres()
            self._rmpflow.visualize_end_effector_position()

            # Set the robot gains to be deliberately poor
            # bad_proportional_gains = self._articulation.get_articulation_controller().get_gains()[0]/50
            # self._articulation.get_articulation_controller().set_gains(kps = bad_proportional_gains)

        print("Created _rmpflow object")

        self._articulation_rmpflow = ArticulationMotionPolicy(self._articulation,self._rmpflow)
        self._kinematics_solver = self._rmpflow.get_kinematics_solver()


        self._articulation_kinematics_solver = ArticulationKinematicsSolver(self._articulation,self._kinematics_solver, eeframe_name)
        ee_pos, ee_rot_mat = self._articulation_kinematics_solver.compute_end_effector_pose()

        self._ee_pos = ee_pos
        self._ee_rot = ee_rot_mat


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
            self._rmpflow.visualize_end_effector_position()


    def set_stiffness_for_all_joints(self, stiffness):
        joint_names = self.lulaHelper.get_active_joints()
        set_stiffness_for_joints(joint_names, stiffness)

    def set_damping_for_all_joints(self, damping):
        joint_names = self.lulaHelper.get_active_joints()
        set_damping_for_joints(joint_names, damping)

    def physics_step(self, step_size):
        target_position, target_orientation = self._target.get_world_pose()

        self._rmpflow.update_world()


        self._rmpflow.set_end_effector_target(
            target_position, target_orientation
        )

        action = self._articulation_rmpflow.get_next_articulation_action(step_size)
        self._articulation.apply_action(action)

        ee_pos, ee_rot_mat = self._articulation_kinematics_solver.compute_end_effector_pose()

        self._ee_pos = ee_pos
        self._ee_rot = ee_rot_mat


    def teardown_scenario(self):
        pass

    def update_scenario(self, step: float):
        if not self._running_scenario:
            return

    def get_actions(self):
        rv = ["Move Target to EE","Adjust Stiffness - All Joints","Adjust Damping - All Joints" ]
        return rv

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
        if actionname == "Move Target to EE":
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