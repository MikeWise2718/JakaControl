import numpy as np
import carb

from pxr import UsdPhysics, Usd, UsdGeom, Gf


from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.stage import add_reference_to_stage

from omni.isaac.core.articulations import Articulation
from omni.isaac.core.objects.cuboid import FixedCuboid
from omni.isaac.core.objects import GroundPlane
from omni.isaac.core.prims import XFormPrim
from omni.isaac.core.world import World

from omni.isaac.core.utils.numpy.rotations import euler_angles_to_quats, rot_matrices_to_quats

from omni.isaac.motion_generation import RmpFlow, ArticulationMotionPolicy

from .senut import add_light_to_stage
from .senut import ScenarioTemplate

from omni.isaac.core.utils.stage import add_reference_to_stage,  get_current_stage
from omni.isaac.motion_generation.lula.interface_helper import LulaInterfaceHelper
from omni.isaac.motion_generation import ArticulationKinematicsSolver

from .senut import adjust_joint_values, set_stiffness_for_joints, set_damping_for_joints


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
    _show_collision_bounds = True

    def __init__(self):
        pass

    def load_scenario(self, robot_name, ground_opt):
        # Here we do object loading and simple initialization

        self.get_robot_config(robot_name, ground_opt)

        self.tot_damping_factor = 1.0
        self.tot_stiffness_factor = 1.0

        self._robot_name = robot_name
        self._ground_opt = ground_opt

        self._target_start_pos = np.array([0.5, 0.0, 0.7])
        self._target_start_rot = euler_angles_to_quats([0, np.pi, 0])
        self._obstacle_start_pos = np.array([0.4, 0.0, 0.65])
        self._obstacle_start_rot = euler_angles_to_quats([0, np.pi, 0])

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
        # roborg.AddRotateXOp().Set(self._start_robot_rot[0])



        # Setup Robot ARm
        add_reference_to_stage(self._cfg_path_to_robot_usd, self._cfg_robot_prim_path)
        self._articulation = Articulation(self._cfg_artpath)
        world.scene.add(self._articulation)


        # Add a target to the stage
        add_reference_to_stage(get_assets_root_path() + "/Isaac/Props/UIElements/frame_prim.usd", "/World/target")
        self._target = XFormPrim("/World/target", scale=[.04,.04,.04])

        self._world = world

        self._obstacle = FixedCuboid("/World/obstacle",size=.05,color=np.array([0.,0.,1.]))


    def post_load_scenario(self):
        # Here we do multi-object initialization - things that needs to be done after all objects are loaded

        self.register_articulation(self._articulation) # this has to happen in post_load_scenario

        # teleport robot to its zero position
        self._articulation.set_joint_positions(self._cfg_joint_zero_pos)

        # Initialize an RmpFlow object
        self._rmpflow = RmpFlow(
            robot_description_path = self._cfg_rdf_path,
            urdf_path = self._cfg_urdf_path,
            rmpflow_config_path = self._cfg_rmp_config_path,
            end_effector_frame_name = self._cfg_eeframe_name,
            maximum_substep_size = self._cfg_max_step_size
        )
        self._rmpflow.add_obstacle(self._obstacle)

        self.lulaHelper = LulaInterfaceHelper(self._rmpflow._robot_description)

        if self._robot_name in ["jaka-minicobo","jaka-minicobo-1"]:
            self.set_stiffness_for_all_joints(10000000.0 / 200) # 1e8 or 10 million seems too high
            self.set_damping_for_all_joints(100000.0 / 20) # 1e5 or 100 thousand seems too high

        self._articulation_rmpflow = ArticulationMotionPolicy(self._articulation,self._rmpflow)
        self._kinematics_solver = self._rmpflow.get_kinematics_solver()


        self._articulation_kinematics_solver = ArticulationKinematicsSolver(self._articulation,self._kinematics_solver, self._cfg_eeframe_name)
        ee_pos, ee_rot_mat = self._articulation_kinematics_solver.compute_end_effector_pose()

        self._ee_pos = ee_pos
        self._ee_rot = ee_rot_mat

        self._rmpflow.add_obstacle(self._obstacle)

    def reset_scenario(self):
        # teleport robot to its zero position
        self._articulation.set_joint_positions(self._cfg_joint_zero_pos)

        # self._target.set_world_pose(np.array([.5,0,.7]),euler_angles_to_quats([0,np.pi,0]))
        self._target.set_world_pose(self._target_start_pos,self._target_start_rot)
        self._obstacle.set_world_pose(self._obstacle_start_pos,self._obstacle_start_rot)


        self._rmpflow.reset()
        if self._show_collision_bounds:
       #     self._rmpflow.set_ignore_state_updates(True)
            self._rmpflow.visualize_collision_spheres()
            self._rmpflow.visualize_end_effector_position()


    def set_stiffness_for_all_joints(self, stiffness):
        joint_names = self.lulaHelper.get_active_joints()
        set_stiffness_for_joints(joint_names, stiffness)

    def set_damping_for_all_joints(self, damping):
        joint_names = self.lulaHelper.get_active_joints()
        set_damping_for_joints(joint_names, damping)

    def physics_step(self, step_size):
        robot_base_translation,robot_base_orientation = self._articulation.get_world_pose()
        self._rmpflow.set_robot_base_pose(robot_base_translation,robot_base_orientation)


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
        self.physics_step(step)


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