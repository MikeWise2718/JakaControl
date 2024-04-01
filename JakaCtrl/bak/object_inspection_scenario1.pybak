import math
import numpy as np
import os
from pxr import Usd, UsdGeom, Gf

from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.stage import get_current_stage
from omni.isaac.core.articulations import Articulation
from omni.isaac.core.objects.cuboid import FixedCuboid
from omni.isaac.core.objects import GroundPlane
from omni.isaac.core.prims import XFormPrim
from omni.isaac.core.world import World

from omni.isaac.core.utils.extensions import get_extension_path_from_name

from omni.isaac.motion_generation import RmpFlow, ArticulationMotionPolicy

from omni.asimov.jaka.minicobo import Minicobo

from .senut import add_light_to_stage
from .senut import calc_robot_circle_pose

from .scenario_base import ScenarioBase

# Copyright (c) 2022-2023, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#


class ObjectInspectionScenario(ScenarioBase):
    _running_scenario = False
    _show_collision_bounds = False

    def __init__(self):
        pass

    def load_scenario(self, robot_name, ground_opt):
        self.get_robot_config(robot_name, ground_opt)

        self._robot_name = robot_name
        self._ground_opt = ground_opt

        add_light_to_stage()

        world = World.instance()
        if self._ground_opt == "default":
            self._ground=world.scene.add_default_ground_plane(z_position=-1.02975)

        elif self._ground_opt == "groundplane":
            self._ground = GroundPlane(prim_path="/World/groundPlane", size=10, color=np.array([0.5, 0.5, 0.5]),position=[0,0,-1.03313])
            world.scene.add(self._ground)

        elif self._ground_opt == "groundplane-blue":
            self._ground = GroundPlane(prim_path="/World/groundPlane", size=10, color=np.array([0.0, 0.0, 0.5]),position=[0,0,-1.03313])
            world.scene.add(self._ground)

        stage = get_current_stage()
        roborg = UsdGeom.Xform.Define(stage,"/World/roborg")
        self._rob_tranop = roborg.AddTranslateOp()
        self._rob_zrotop = roborg.AddRotateZOp()
        self._rob_yrotop = roborg.AddRotateYOp()
        self._rob_xrotop = roborg.AddRotateXOp()

        roborg1 = UsdGeom.Xform.Define(stage,"/World/roborg1")
        self._rob_tranop1 = roborg1.AddTranslateOp()
        self._rob_zrotop1 = roborg1.AddRotateZOp()
        self._rob_yrotop1 = roborg1.AddRotateYOp()
        self._rob_xrotop1 = roborg1.AddRotateXOp()


        self._rob_ang = 180
        self._start_robot_pos = Gf.Vec3d([0, 0, 0])
        self._start_robot_rot = [0, 0, 0]
        if self._robot_name == "ur10-suction-short":
            self._start_robot_pos = Gf.Vec3d([0, 0, 0.4])
            self._start_robot_rot = [0, 0, 0]
        elif self._robot_name in ["minicobo-rg2-high","minicobo-suction-high","minicobo-dual-high"]:
            # self._start_robot_pos = Gf.Vec3d([-0.35, 0, 0.80])
            # self._start_robot_rot = [0, 130, 0]
            # cen = [0.11, 0, 0.77]
            # rad = 0.35
            # xang = 0
            # yang = 150
            cen = [-0.11, 0, 0.77]
            rad = 0.35
            xang = 0
            yang = -150
            pos, rot = calc_robot_circle_pose(self._rob_ang, cen=cen, rad=rad, xang=xang, yang=yang)
            self._start_robot_pos = pos
            self._start_robot_rot = rot
            cen = [0.11, 0, 0.77]
            rad = 0.35
            xang = 0
            yang = 150
            pos1, rot1 = calc_robot_circle_pose(self._rob_ang, cen=cen, rad=rad, xang=xang, yang=yang)
            self._start_robot_pos1 = pos1
            self._start_robot_rot1 = rot1

        self.set_robot_circle_pose(self._start_robot_pos, self._start_robot_rot)

        add_reference_to_stage(self._cfg_robot_usd_file_path, self._cfg_robot_prim_path)

        self.set_robot_circle_pose1(self._start_robot_pos1, self._start_robot_rot1)

        self._cfg_robot_prim_path1 = self._cfg_robot_prim_path.replace("roborg", "roborg1")

        add_reference_to_stage(self._cfg_robot_usd_file_path, self._cfg_robot_prim_path1)

#         self.robot = Minicobo(self._cfg_robot_prim_path, self._robot_name, self._cfg_robot_usd_file_path)


        self._articulation = Articulation(self._cfg_artpath)

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
        self._target = XFormPrim("/World/target", scale=[.04,.04,.04], position=[0.29915,-0.03815,1.11373])
        self._object =  self._target

        jakacontrol_extension_path = get_extension_path_from_name("JakaControl")

        path_to_cage_usd = f"{jakacontrol_extension_path}/usd/cage_v1.usd"
        add_reference_to_stage(path_to_cage_usd, "/World/cage_v1")
        self._cage = XFormPrim("/World/cage_v1", scale=[1,1,1], position=[-0.38605,0,-0.00045])

        self._world = world

    def setup_scenario(self):
        print("ObjectInspection setup_scenario")

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

        self._obstacle = FixedCuboid("/World/obstacle",size=.05,position=np.array([0.4, 0.0, 1.65]),color=np.array([0.,0.,1.]))



        print("rdf_path:",self._cfg_rdf_path)
        print("urdf_path:",self._cfg_urdf_path)
        print("rmp_config_path:",self._cfg_rmp_config_path)
        print("eeframe_name:",self._cfg_eeframe_name)
        print("max_step_size:",self._cfg_max_step_size)
        # Initialize an RmpFlow object
        self._rmpflow = RmpFlow(
            robot_description_path = self._cfg_rdf_path,
            urdf_path = self._cfg_urdf_path,
            rmpflow_config_path = self._cfg_rmp_config_path,
            end_effector_frame_name = self._cfg_eeframe_name,
            maximum_substep_size = self._cfg_max_step_size
        )
        self._rmpflow.add_obstacle(self._obstacle)

        if self._show_collision_bounds:
            self._rmpflow.set_ignore_state_updates(True)
            self._rmpflow.visualize_collision_spheres()

            # Set the robot gains to be deliberately poor
            bad_proportional_gains = self._articulation.get_articulation_controller().get_gains()[0]/50
            self._articulation.get_articulation_controller().set_gains(kps = bad_proportional_gains)

        print("Created _rmpflow object")

        self._articulation_rmpflow = ArticulationMotionPolicy(self._articulation,self._rmpflow)

        # self._target.set_world_pose(np.array([.5,0,.7]),euler_angles_to_quats([0,np.pi,0]))

    def set_robot_circle_pose(self, pos, rot):
        self._rob_tranop.Set(pos)
        self._rob_zrotop.Set(rot[2])
        self._rob_yrotop.Set(rot[1])
        self._rob_xrotop.Set(rot[0])
        self._start_robot_pos = pos
        self._start_robot_rot = rot

    def set_robot_circle_pose1(self, pos, rot):
        self._rob_tranop1.Set(pos)
        self._rob_zrotop1.Set(rot[2])
        self._rob_yrotop1.Set(rot[1])
        self._rob_xrotop1.Set(rot[0])
        self._start_robot_pos1 = pos
        self._start_robot_rot1 = rot

    def post_load_scenario(self):
        self._rmpflow.add_obstacle(self._obstacle)

        self._world.add_physics_callback("sim_step", callback_fn=self.physics_step)
        pass

    def reset_scenario(self):
        # self._target.set_world_pose(np.array([.5,0,.7]),euler_angles_to_quats([0,np.pi,0]))
        if self._show_collision_bounds:
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
