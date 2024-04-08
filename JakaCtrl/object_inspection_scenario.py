import numpy as np
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
from omni.isaac.core.utils.rotations import euler_angles_to_quat

from .senut import add_dome_light_to_stage
from .senut import apply_convex_decomposition_to_mesh_and_children, apply_material_to_prim_and_children
from .senut import apply_diable_gravity_to_rigid_bodies, adjust_articulation

from .senut import add_camera_to_robot

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
    _colorScheme = "transparent"

    def __init__(self):
        super().__init__()
        self._scenario_name = "object-inspection"
        self._scenario_description = ScenarioBase.get_scenario_desc(self._scenario_name)
        self._nrobots = 2
        pass

    def load_scenario(self, robot_name, ground_opt):
        super().load_scenario(robot_name, ground_opt)
        # self.get_robot_config(robot_name, ground_opt)
        self._robcfg = self.get_robcfg(robot_name, ground_opt)
        self._robcfg1 = self.get_robcfg(robot_name, ground_opt)

        self._robot_name = robot_name
        self._ground_opt = ground_opt

        add_dome_light_to_stage()

        world = World.instance()
        if self._ground_opt == "default":
            self._ground=world.scene.add_default_ground_plane(z_position=-1.02)

        elif self._ground_opt == "groundplane":
            self._ground = GroundPlane(prim_path="/World/groundPlane", size=10, color=np.array([0.5, 0.5, 0.5]),position=[0,0,-1.03313])
            world.scene.add(self._ground)

        elif self._ground_opt == "groundplane-blue":
            self._ground = GroundPlane(prim_path="/World/groundPlane", size=10, color=np.array([0.0, 0.0, 0.5]),position=[0,0,-1.03313])
            world.scene.add(self._ground)

        stage = get_current_stage()

        self._start_robot_pos = Gf.Vec3d([0, 0, 0])
        self._start_robot_rot = [0, 0, 0]

        # Robot 0
        cen, rad = [-0.08, 0, 0.77], 0
        xang, yang, zang = 0, -150, 0
        pos, rot = self.calc_oi_robot_pose(cen=cen, radius=rad, xang=xang, yang=yang, zang=zang)

        roborg = UsdGeom.Xform.Define(stage,"/World/roborg")
        self.set_oi_robot_pose(roborg, self._robcfg, pos, rot)

        add_reference_to_stage(self._robcfg.robot_usd_file_path, self._robcfg.robot_prim_path)
        apply_convex_decomposition_to_mesh_and_children(stage, self._robcfg.robot_prim_path)
        apply_diable_gravity_to_rigid_bodies(stage, self._robcfg.robot_prim_path)
        adjust_articulation(stage, self._robcfg.robot_prim_path)
        self._articulation = Articulation(self._robcfg.artpath,"mico-0")

        # Robot 1
        cen1, rad1 = [0.14, 0, 0.77], 0
        xang1, yang1, zang1 = 0, 150, 0
        pos1, rot1 = self.calc_oi_robot_pose(cen=cen1, radius=rad1, xang=xang1, yang=yang1, zang=zang1)

        roborg1 = UsdGeom.Xform.Define(stage,"/World/roborg1")
        self.set_oi_robot_pose(roborg1, self._robcfg1, pos1, rot1)

        self._robcfg1.artpath = self._robcfg1.artpath.replace("roborg", "roborg1")
        self._robcfg1.robot_prim_path = self._robcfg1.robot_prim_path.replace("roborg", "roborg1")

        add_reference_to_stage(self._robcfg1.robot_usd_file_path, self._robcfg1.robot_prim_path)
        apply_convex_decomposition_to_mesh_and_children(stage, self._robcfg1.robot_prim_path)
        apply_diable_gravity_to_rigid_bodies(stage, self._robcfg1.robot_prim_path)
        adjust_articulation(stage, self._robcfg1.robot_prim_path)
        self._articulation1 = Articulation(self._robcfg1.artpath,"mico-1")

        world.scene.add(self._articulation)
        world.scene.add(self._articulation1)

        # tagets
        quat = euler_angles_to_quat([-np.pi/2,0,0])
        self._target = XFormPrim("/World/target", scale=[.04,.04,.04], position=[-0.15, 0.00, 0.02], orientation=quat)
        add_reference_to_stage(get_assets_root_path() + "/Isaac/Props/UIElements/frame_prim.usd", "/World/target")

        quat = euler_angles_to_quat([-np.pi/2,0,0])
        self._target1 = XFormPrim("/World/target1", scale=[.04,.04,.04], position=[0.15, 0.00, 0.02], orientation=quat)
        add_reference_to_stage(get_assets_root_path() + "/Isaac/Props/UIElements/frame_prim.usd", "/World/target1")

        # cage
        jakacontrol_extension_path = get_extension_path_from_name("JakaControl")
        path_to_cage_usd = f"{jakacontrol_extension_path}/usd/cage_v1.usd"
        add_reference_to_stage(path_to_cage_usd, "/World/cage_v1")

        cagepath = "/World/cage_v1"
        self._cage = XFormPrim(cagepath, scale=[1,1,1], position=[0,0,0])
        if self._colorScheme == "default":
            self._cage.set_color([0.5, 0.5, 0.5, 1.0])
        elif self._colorScheme == "transparent":
            apply_material_to_prim_and_children(stage, self._matman, "Steel_Blued", cagepath)

        self._world = world

    def setup_scenario(self):
        print("ObjectInspection setup_scenario")

        self.register_articulation(self._articulation, self._robcfg) # this has to happen in post_load_scenario
        self.register_articulation(self._articulation1, self._robcfg1) # this has to happen in post_load_scenario

        self._running_scenario = True

        self._joint_index = 0
        self._lower_joint_limits = self._articulation.dof_properties["lower"]
        self._upper_joint_limits = self._articulation.dof_properties["upper"]
        self._njoints = self._articulation.num_dof
        self._zeros = np.zeros(self._njoints)
        print(f"jaka - njoints:{self._njoints} lower:{self._lower_joint_limits} upper:{self._upper_joint_limits}")

        # teleport robot to lower joint range
        epsilon = 0.001
        self._articulation.set_joint_positions(self._zeros + epsilon)
        self._articulation1.set_joint_positions(self._zeros + epsilon)

        self._obstacle = FixedCuboid("/World/obstacle",size=.05,position=np.array([0.4, 0.0, 1.65]),color=np.array([0.,0.,1.]))

        # Initialize an RmpFlow object
        self._rmpflow = RmpFlow(
            robot_description_path = self._robcfg.rdf_path,
            urdf_path = self._robcfg.urdf_path,
            rmpflow_config_path = self._robcfg.rmp_config_path,
            end_effector_frame_name = self._robcfg.eeframe_name,
            maximum_substep_size = self._robcfg.max_step_size
        )
        quat = euler_angles_to_quat(self._robcfg.robot_rotvek)
        self._rmpflow.set_robot_base_pose(self._robcfg.start_robot_pos, quat)
        self._rmpflow.add_obstacle(self._obstacle)
        # self._rmpflow.add_obstacle(self._cage)

        self._rmpflow1 = RmpFlow(
            robot_description_path = self._robcfg1.rdf_path,
            urdf_path = self._robcfg1.urdf_path,
            rmpflow_config_path = self._robcfg1.rmp_config_path,
            end_effector_frame_name = self._robcfg1.eeframe_name,
            maximum_substep_size = self._robcfg1.max_step_size
        )
        quat = euler_angles_to_quat(self._robcfg1.robot_rotvek)
        self._rmpflow1.set_robot_base_pose(self._robcfg1.start_robot_pos, quat)
        self._rmpflow1.add_obstacle(self._obstacle)
        # self._rmpflow1.add_obstacle(self._cage)

        self.set_stiffness_and_damping_for_all_joints(self._robcfg)
        self.set_stiffness_and_damping_for_all_joints(self._robcfg1)

        _, campath = add_camera_to_robot(self._robcfg.robot_name, self._robcfg.robot_id, self._robcfg.robot_prim_path)
        self.add_camera_to_camlist(self._robcfg.robot_id, self._robcfg.robot_name, campath)
        _, campath = add_camera_to_robot(self._robcfg1.robot_name, self._robcfg1.robot_id, self._robcfg1.robot_prim_path)
        self.add_camera_to_camlist(self._robcfg1.robot_id, self._robcfg1.robot_name, campath)

        self._rmpflow.set_ignore_state_updates(True)
        self._rmpflow.visualize_collision_spheres()
        self._rmpflow1.set_ignore_state_updates(True)
        self._rmpflow1.visualize_collision_spheres()

        self._articulation_rmpflow = ArticulationMotionPolicy(self._articulation,self._rmpflow)
        self._articulation_rmpflow1 = ArticulationMotionPolicy(self._articulation1,self._rmpflow1)

    def calc_oi_robot_pose(self, cen=[0, 0, 0.85], radius=0.35, xang=0, yang=130, zang=0):
        ang_rads = np.pi
        pos = cen + radius*np.array([np.cos(ang_rads), np.sin(ang_rads), 0])
        pos = Gf.Vec3d(list(pos))
        rot = [xang, yang, zang]
        return pos, rot

    def set_oi_robot_pose(self, gprim, rcfg, pos, rot):
        rcfg.tranop = gprim.AddTranslateOp()
        rcfg.zrotop = gprim.AddRotateZOp()
        rcfg.yrotop = gprim.AddRotateYOp()
        rcfg.xrotop = gprim.AddRotateXOp()
        rcfg.tranop.Set(pos)
        rcfg.zrotop.Set(rot[2])
        rcfg.yrotop.Set(rot[1])
        rcfg.xrotop.Set(rot[0])
        rcfg.start_robot_pos = pos
        rcfg.start_robot_rot = rot
        rcfg.robot_rotvek = np.array(rot)*np.pi/180

    def reset_scenario(self):
        self._rmpflow.visualize_collision_spheres()
        self._rmpflow.visualize_end_effector_position()
        self._rmpflow1.visualize_collision_spheres()
        self._rmpflow1.visualize_end_effector_position()

    def physics_step(self, step_size):
        self.global_time += step_size
        target_position, target_orientation = self._target.get_world_pose()
        target1_position, target1_orientation = self._target1.get_world_pose()

        self._rmpflow.update_world()
        self._rmpflow1.update_world()

        self._rmpflow.set_end_effector_target( target_position, target_orientation )
        self._rmpflow1.set_end_effector_target(  target1_position, target1_orientation  )

        if self.rmpactive:
            action = self._articulation_rmpflow.get_next_articulation_action(step_size)
            self._articulation.apply_action(action)
            action1 = self._articulation_rmpflow1.get_next_articulation_action(step_size)
            self._articulation1.apply_action(action1)

    def update_scenario(self, step: float):
        if not self._running_scenario:
            return
        self.physics_step(step)

    def scenario_action(self, action_name, action_args):
        if action_name in self.base_actions:
            rv = super().scenario_action(action_name, action_args)
            return rv

    def get_scenario_actions(self):
        self.base_actions = super().get_scenario_actions()
        combo  = self.base_actions + []
        return combo
