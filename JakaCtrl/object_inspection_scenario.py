import numpy as np
from pxr import Usd, UsdGeom, Gf

from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.stage import get_current_stage
from omni.isaac.core.objects.cuboid import FixedCuboid
from omni.isaac.core.objects import GroundPlane
from omni.isaac.core.prims import XFormPrim
from omni.isaac.core.world import World

from omni.isaac.core.utils.extensions import get_extension_path_from_name

from omni.isaac.motion_generation import RmpFlow, ArticulationMotionPolicy
from omni.isaac.core.utils.rotations import euler_angles_to_quat

from .senut import add_dome_light_to_stage
from .senut import apply_material_to_prim_and_children

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
        self._robcfg = self.create_robot_config(robot_name,"/World/roborg", ground_opt)
        self._robcfg1 = self.create_robot_config(robot_name,"/World/roborg1", ground_opt)

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
        (cen, rad, rot) = ([-0.08, 0, 0.77], 0, [0, -150, 0])
        # self._articulation = self.setup_robot(0, cen, rad, rot)
        self._articulation = self.load_robot_into_scene(0, cen, rot)

        # Robot 1
        (cen1, rad1, rot1) = ([0.14, 0, 0.77], 0, [0, 150, 0])
        # self._articulation1 = self.setup_robot(1, cen1, rad1, rot1)
        self._articulation1 = self.load_robot_into_scene(1, cen1, rot1)

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

        self.register_robot_articulations()

        self.teleport_robots_to_zeropos()

        self._obstacle = FixedCuboid("/World/obstacle",size=.05,position=np.array([0.4, 0.0, 1.65]),color=np.array([0.,0.,1.]))

        # Initialize RmpFlow objects
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

        self.set_stiffness_and_damping_for_all_joints(self._robcfg)

        self._rmpflow.set_ignore_state_updates(True)
        self._rmpflow.visualize_collision_spheres()
        self._articulation_rmpflow = ArticulationMotionPolicy(self._articulation,self._rmpflow)

        # Initialize RmpFlow1 objects
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

        self.set_stiffness_and_damping_for_all_joints(self._robcfg1)

        self._rmpflow1.set_ignore_state_updates(True)
        self._rmpflow1.visualize_collision_spheres()

        self._articulation_rmpflow1 = ArticulationMotionPolicy(self._articulation1,self._rmpflow1)

        # Camera
        _, campath = add_camera_to_robot(self._robcfg.robot_name, self._robcfg.robot_id, self._robcfg.robot_prim_path)
        self.add_camera_to_camlist(self._robcfg.robot_id, self._robcfg.robot_name, campath)
        _, campath = add_camera_to_robot(self._robcfg1.robot_name, self._robcfg1.robot_id, self._robcfg1.robot_prim_path)
        self.add_camera_to_camlist(self._robcfg1.robot_id, self._robcfg1.robot_name, campath)

        self._running_scenario = True

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
