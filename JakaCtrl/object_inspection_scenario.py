import math
import numpy as np
import os
from pxr import Usd, UsdGeom, Gf
import omni
import omni.ui as ui

from omni.kit.widget.viewport import ViewportWidget

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

from omni.asimov.jaka.minicobo import Minicobo

from .senut import add_light_to_stage, add_dome_light_to_stage
from .senut import calc_robot_circle_pose
from .senut import apply_convex_decomposition_to_mesh_and_children, apply_material_to_prim_and_children
from .senut import apply_diable_gravity_to_rigid_bodies, adjust_articulation

from .senut import adjust_joint_values, set_stiffness_for_joints, set_damping_for_joints
from .senut import add_camera_to_robot

from omni.isaac.motion_generation.lula.interface_helper import LulaInterfaceHelper

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
    _show_collision_bounds = True
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
        pos, rot = self.calc_oi_robot_pose(cen=cen, rad=rad, xang=xang, yang=yang, zang=zang)
        self._start_robot_pos = pos
        self._start_robot_rot = rot
        self.robot_rotvek = np.array(rot)*np.pi/180

        roborg = UsdGeom.Xform.Define(stage,"/World/roborg")
        self._rob_tranop = roborg.AddTranslateOp()
        self._rob_zrotop = roborg.AddRotateZOp()
        self._rob_yrotop = roborg.AddRotateYOp()
        self._rob_xrotop = roborg.AddRotateXOp()

        self.set_oi_circle_pose(self._start_robot_pos, self._start_robot_rot)

        add_reference_to_stage(self._robcfg.robot_usd_file_path, self._robcfg.robot_prim_path)
        apply_convex_decomposition_to_mesh_and_children(stage, self._robcfg.robot_prim_path)
        adjust_articulation(stage, self._robcfg.robot_prim_path)
        apply_diable_gravity_to_rigid_bodies(stage, self._robcfg.robot_prim_path)

        self._articulation = Articulation(self._robcfg.artpath,"mico-0")

        # Robot 1
        cen1, rad1 = [0.14, 0, 0.77], 0
        xang1, yang1, zang1 = 0, 150, 0
        pos1, rot1 = self.calc_oi_robot_pose(cen=cen1, rad=rad1, xang=xang1, yang=yang1, zang=zang1)
        self._start_robot_pos1 = pos1
        self._start_robot_rot1 = rot1
        self.robot1_rotvek = np.array(rot1)*np.pi/180

        roborg1 = UsdGeom.Xform.Define(stage,"/World/roborg1")
        self._rob_tranop1 = roborg1.AddTranslateOp()
        self._rob_zrotop1 = roborg1.AddRotateZOp()
        self._rob_yrotop1 = roborg1.AddRotateYOp()
        self._rob_xrotop1 = roborg1.AddRotateXOp()
        self.set_oi_robot_pose1(self._start_robot_pos1, self._start_robot_rot1)

        self._robcfg1.artpath = self._robcfg1.artpath.replace("roborg", "roborg1")
        self._robcfg1.robot_prim_path = self._robcfg1.robot_prim_path.replace("roborg", "roborg1")

        add_reference_to_stage(self._robcfg1.robot_usd_file_path, self._robcfg1.robot_prim_path)
        apply_convex_decomposition_to_mesh_and_children(stage, self._robcfg1.robot_prim_path)
        apply_diable_gravity_to_rigid_bodies(stage, self._robcfg1.robot_prim_path)
        adjust_articulation(stage, self._robcfg1.robot_prim_path)
        self._articulation1 = Articulation(self._robcfg1.artpath,"mico-1")


        # apply_material_to_prim_and_children(stage, self._matman, "Red_Glass", self._robcfg.robot_prim_path)
        # apply_material_to_prim_and_children(stage, self._matman, "Green_Glass", self._robcfg1.robot_prim_path)

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
        # path_to_cage_usd = f"{jakacontrol_extension_path}/usd/FlexBenchV1.usda"
        add_reference_to_stage(path_to_cage_usd, "/World/cage_v1")
        # self._cage = XFormPrim("/World/cage_v1", scale=[1,1,1], position=[-0.38605,0,-0.00045])
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
        # articulation.set_joint_positions(self._lower_joint_limits + epsilon)
        self._articulation.set_joint_positions(self._zeros + epsilon)
        self._articulation1.set_joint_positions(self._zeros + epsilon)

        self._obstacle = FixedCuboid("/World/obstacle",size=.05,position=np.array([0.4, 0.0, 1.65]),color=np.array([0.,0.,1.]))

        print("rdf_path:",self._robcfg.rdf_path)
        print("urdf_path:",self._robcfg.urdf_path)
        print("rmp_config_path:",self._robcfg.rmp_config_path)
        print("eeframe_name:",self._robcfg.eeframe_name)
        print("max_step_size:",self._robcfg.max_step_size)
        # Initialize an RmpFlow object
        self._rmpflow = RmpFlow(
            robot_description_path = self._robcfg.rdf_path,
            urdf_path = self._robcfg.urdf_path,
            rmpflow_config_path = self._robcfg.rmp_config_path,
            end_effector_frame_name = self._robcfg.eeframe_name,
            maximum_substep_size = self._robcfg.max_step_size
        )
        quat = euler_angles_to_quat(self.robot_rotvek)
        self._rmpflow.set_robot_base_pose(self._start_robot_pos, quat)
        self._rmpflow.add_obstacle(self._obstacle)
        # self._rmpflow.add_obstacle(self._cage)

        self._rmpflow1 = RmpFlow(
            robot_description_path = self._robcfg1.rdf_path,
            urdf_path = self._robcfg1.urdf_path,
            rmpflow_config_path = self._robcfg1.rmp_config_path,
            end_effector_frame_name = self._robcfg1.eeframe_name,
            maximum_substep_size = self._robcfg1.max_step_size
        )
        quat1 = euler_angles_to_quat(self.robot1_rotvek)
        self._rmpflow1.set_robot_base_pose(self._start_robot_pos1, quat1)
        self._rmpflow1.add_obstacle(self._obstacle)
        # self._rmpflow1.add_obstacle(self._cage)


        self.lulaHelper = LulaInterfaceHelper(self._rmpflow._robot_description)
        if self._robcfg.stiffness>0:
            self.set_stiffness_for_all_joints(self._robcfg.stiffness)

        if self._robcfg.damping>0:
            self.set_damping_for_all_joints(self._robcfg.damping)

        if self._robcfg1.stiffness>0:
            self.set_stiffness_for_all_joints(self._robcfg1.stiffness)

        if self._robcfg1.damping>0:
            self.set_damping_for_all_joints(self._robcfg1.damping)

        _, campath = add_camera_to_robot(self._robcfg.robot_name, self._robcfg.robot_id, self._robcfg.robot_prim_path)
        self.add_camera_to_camlist(self._robcfg.robot_id, self._robcfg.robot_name, campath)
        _, campath = add_camera_to_robot(self._robcfg1.robot_name, self._robcfg1.robot_id, self._robcfg1.robot_prim_path)
        self.add_camera_to_camlist(self._robcfg1.robot_id, self._robcfg1.robot_name, campath)

        if self._show_collision_bounds:
            self._rmpflow.set_ignore_state_updates(True)
            self._rmpflow.visualize_collision_spheres()
            self._rmpflow1.set_ignore_state_updates(True)
            self._rmpflow1.visualize_collision_spheres()

            # Set the robot gains to be deliberately poor
            bad_proportional_gains = self._articulation.get_articulation_controller().get_gains()[0]/50
            self._articulation.get_articulation_controller().set_gains(kps = bad_proportional_gains)

        print("Created _rmpflow object")

        self._articulation_rmpflow = ArticulationMotionPolicy(self._articulation,self._rmpflow)
        self._articulation_rmpflow1 = ArticulationMotionPolicy(self._articulation1,self._rmpflow1)

        # self._target.set_world_pose(np.array([.5,0,.7]),euler_angles_to_quats([0,np.pi,0]))

    def calc_oi_robot_pose(self, cen=[0, 0, 0.85], rad=0.35, xang=0, yang=130, zang=0):
        rads = np.pi
        pos = cen + rad*np.array([np.cos(rads), np.sin(rads), 0])
        pos = Gf.Vec3d(list(pos))
        rot = [xang, yang, zang]
        # print("pos:",pos," rot:",rot)
        return pos, rot

    def set_oi_circle_pose(self, pos, rot):
        self._rob_tranop.Set(pos)
        self._rob_zrotop.Set(rot[2])
        self._rob_yrotop.Set(rot[1])
        self._rob_xrotop.Set(rot[0])
        self._start_robot_pos = pos
        self._start_robot_rot = rot


    def set_oi_robot_pose1(self, pos, rot):
        self._rob_tranop1.Set(pos)
        self._rob_zrotop1.Set(rot[2])
        self._rob_yrotop1.Set(rot[1])
        self._rob_xrotop1.Set(rot[0])
        self._start_robot_pos1 = pos
        self._start_robot_rot1 = rot

    def post_load_scenario(self):
        self._world.add_physics_callback("sim_step", callback_fn=self.physics_step)

        # if self.robcfg.stiffness>0:
        #     self.set_stiffness_for_all_joints(self._cfg_stiffness) # 1e8 or 10 million seems too high

        # if self.robcfg.damping>0:
        #     self.set_damping_for_all_joints(self.robcfg1.amping) # 1e5 or 100 thousand seems too high

        pass

    def reset_scenario(self):
        # self._target.set_world_pose(np.array([.5,0,.7]),euler_angles_to_quats([0,np.pi,0]))

        # self._rmpflow.reset()
        # self._rmpflow1.reset()
        if self._show_collision_bounds:
            self._rmpflow.visualize_collision_spheres()
            self._rmpflow.visualize_end_effector_position()
            self._rmpflow1.visualize_collision_spheres()
            self._rmpflow1.visualize_end_effector_position()

    def set_stiffness_for_all_joints(self, stiffness):
        joint_names = self.lulaHelper.get_active_joints()
        set_stiffness_for_joints(joint_names, stiffness)

    def set_damping_for_all_joints(self, damping):
        joint_names = self.lulaHelper.get_active_joints()
        set_damping_for_joints(joint_names, damping)

    def physics_step(self, step_size):
        target_position, target_orientation = self._target.get_world_pose()
        target1_position, target1_orientation = self._target1.get_world_pose()

        self._rmpflow.update_world()
        self._rmpflow1.update_world()


        self._rmpflow.set_end_effector_target(
            target_position, target_orientation
        )

        self._rmpflow1.set_end_effector_target(
             target1_position, target1_orientation
        )

        action = self._articulation_rmpflow.get_next_articulation_action(step_size)
        self._articulation.apply_action(action)
        action1 = self._articulation_rmpflow1.get_next_articulation_action(step_size)
        self._articulation1.apply_action(action1)

    def teardown_scenario(self):
        pass

    def update_scenario(self, step: float):
        if not self._running_scenario:
            return

    def make_camera_views(self):
        # https://docs.omniverse.nvidia.com/kit/docs/omni.kit.viewport.docs/latest/overview.html
        if hasattr(self, "camviews") and self.camviews is not None:
            self.camviews.destroy()
            self.camviews = None
        wintitle = "Robot Cameras"
        wid = 1280
        heit = 720
        ncam = len(self.camlist)
        camviews = omni.ui.Window(wintitle, width=wid, height=heit+20) # Add 20 for the title-bar

        with camviews.frame:
            if ncam==0:
                ui.Label("No Cameras Found (camlist is empty)")
            else:
                with ui.VStack():
                    vh = heit / len(self.camlist)
                    for camname in self.camlist:
                        cam = self.camlist[camname]
                        viewport_widget = ViewportWidget(resolution = (wid, vh))

                        # Control of the ViewportTexture happens through the object held in the viewport_api property
                        viewport_api = viewport_widget.viewport_api

                        # We can reduce the resolution of the render easily
                        viewport_api.resolution = (wid, vh)

                        # We can also switch to a different camera if we know the path to one that exists
                        viewport_api.camera_path = cam["usdpath"]


        # from functools import partial
        # ui.Workspace.set_show_window_fn(wintitle, partial(ui.Workspace.show_window, wintitle))

        # # Add a Menu Item for the window
        # editor_menu = omni.kit.ui.get_editor_menu()
        # if editor_menu:
        #     self._menu = editor_menu.add_item(
        #         "CamViews", ui.Workspace.show_window, toggle=True, value=True
        #     )
        self.camviews = camviews
        return wintitle

    def scenario_action(self, action_name, action_args):
        if action_name in self.base_actions:
            rv = super().scenario_action(action_name, action_args)
            return rv
        # match action_name:
        #     case "Camera Viewports":
        #         if not hasattr(self, "camlist"):
        #             return
        #         if len(self.camlist)==0:
        #             return
        #         self.wtit = self.make_camera_views()
        #         # ui.Workspace.show_window(wtit,True)
        #         pass
        #     case "Show CV Window":
        #         ui.Workspace.show_window(self.wtit,True)
        #         pass

    def get_scenario_actions(self):
        self.base_actions = super().get_scenario_actions()
        combo  = self.base_actions + []
        return combo
