import numpy as np
from pxr import Usd, UsdGeom, Gf, UsdPhysics, PhysxSchema

import omni
import carb

from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.objects.cuboid import FixedCuboid
from omni.isaac.core.prims import XFormPrim

from omni.isaac.core.utils.stage import get_current_stage
from omni.isaac.core.world import World

from omni.isaac.core.utils.extensions import get_extension_path_from_name
from omni.isaac.core.utils.rotations import euler_angles_to_quat
from omni.isaac.core.utils.viewports import set_camera_view

from .senut import apply_material_to_prim_and_children, GetXformOps, GetXformOpsFromPath
from .senut import add_rob_cam

from .scenario_base import ScenarioBase
from .senut import make_cam_view_window
from .senut import apply_convex_decomposition_to_mesh_and_children
from .senut import apply_collisionapis_to_mesh_and_children
from .senut import apply_diable_gravity_to_rigid_bodies
# Copyright (c) 2022-2023, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#


class CageRmpflowScenario(ScenarioBase):

    _running_scenario = False
    _colorScheme = "transparent"
    rotate_target0 = False
    rotate_target1 = False
    target_rot_speed = 2*np.pi/10 # 10 seconds for a full rotation
    cagecamviews = None

    def __init__(self):
        super().__init__()
        self._scenario_name = "cage-rmpflow"
        self._scenario_description = ScenarioBase.get_scenario_desc(self._scenario_name)
        self._nrobots = 2
        self._moto50mp_list = []
        self._moto_tray_list = []

    def AddMoto50mp(self, name, pos=[0,0,0],rot=[0,0,0],ska=[1,1,1]):
        idx = len(self._moto50mp_list)
        usdpath = f"/World/moto_50mp_{idx}"
        filepath_to_moto_50mp_usd = f"{self.current_extension_path}/usd/MOTO_50MP_v2fix.usda"
        add_reference_to_stage(filepath_to_moto_50mp_usd, usdpath)
        quat = euler_angles_to_quat(rot)
        self._moto = XFormPrim(usdpath, scale=ska, position=pos, orientation=quat )
        meth = UsdPhysics.Tokens.convexHull
        apply_collisionapis_to_mesh_and_children(self._stage, usdpath, method=meth)

        prim = self._stage.GetPrimAtPath(usdpath)
        UsdPhysics.RigidBodyAPI.Apply(prim)
        mapi = UsdPhysics.MassAPI.Apply(prim)
        mapi.CreateMassAttr(0.192) # g54 stats w=73.82 mm, h=161.56, d=8.89, pearl blue
        moto = {"usdpath":usdpath, "prim":prim, "idx":idx, "name":name}
        self._moto50mp_list.append(moto)

    def GetMoto50mpByIdx(self, idx):
        if idx>=len(self._moto50mp_list):
            carb.log_error(f"GetMoto50mpByIdx: idx {idx} out of range")
            return None
        return self._moto50mp_list[idx]

    def GetMoto50mpByName(self, name):
        for moto in self._moto50mp_list:
            if moto["name"] == name:
                return moto
        carb.log_error(f"GetMoto50mpByName: name {name} not found")
        return None

    def AddMotoTray(self, name, fillstr="000000", pos=[0,0,0],rot=[0,0,0],ska=[1.01,1.01,1.01]):
        idx = len(self._moto_tray_list)
        usdpath = f"/World/moto_tray_{idx}"
        filepath_to_moto_tray_usd = f"{self.current_extension_path}/usd/MOTO_TRAY_v2fix.usda"
        add_reference_to_stage(filepath_to_moto_tray_usd, usdpath)
        quat = euler_angles_to_quat(rot)
        self._moto = XFormPrim(usdpath, scale=ska, position=pos, orientation=quat )
        # Don't do body1 for now, all the options are too big to let the phone slip through
        #     it needs to be custom vertical and horizontal strips
        # meth = UsdPhysics.Tokens.boundingCube
        # meth = UsdPhysics.Tokens.convexHull
        # apply_collisionapis_to_mesh_and_children(self._stage, usdpath,
        #                                          filt_end_path=["Body1"],method=meth )
        # options are: boundingCube, convexHull, convexDecomposition and probably a few more
        meth = UsdPhysics.Tokens.convexDecomposition
        apply_collisionapis_to_mesh_and_children(self._stage, usdpath,
                                                 include=["Body2"],method=meth )

        prim = self._stage.GetPrimAtPath(usdpath)
        UsdPhysics.RigidBodyAPI.Apply(prim)
        mapi = UsdPhysics.MassAPI.Apply(prim)
        mapi.CreateMassAttr(0.2)
        # apply_diable_gravity_to_rigid_bodies(self._stage, usdpath)

        mototray = {"usdpath":usdpath, "prim":prim, "idx":idx, "name":name}
        self._moto_tray_list.append(mototray)

        while len(fillstr)<6:
            fillstr += "0"

        a90 = np.pi/2

        w = 0.07382
        h = 0.16156
        iw = 0 # 0,1,2  - corresponds to width of mp50 which is 0.07382 meters
        ih = 0 # 0,1    - corresponds to height of mp50 which is 0.16156 meters
        for c in fillstr:
            yp = (iw-2.5)*w + pos[0] + iw*0.01
            xp = (ih+0.0)*h + pos[1] + ih*0.01 + 0.015
            zp = 0.02 + pos[2]
            if c=="1":
                self.AddMoto50mp(f"{name}_moto{idx}",pos=[xp,yp,zp],rot=[-a90,0,a90],ska=[1,1,1])
            iw += 1
            if iw>2:
                iw  = 0
                ih += 1


    def GetMotoTrayByIdx(self, idx):
        if idx>=len(self._moto_tray_list):
            carb.log_error(f"GetMotoTrayByIdx: idx {idx} out of range")
            return None
        return self._moto_tray_list[idx]

    def GetMotoTrayByName(self, name):
        for moto in self._moto_tray_list:
            if moto["name"] == name:
                return moto
        carb.log_error(f"GetMotoTrayByName: name {name} not found")
        return None


    def add_cage(self):
        usdpath = "/World/cage_v1"
        self.current_extension_path = get_extension_path_from_name("JakaControl")
        # cagevariant = "cage_with_static_colliders"
        cagevariant = "cage_v1"
        if cagevariant == "cage_v1":
            filepath_to_cage_usd = f"{self.current_extension_path}/usd/cage_v1.usda"
            self._cage = XFormPrim(usdpath, scale=[1,1,1], position=[0,0,0])
        else:
            filepath_to_cage_usd = f"{self.current_extension_path}/usd/cage_with_static_colliders.usda"
            sz = 0.0254
            quat = euler_angles_to_quat([np.pi/2,0,0])
            self._cage = XFormPrim(usdpath, scale=[sz,sz,sz], position=[0,0,0], orientation=quat)

        add_reference_to_stage(filepath_to_cage_usd, usdpath)

        # adjust collision shapes
        if cagevariant == "cage_v1":
            meth = UsdPhysics.Tokens.convexHull
            apply_collisionapis_to_mesh_and_children(self._stage, usdpath, method=meth )
        else:
            ppath1 = "ACRYLIC___FIXTURE_V1_v8_1/ACRYLIC___FIXTURE_V1_v8/Body1/Body1"
            ppath2 = "ACRYLIC___FIXTURE_V1_v8_2/ACRYLIC___FIXTURE_V1_v8/Body1/Body1"
            meth = UsdPhysics.Tokens.convexHull
            apply_collisionapis_to_mesh_and_children(self._stage, usdpath, include=[ppath1,ppath2],method=meth )


        if self._colorScheme == "default":
            self._cage.set_color([0.5, 0.5, 0.5, 1.0])
        elif self._colorScheme == "transparent":
            apply_material_to_prim_and_children(self._stage, self._matman, "Steel_Blued", usdpath)
        self.cagepath = usdpath




    def load_scenario(self, robot_name, ground_opt, light_opt="dome_light"):
        super().load_scenario(robot_name, ground_opt)

        self.create_robot_config(robot_name, "/World/roborg0")
        self.create_robot_config(robot_name, "/World/roborg1")

        self.add_light(light_opt)
        self.add_ground(ground_opt)

        # Robots
        # (pos0, rot0) = ([0.14, 0, 0.77], [0, 150, 180])
        # this seems to be related to robot_rotvek
        # used in set_robot_base_pose in scenario_base self.make_rmpflow()
        # the latter uses a quaternion so we really should just be using that
        #
        # This works but we have no good way to rotate the robot arm around its long axis
        # order = "XYZ"
        # (pos0, rot0) = ([0.14, 0, 0.77], [0, -150, 180])
        # self.load_robot_into_scene(0, pos0, rot0, order=order)


        # (pos1, rot1) = ([-0.08, 0, 0.77], [0, -150, 0])
        # self.load_robot_into_scene(1, pos1, rot1, order=order)

        # This works but it is oriented the same as the other robot which is wrong
        # order = "ZYX"
        # (pos0, rot0) = ([0.14, 0, 0.77], [0, 150, 0])
        # self.load_robot_into_scene(0, pos0, rot0, order=order)


        # (pos1, rot1) = ([-0.08, 0, 0.77], [0, -150, 0])
        # self.load_robot_into_scene(1, pos1, rot1, order=order)

        order = "ZYX"
        (pos0, rot0) = ([0.14, 0, 0.77], [0, 150, 0])
        self.load_robot_into_scene(0, pos0, rot0, order=order)


        (pos1, rot1) = ([-0.08, 0, 0.77], [0, -150, 0])
        self.load_robot_into_scene(1, pos1, rot1, order=order)

        self.add_cameras_to_robots()

        # tagets
        quat = euler_angles_to_quat([-np.pi/2,0,0])
        t0path = "/World/target0"
        self._target0 = XFormPrim(t0path, scale=[.04,.04,.04], position=[0.15, 0.00, 0.02], orientation=quat)
        (self.targ0top,_,_,_) = GetXformOpsFromPath(t0path)
        add_reference_to_stage(get_assets_root_path() + "/Isaac/Props/UIElements/frame_prim.usd", t0path)

        quat = euler_angles_to_quat([-np.pi/2,0,0])
        t1path = "/World/target1"
        self._target1 = XFormPrim(t1path, scale=[.04,.04,.04], position=[-0.15, 0.00, 0.02], orientation=quat)
        (self.targ1top,_,_,_) = GetXformOpsFromPath(t1path)
        add_reference_to_stage(get_assets_root_path() + "/Isaac/Props/UIElements/frame_prim.usd", t1path)

        # obstacles
        self._obstacle = FixedCuboid("/World/obstacle",size=.05,position=np.array([0.4, 0.0, 1.65]),color=np.array([0.,0.,1.]))

        # cage
        self.add_cage()

        a90 = np.pi/2

        # moto_50mp
        self.AddMoto50mp("moto1",rot=[-a90,0,a90],pos=[0,0,0.1])
        self.AddMoto50mp("moto2",rot=[-a90,0,a90],pos=[0.1,0.1,0.1])

        # moto_tray
        self.AddMotoTray("tray1", "111111", rot=[a90,0,0],pos=[0.35,0.25,0.0])



    def setup_scenario(self):
        self.register_robot_articulations()
        self.adjust_stiffness_and_damping_for_robots()
        self.teleport_robots_to_zeropos()

        self.make_robot_mpflows([self._obstacle])

        set_camera_view(eye=[0.0, 2.5, 1.0], target=[0,0,0], camera_prim_path="/OmniverseKit_Persp")

        self._running_scenario = True

    def reset_scenario(self):
        self.reset_robot_rmpflows()

    gang = 0
    def rotate_target(self, target, top, cen, radius, step_size):
        # pos, ori = target.get_world_pose()
        cen = np.array(cen)
        (xp,yp,zp) = cen
        # newpos = np.array([radius*np.cos(ang), radius*np.sin(ang), zp])
        self.gang += self.target_rot_speed*step_size
        newpos = Gf.Vec3d([xp+radius*np.cos(self.gang), yp+radius*np.sin(self.gang), zp])
        top.Set(newpos)

    def physics_step(self, step_size):
        self.global_time += step_size

        if self.rmpactive:
            self.rmpflow_update_world_for_all()

        if self.rotate_target0:
            self.rotate_target(self._target0, self.targ0top, [+0.3, 0.00, 0.02], 0.15, step_size)
        if self.rotate_target1:
            self.rotate_target(self._target1, self.targ1top,  [-0.3, 0.00, 0.02], 0.15, step_size)

        target0_position, target0_orientation = self._target0.get_world_pose()
        target1_position, target1_orientation = self._target1.get_world_pose()

        if self.rmpactive:
            self.set_end_effector_target_for_robot(0, target0_position, target0_orientation)
            self.set_end_effector_target_for_robot(1, target1_position, target1_orientation)
            self.forward_rmpflow_step_for_robots(step_size)

    def update_scenario(self, step: float):
        if not self._running_scenario:
            return
        self.physics_step(step)

    cagecamlist = {}
    def add_camera_to_cagecamlist(self, cam_name, cam_display_name, campath):
        self.cagecamlist[cam_name] = {}
        self.cagecamlist[cam_name]["name"] = cam_name
        self.cagecamlist[cam_name]["display_name"] = cam_display_name
        self.cagecamlist[cam_name]["usdpath"] = campath

    def add_1_ccam(self, cam_root, cam_name, cam_display_name, cam_ring_rot, cam_mount, cam_pt_quat):
        cam_root = f"{cam_root}/{cam_name}"
        _, campath = add_rob_cam(cam_root, cam_ring_rot, cam_mount, cam_pt_quat, cam_name)
        self.add_camera_to_cagecamlist(cam_name, cam_display_name, campath)

    def make_cage_cameras(self):
        cagepath = "/World/cage_v1"
        cage = self._stage.GetPrimAtPath(cagepath)
        if cage:
            cc_rr = Gf.Vec3f([0.0, 0, 0.0])
            cc_pt = Gf.Quatf(1, Gf.Vec3f([0,1,0]))
            self.add_1_ccam(cagepath, "cage_cam_0", "Cage Cam 0", cc_rr, Gf.Vec3f([+0.559,+0.388,0.794]), cc_pt)
            self.add_1_ccam(cagepath, "cage_cam_1", "Cage Cam 1", cc_rr, Gf.Vec3f([-0.559,+0.388,0.794]), cc_pt)
            self.add_1_ccam(cagepath, "cage_cam_2", "Cage Cam 0", cc_rr, Gf.Vec3f([+0.559,-0.388,0.794]), cc_pt)
            self.add_1_ccam(cagepath, "cage_cam_3", "Cage Cam 1", cc_rr, Gf.Vec3f([-0.559,-0.388,0.794]), cc_pt)
            # _, campath = add_rob_cam(cc_path, cc_ring_rot, cc_mount, cc_pt_quat)
            # self.add_camera_to_cagecamlist(cc_name, cc_display_name, campath)


    def make_cage_cam_views(self):
        if self.cagecamviews is not None:
            self.cagecamviews.destroy()
            self.cagecamviews = None
        wintitle = "Cage Cameras"
        wid = 1280
        heit = 720
        self.cagecamviews = make_cam_view_window(self.cagecamlist, wintitle, wid, heit)
        self.cage_wintitle = wintitle


    def scenario_action(self, action_name, action_args):
        if action_name in self.base_actions:
            rv = super().scenario_action(action_name, action_args)
            return rv
        match action_name:
            case "RotateTarget0":
                self.rotate_target0 = not self.rotate_target0
            case "RotateTarget1":
                self.rotate_target1 = not self.rotate_target1
            case "ChangeSpeed":
                m = action_args.get("m",0)
                b = action_args.get("b",0)
                if m!=0:
                    self.target_rot_speed *= -1
                else:
                    if b>0:
                        self.target_rot_speed /= 2
                    else:
                        self.target_rot_speed *= 2
            case "CageCamViews":
                self.make_cage_cameras()
                self.make_cage_cam_views()
            case _:
                print(f"Action {action_name} not implemented")
                return False

    def get_action_button_text(self, action_name, action_args=None):
        if action_name in self.base_actions:
            rv = super().get_action_button_text(action_name, action_args)
            return rv
        match action_name:
            case "RotateTarget0":
                rv = "Rotate Target 0"
            case "RotateTarget1":
                rv = "Rotate Target 1"
            case "ChangeSpeed":
                rv = f"Change Speed {self.target_rot_speed:.1f}"
            case "CageCamViews":
                rv = "Cage Cam Views"
            case _:
                rv = f"{action_name} TBD"
        return rv

    def get_action_button_tooltip(self, action_name, action_args=None):
        if action_name in self.base_actions:
            rv = super().get_action_button_tooltip(action_name, action_args)
            return rv
        match action_name:
            case "ChangeSpeed":
                rv = f"L*2,R /2, Ctrl to reverse"
            case _:
                rv = f"No tooltip for action {action_name}"
        return rv


    def get_scenario_actions(self):
        self.base_actions = super().get_scenario_actions()
        combo  = self.base_actions + ["RotateTarget0", "RotateTarget1",
                                      "ChangeSpeed",
                                      "CageCamViews"]
        return combo
