import numpy as np
from pxr import Usd, UsdGeom, Gf, UsdPhysics, PhysxSchema
from types import SimpleNamespace
import time
import omni
import carb
import websockets
import asyncio
from datetime import datetime

from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.objects.cuboid import FixedCuboid
from omni.isaac.core.prims import XFormPrim


from omni.isaac.core.utils.stage import get_current_stage
from omni.isaac.core.world import World

from omni.isaac.core.utils.rotations import euler_angles_to_quat
from omni.isaac.core.utils.viewports import set_camera_view

from .senut import apply_material_to_prim_and_children, GetXformOps, GetXformOpsFromPath
from .senut import add_rob_cam

from .scenario_base import ScenarioBase
from .senut import make_rob_cam_view_window


from .motomod import MotoMan
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
    websocketWS = websockets
    open = False
    async def Connect2WebsocketWS(self):
        self.websocketWS =  await websockets.connect("ws://imvplaygroundsockets.azurewebsites.net/8888", ping_interval=None)
        await self.websocketWS.send("OPEN")
        open = True
    

    
    async def send_positions_by_websocket(self,mess):
        try:
            await self.websocketWS.send(mess)
        except websockets.ConnectionClosedError:
            await self.Connect2WebsocketWS()


    def __init__(self, uibuilder=None):
        super().__init__()
        self._scenario_name = "cage-rmpflow"
        self._scenario_description = ScenarioBase.get_scenario_desc(self._scenario_name)
        self._nrobots = 2
        self.uibuilder = uibuilder
        self.current_robot_action = "FollowTarget"
        self.start = time.process_time()

    def load_scenario(self, robot_name, ground_opt, light_opt="dome_light"):
        super().load_scenario(robot_name, ground_opt)

        self.create_robot_config(robot_name, "/World/roborg0")
        self.create_robot_config(robot_name, "/World/roborg1")

        self.add_light(light_opt)
        self.add_ground(ground_opt)

        # Robots
        # (pos0, rot0) = ([0.14, 0, 0.77], [0, 150, 180])
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

        order = "XYZ"
        #right arm
        pre_rot = [0,0,0]
        (pos0, rot0) = ([0.14, 0, 0.77], [0, -150, 180])
        self.load_robot_into_scene(0, pos0, rot0, order=order, pre_rot=pre_rot)

        #left arm
        pre_rot = [0,0,0]
        (pos1, rot1) = ([-0.08, 0, 0.77], [0, 150, 180])
        self.load_robot_into_scene(1, pos1, rot1, order=order, pre_rot=pre_rot)

        self.add_cameras_to_robots()

        # tagets
        quat = euler_angles_to_quat([-np.pi/2,0,0])
        t0path = "/World/target0"
        self._target0 = XFormPrim(t0path, scale=[.04,.04,.04], position=[0.15, 0.00, 0.02], orientation=quat)
        (self.targ0top,_,_,_) = GetXformOpsFromPath(t0path)
        add_reference_to_stage(get_assets_root_path() + "/Isaac/Props/UIElements/frame_prim.usd", t0path)

        quat = euler_angles_to_quat([-np.pi/2,0,np.pi])
        t1path = "/World/target1"
        self._target1 = XFormPrim(t1path, scale=[.04,.04,.04], position=[-0.15, 0.00, 0.02], orientation=quat)
        (self.targ1top,_,_,_) = GetXformOpsFromPath(t1path)
        add_reference_to_stage(get_assets_root_path() + "/Isaac/Props/UIElements/frame_prim.usd", t1path)

        # obstacles
        self._obstacle = FixedCuboid("/World/obstacle",size=.05,position=np.array([0.4, 0.0, 1.65]),color=np.array([0.,0.,1.]))

        mm = MotoMan(self._stage, self._matman)
        # cage
        mm.AddCage()

        a90 = np.pi/2

        # moto_50mp
        # mm.AddMoto50mp("moto1",rot=[-a90,0,a90],pos=[0,0,0.1])
        # mm.AddMoto50mp("moto2",rot=[-a90,0,a90],pos=[0.1,0.1,0.1])

        # moto_tray
        zang = 5*np.pi/4
        zang = np.pi/2
        zang = 0
        xoff = 0.20
        yoff = 0.15
        self.mototray1 = mm.AddMotoTray("tray1", "rgb000", rot=[a90,0,zang],pos=[+xoff,+yoff,0.0])
        self.mototray2 = mm.AddMotoTray("tray2", "000000", rot=[a90,0,zang],pos=[-xoff,+yoff,0.0])
        self.mototray3 = mm.AddMotoTray("tray3", "myc000", rot=[a90,0,zang],pos=[-xoff,-yoff,0.0])
        self.mototray4 = mm.AddMotoTray("tray4", "000000", rot=[a90,0,zang],pos=[+xoff,-yoff,0.0])
        asyncio.ensure_future(self.Connect2WebsocketWS())

    def add_grippers_to_robots(self):
        for i in range(self._nrobots):
            rcfg = self.get_robot_config(i)
            rcfg.gripper = self.get_or_create_gripper(i)

    def setup_scenario(self):
        self.register_robot_articulations()
        self.adjust_stiffness_and_damping_for_robots()
        self.teleport_robots_to_zeropos()

        self.make_robot_mpflows([self._obstacle])

        self.add_grippers_to_robots()
        self.add_pp_controllers_to_robots()

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

    def physics_step_old(self, step_size):
        self.global_time += step_size

        if self.rmpactive:
            self.rmpflow_update_world_for_all()

        if self.rotate_target0:
            self.rotate_target(self._target0, self.targ0top, [+0.3, 0.00, 0.02], 0.15, step_size)
        if self.rotate_target1:
            self.rotate_target(self._target1, self.targ1top, [-0.3, 0.00, 0.02], 0.15, step_size)

        target0_position, target0_orientation = self._target0.get_world_pose()
        target1_position, target1_orientation = self._target1.get_world_pose()

        if self.rmpactive:
            self.set_end_effector_target_for_robot(0, target0_position, target0_orientation)
            self.set_end_effector_target_for_robot(1, target1_position, target1_orientation)
            self.forward_rmpflow_step_for_robots(step_size)

    lasttime = 0
    
    def physics_step(self, step_size):
        self.global_time += step_size

        if self.rotate_target0:
            self.rotate_target(self._target0, self.targ0top, [+0.3, 0.00, 0.02], 0.15, step_size)
        if self.rotate_target1:
            self.rotate_target(self._target1, self.targ1top, [-0.3, 0.00, 0.02], 0.15, step_size)

        for i in range(self._nrobots):
            rcfg = self.get_robot_config(i)

            if self.rmpactive:
                if rcfg.current_robot_action == "FollowTarget":
                    rcfg.rmpflow.update_world()

                    if i==0:
                        targ_pos, targ_ori = self._target0.get_world_pose()
                    elif i==1:
                        targ_pos, targ_ori = self._target1.get_world_pose()

                    self.set_end_effector_target_for_robot(i, targ_pos, targ_ori)
                    action = rcfg.articulation_rmpflow.get_next_articulation_action(step_size)
                    rcfg._articulation.apply_action(action)
                elif rcfg.current_robot_action == "PickAndPlace":
                    eeoff = np.array([0,0,-0.01])
                    cp, _ = rcfg.pickobj.get_world_pose()
                    moto_pos = np.array([cp[0],cp[1],cp[2]+0.013])
                    current_joint_positions = rcfg._articulation.get_joint_positions()
                    args = dict(
                        picking_position=moto_pos,
                        placing_position=rcfg.targpos,
                        current_joint_positions=current_joint_positions,
                        end_effector_offset=eeoff,
                        end_effector_orientation=rcfg.grip_eeori
                    )
                    actions = rcfg._controller.forward(**args)
                    rcfg._articulation.apply_action(actions)
                    # for debugging only:
                    ee_pos, ee_rot = rcfg._articulation_kinematics_solver.compute_end_effector_pose()

                    if rcfg._controller.is_done() and rcfg.pickobj is not None:
                        self.finish_pick_and_place(i)
                        rcfg._controller.reset()
                        rcfg.current_robot_action = "None"

                    elap = self.global_time - self.lasttime
                    if elap>0.5:
                        print(f"ee_pos:{ee_pos}  phone:{cp}  targpos:{rcfg.targpos}  elap:{elap:.2f}")
                        self.lasttime = self.global_time
                elif rcfg.current_robot_action == "MoveToZero":
                    action = SimpleNamespace()
                    action.joint_indices = [0,1,2,3,4,5]
                    action.joint_positions = rcfg.dof_zero_pos
                    action.joint_velocities = None
                    action.joint_efforts = None
                    rcfg._articulation.apply_action(action)

        rcfg0 = self.get_robot_config(0)
        current_joint_positions0 = rcfg0._articulation.get_joint_positions()

        rcfg1 = self.get_robot_config(1)
        current_joint_positions1 = rcfg1._articulation.get_joint_positions()

     
        elapsed_time = time.process_time() - self.start
        #print(elapsed_time)
        if(elapsed_time > 1):   
            strs = ','.join(str(x) for x in (current_joint_positions0))
            print("joints:",strs)
            current_time_str = datetime.now().strftime ('%Y-%m-%d %H:%M:%S')
            valRight = "[R,"+ current_time_str + ", "+ ", ".join(map(str, current_joint_positions0)) + "]"        
            asyncio.ensure_future(self.send_positions_by_websocket(valRight))
            valLeft = "[L,"+ current_time_str + ", "+ ", ".join(map(str, current_joint_positions1)) + "]" 
            asyncio.ensure_future(self.send_positions_by_websocket(valLeft))
            self.start = time.process_time()
  


    
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
            cx = 0.559
            cy = 0.388
            cz = 0.794
            self.add_1_ccam(cagepath, "cage_cam_0", "Cage Cam 0", cc_rr, Gf.Vec3f([+cx,+cy,cz]), cc_pt)
            self.add_1_ccam(cagepath, "cage_cam_1", "Cage Cam 1", cc_rr, Gf.Vec3f([-cx,+cy,cz]), cc_pt)
            self.add_1_ccam(cagepath, "cage_cam_2", "Cage Cam 0", cc_rr, Gf.Vec3f([+cx,-cy,cz]), cc_pt)
            self.add_1_ccam(cagepath, "cage_cam_3", "Cage Cam 1", cc_rr, Gf.Vec3f([-cx,-cy,cz]), cc_pt)
            # _, campath = add_rob_cam(cc_path, cc_ring_rot, cc_mount, cc_pt_quat)
            # self.add_camera_to_cagecamlist(cc_name, cc_display_name, campath)

    def make_cage_cam_views(self):
        if self.cagecamviews is not None:
            self.cagecamviews.destroy()
            self.cagecamviews = None
        wintitle = "Cage Cameras"
        wid = 1280
        heit = 720
        self.cagecamviews = make_rob_cam_view_window(self.cagecamlist, wintitle, wid, heit)
        self.cage_wintitle = wintitle


    def scenario_action(self, action_name, action_args):
        if action_name in self.base_scenario_actions:
            rv = super().scenario_action(action_name, action_args)
            return rv
        match action_name:
            case "RotateRmp":
                self.rmpactive = not self.rmpactive
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

    def get_scenario_action_button_text(self, action_name, action_args=None):
        if action_name in self.base_scenario_actions:
            rv = super().get_scenario_action_button_text(action_name, action_args)
            return rv
        match action_name:
            case "RotateRmp":
                if self.rmpactive:
                    rv = "Stop RMP"
                else:
                    rv = "Start RMP"
            case "RotateTarget0":
                rv = "Rotate Target 0"
            case "RotateTarget1":
                rv = "Rotate Target 1"
            case "ChangeSpeed":
                rv = f"Change Speed {self.target_rot_speed:.1f}"
            case "CageCamViews":
                rv = "Cage Cam Views"
            case _:
                rv = f"{action_name}"
        return rv

    def get_scenario_action_button_tooltip(self, action_name, action_args=None):
        if action_name in self.base_scenario_actions:
            rv = super().get_scenario_action_button_tooltip(action_name, action_args)
            return rv
        match action_name:
            case "ChangeSpeed":
                rv = f"L*2,R /2, Ctrl to reverse"
            case _:
                rv = f"No tooltip for action {action_name}"
        return rv

    def get_scenario_actions(self):
        self.base_scenario_actions = super().get_scenario_actions()
        combo  = self.base_scenario_actions + ["RotateRmp","RotateTarget0", "RotateTarget1",
                                      "ChangeSpeed","CageCamViews"]
        return combo

    def robot_action(self, action_name, action_args):
        if action_name in self.base_scenario_actions:
            rv = super().robot_action(action_name, action_args)
            return rv
        match action_name:
            case "FollowTarget 0":
                rcfg = self.get_robot_config(0)
                rcfg.current_robot_action = "FollowTarget"
            case "FollowTarget 1":
                rcfg = self.get_robot_config(1)
                rcfg.current_robot_action = "FollowTarget"
            case "PickAndPlace 0":
                rcfg = self.get_robot_config(0)
                rcfg.current_robot_action = "PickAndPlace"
                self.setup_pick_and_place(0)
            case "PickAndPlace 1":
                rcfg = self.get_robot_config(1)
                rcfg.current_robot_action = "PickAndPlace"
                self.setup_pick_and_place(1)
            case "MoveToZero 0":
                rcfg = self.get_robot_config(0)
                rcfg.current_robot_action = "MoveToZero"
            case "MoveToZero 1":
                rcfg = self.get_robot_config(1)
                rcfg.current_robot_action = "MoveToZero"
            case _:
                print(f"Action {action_name} not implemented")
                return False

    def finish_pick_and_place(self, robot_id):
        rcfg = self.get_robot_config(robot_id)
        rcfg.sourcetray.empty_slot(rcfg.sourceidx)
        rcfg.targettray.fill_slot(rcfg.targetidx, rcfg.pickobj)
        rcfg.pickobj = None
        rcfg.pickpos = None
        rcfg.pickori = None
        rcfg.targpos = None
        rcfg.targori = None
        rcfg.sourcetray = None
        rcfg.sourceidx = None
        rcfg.targettray = None
        rcfg.targetidx = None

    def setup_pick_and_place(self, robot_id):
        if robot_id == 0:
            sourcetray = self.mototray1
            targettray = self.mototray4
        else:
            sourcetray = self.mototray3
            targettray = self.mototray2
        rcfg = self.get_robot_config(robot_id)
        moto = sourcetray.get_first_phone()

        ok, pickpos, pickori = sourcetray.get_first_full_slot_pose()
        if not ok:
            carb.log_error("No object in source tray")
            return

        ok, targpos, targori = targettray.get_first_empty_slot_pose()
        if not ok:
            carb.log_error("No space in target tray")
            return
        print(f"setuppap - r:{robot_id} pickpos:{pickpos} targpos:{targpos}")

        sourceidx = sourcetray.get_first_full_slot()
        targetidx = targettray.get_first_empty_slot()

        self.activate_ee_collision( robot_id, False)
        rcfg.sourcetray = sourcetray
        rcfg.sourceidx = sourceidx
        rcfg.targettray = targettray
        rcfg.targetidx = targetidx
        rcfg.pickobj = moto
        rcfg.pickpos = pickpos
        rcfg.pickori = pickori
        rcfg.targpos = targpos
        rcfg.targori = targori

        for i in range(6):
            pos = targettray.get_trayslot_pos_idx(i)
            print(f"   targettray slot positions - idx:{i} pos:{pos}")

    def get_robot_action_button_text(self, action_name, action_args=None):
        if action_name in self.base_robot_actions:
            rv = super().get_robot_action_button_text(action_name, action_args)
            return rv
        actwrd = "- active"
        match action_name:
            case "FollowTarget 0":
                rcfg = self.get_robot_config(0)
                word = actwrd if rcfg.current_robot_action == "FollowTarget" else ""
                rv = f"Follow Target 0 {word}"
            case "FollowTarget 1":
                rcfg = self.get_robot_config(1)
                word = actwrd if rcfg.current_robot_action == "FollowTarget" else ""
                rv = f"Follow Target 1 {word}"
            case "PickAndPlace 0":
                rcfg = self.get_robot_config(0)
                word = actwrd if rcfg.current_robot_action == "PickAndPlace" else ""
                rv = f"PickAndPlace 0 {word}"
            case "PickAndPlace 1":
                rcfg = self.get_robot_config(1)
                word = actwrd if rcfg.current_robot_action == "PickAndPlace" else ""
                rv = f"PickAndPlace 1  {word}"
            case "MoveToZero 0":
                rcfg = self.get_robot_config(0)
                word = actwrd if rcfg.current_robot_action == "MoveToZero" else ""
                rv = f"MoveToZero 0 {word}"
            case "MoveToZero 1":
                rcfg = self.get_robot_config(1)
                word = actwrd if rcfg.current_robot_action == "MoveToZero" else ""
                rv = f"MoveToZero 1 {word}"
            case _:
                rv = f"{action_name}"
        return rv

    def get_robot_action_button_tooltip(self, action_name, action_args=None):
        if action_name in self.base_robot_actions:
            rv = super().get_robot_action_button_tooltip(action_name, action_args)
            return rv
        match action_name:
            case _:
                rv = f"No tooltip for action {action_name}"
        return rv

    def get_robot_actions(self):
        self.base_robot_actions = super().get_robot_actions()
        baselist = ["FollowTarget", "PickAndPlace", "MoveToZero"]
        newlist = []
        for act in baselist:
            for i in range(self._nrobots):
                actname = f"{act} {i}"
                newlist.append(actname)
        combo  = self.base_robot_actions + newlist
        return combo
