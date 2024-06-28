import os
import numpy as np
import lula
import copy

import carb
import carb.settings
from omni.isaac.motion_generation.lula.interface_helper import LulaInterfaceHelper
from .matman import MatMan
from pxr import Usd, UsdGeom, UsdShade, Gf, UsdPhysics
from typing import List

from omni.isaac.core.articulations import Articulation
from omni.isaac.motion_generation import ArticulationMotionPolicy
from omni.isaac.motion_generation import ArticulationKinematicsSolver

from omni.isaac.core.utils.stage import get_current_stage
from omni.isaac.motion_generation import RmpFlow
from omni.isaac.core.utils.rotations import euler_angles_to_quat

from omni.isaac.core.world import World
from omni.isaac.core.objects import GroundPlane
from omni.isaac.core.prims import XFormPrim
from omni.isaac.core.prims.rigid_prim import RigidPrim

from .scenario_robot_configs import create_and_populate_robot_config, init_configs

from .senut import find_prims_by_name
from .senut import add_rob_cam
from .senut import build_material_dict, apply_material_to_prim_and_children
from .senut import apply_matdict_to_prim_and_children
from .senut import set_stiffness_for_joints, set_damping_for_joints

from omni.isaac.core.utils.extensions import get_extension_path_from_name
from omni.isaac.core.utils.stage import add_reference_to_stage

from .senut import apply_convex_decomposition_to_mesh_and_children
from .senut import apply_diable_gravity_to_rigid_bodies, adjust_articulationAPI_location_if_needed
from .senut import add_sphere_light_to_stage, add_dome_light_to_stage
from .senut import get_link_paths

from .senut import apply_convex_decomposition_to_mesh_and_children
from .senut import apply_collisionapis_to_mesh_and_children
from .senut import apply_diable_gravity_to_rigid_bodies

from .franka.controllers import PickPlaceController as franka_PickPlaceController
from omni.asimov.jaka.controllers.pick_place_controller import PickPlaceController as jaka_PickPlaceController
from .universal_robots.omni.isaac.universal_robots.controllers import PickPlaceController as ur10_PickPlaceController

from omni.asimov.manipulators.grippers.parallel_gripper import ParallelGripper
from omni.asimov.manipulators.grippers.surface_gripper import SurfaceGripper

class ScenarioBase:
    rmpactive = False
    global_time = 0
    _nrobots = 0
    _rcfg_list = []
    _show_joints_close_to_limits = False

    _moto50mp_list = []
    _moto_tray_list = []
    _colprims = None
    _matman = None

    _show_rmp_target = False
    _show_rmp_target_opt = "invisible" # don't delete
    _show_collision_bounds = False
    _show_collision_bounds_opt = "invisible" # don't delete
    _show_endeffector_box = False
    robcamviews = None
    show_joint_limits_for_all_robots = False
    current_robot_action = ""

    def __init__(self):
        self._scenario_name = "empty scenario"
        self._secnario_desc = "description from ScenarioBase class"
        self._nrobots = 0
        self._rcfg_list = []
        self._stage = get_current_stage()
        self.robcamlist = {}
        self.rmpactive = False
        self.cam_snapshot_active = False
        self.websocket_send_message_active = False
        self.current_extension_path = get_extension_path_from_name("JakaControl")
        init_configs()
        pass

    @staticmethod
    def get_scenario_names():
        rv = [ "sinusoid-joint","inverse-kinematics","rmpflow","rmpflow-new","gripper","object-inspection","cage-rmpflow",
             "franka-pick-and-place","pick-and-place","pick-and-place-new"]
        return rv

    @staticmethod
    def get_default_scenario():
        rv = "cage-rmpflow"
        return rv

    @staticmethod
    def get_scenario_desc(scenario_name):
        match scenario_name:
            case "sinusoid-joint":
                rv = "Move robot through its joints in a sinusoid - from Nvidia example."
            case "object-inspection":
                rv = "Two Jaka Minicobo robots - Object Inspection Scenario"
            case "cage-rmpflow":
                rv = "Two Jaka Minicobo robots - Cage RMPflow Scenario"
            case "franka-pick-and-place":
                rv = "Franka Pick and Place"
            case "pick-and-place":
                rv = "Pick and Place Scenario - for testing pick and place controllers"
            case "pick-and-place-new":
                rv = "Pick and Place Scenario - for testing pick and place controllers - new version"
            case "rmpflow":
                rv = "RMPflow - For testing robots with RMPFlow controllers"
            case "rmpflow-new":
                rv = "RMPflow - For testing robots with RMPFlow controllers - new version"
            case "inverse-kinematics":
                rv = "Inverse Kinematics - For testing robots with lula inverse kinematics controllers"
            case "gripper":
                rv = "Gripper - For testing robots with lula inverse kinematics controllers"
            case _:
                rv = f"Unknown Scenario:{scenario_name}"
                print(rv)
        return rv

    @staticmethod
    def get_robot_desc(robot_name):
        # this is ugly - please change me
        robcfg = create_and_populate_robot_config(robot_name, skiplula=True)
        return robcfg.desc

    @staticmethod
    def get_scenario_robots(scenario_name):
        match scenario_name:
            case "sinusoid-joint":
                rv = ["franka", "ur10e", "ur5e", "ur3e", "jaka-minicobo-0"]
            case "object-inspection":
                rv = ["minicobo-dual-high","minicobo-rg2-high","jaka-minicobo-1a","minicobo-dual-sucker","rs007n"]
            case "cage-rmpflow":
                rv = ["minicobo-dual-high","minicobo-rg2-high","jaka-minicobo-1a","minicobo-dual-sucker","rs007n"]
            case "franka-pick-and-place":
                rv = ["franka", "fancy_franka","rs007n", "ur10-suction-short"]
            case "pick-and-place" | "pick-and-place-new" | "rmpflow"  | "rmpflow-new"  | "inverse-kinematics":
                rv = ["franka", "fancy_franka","rs007n", "ur10-suction-short",
                    "jaka-minicobo-1","jaka-minicobo-2","jaka-minicobo-1a", "minicobo-dual-sucker",
                    "minicobo-rg2-high", "minicobo-suction-dual", "minicobo-suction", "minicobo-suction-high", "minicobo-dual-high"]
            case "gripper":
                rv = ["cone","inverted-cone","sphere","cube","cube-yrot","cylinder","suction-short","suction-dual","suction-dual-0"]
            case _:
                rv = ["ur3e", "ur5e", "ur10e", "ur10e-gripper", "ur10-suction-short",
                    "jaka-minicobo-0","jaka-minicobo-1", "jaka-minicobo-1a","minicobo-dual-sucker", "jaka-minicobo-2",
                    "minicobo-rg2-high","minicobo-suction-dual","minicobo-suction","minicobo-suction-high","minicobo-dual-high",
                    "rs007n", "franka", "fancy_franka", "m0609",
                    "cone","inverted-cone","sphere","cube","cube-yrot","cylinder","suction-short","suction-dual","suction-dual-0"]
        return rv

    @staticmethod
    def can_handle_robot(scenario_name, robot_name):
        robs = ScenarioBase.get_scenario_robots(scenario_name)
        rv = robot_name in robs
        return rv


    def get_robot_config(self, robidx=0):
        if robidx<len(self._rcfg_list):
            return self._rcfg_list[robidx]
        else:
            carb.log_warn(f"get_robot_config - index {robidx} out of range")
        return None

    def create_robot_config(self, robot_name, robot_root_path, ground_opt=""):
        robcfg = create_and_populate_robot_config(robot_name, robot_root_path)
        robcfg.listidx = len(self._rcfg_list)
        self._rcfg_list.append(robcfg)

        if robcfg.rdf_path=="" or robcfg.urdf_path=="":
            msg = f"Robot {robot_name} rdf_path or urdf_path not specified"
            carb.log_warn(msg)
            print(msg)
            return

        if not os.path.isfile(robcfg.rdf_path):
            msg = f"Robot {robot_name} rdf_path bad - file not found:{robcfg.rdf_path}"
            carb.log_error(msg)
            print(msg)
            return

        if not os.path.isfile(robcfg.urdf_path):
            msg = f"Robot {robot_name} urdf_path bad - file not found:{robcfg.urdf_path}"
            carb.log_error(msg)
            print(msg)
            return

        if not os.path.isfile(robcfg.rmp_config_path):
            msg = f"Robot {robot_name} rmp_config_path bad - file not found:{robcfg.rmp_config_path}"
            carb.log_error(msg)
            print(msg)
            return

        try:
            robcfg.robot_description = lula.load_robot(robcfg.rdf_path, robcfg.urdf_path)
            robcfg.lulaHelper = LulaInterfaceHelper(robcfg.robot_description)
        except Exception as e:
            msg = f"ScenarioBase - Robot {robot_name} lula.load_robot of rdf and urdf failed:{e}"
            carb.log_error(msg)
            print(msg)
        return robcfg

    def add_light(self,light_opt):
        match light_opt:
            case "default" | "sphere_light":
                add_sphere_light_to_stage()
            case "dome_light":
                add_dome_light_to_stage()

    def add_ground(self,ground_opt):
        self._ground_opt = ground_opt
        world = World.instance()

        if self._ground_opt == "default":
            self._ground=world.scene.add_default_ground_plane(z_position=-1.02)

        elif self._ground_opt == "groundplane":
            self._ground = GroundPlane(prim_path="/World/groundPlane", size=10, color=np.array([0.5, 0.5, 0.5]))
            world.scene.add(self._ground)

        elif self._ground_opt == "groundplane-blue":
            self._ground = GroundPlane(prim_path="/World/groundPlane", size=10, color=np.array([0.0, 0.0, 0.5]))
            world.scene.add(self._ground)

    def activate_ee_collision(self, robidx, active):
        rcfg = self.get_robot_config(robidx)
        if rcfg is not None:
            usdpath = rcfg.robot_prim_path
            include = [f"{rcfg.eeframe_name}/visuals", f"{rcfg.eeframe_name}/collisions"]
            remove = not active
            apply_collisionapis_to_mesh_and_children(self._stage, usdpath, include=include, remove=remove)

    robcamlist = {}

    def get_or_create_gripper(self, ridx):
        rcfg = self.get_robot_config(ridx)
        art = rcfg._articulation
        if not hasattr(art, "_policy_robot_name"):
            art._policy_robot_name = rcfg.mopo_robot_name #ugly hack, should remove at some point
        if hasattr(art,"gripper"):
            # this is the case for robots with pre-configured grippers
            if not hasattr(self,"grip_eeori"):
                rcfg.grip_eeori = euler_angles_to_quat(np.array([0,0,0]))
            if not hasattr(self,"grip_eeoff"):
                rcfg.grip_eeoff = np.array([0,0,0])
            return art.gripper
        else:
            art = rcfg._articulation
            rcfg._gripper_type = "parallel"
            art._policy_robot_name = rcfg.mopo_robot_name
            world = World.instance()
            rcfg.physics_sim_view = world.physics_sim_view
            rcfg.grip_eeori = euler_angles_to_quat(np.array([0,0,0]))
            rcfg.grip_eeoff = np.array([0,0,0])
            grippername = rcfg.grippername

            # if self._robot_name in ["franka","fancy_franka"]:   # franka gripper
            if grippername=="franka gripper":   # franka gripper
                eepp = "/World/roborg/franka/panda_rightfinger"
                jpn = ["panda_finger_joint1", "panda_finger_joint2"]
                jop = np.array([0.05, 0.05])
                jcp = np.array([0, 0])
                ad = np.array([0.05, 0.05])
                art._policy_robot_name = "Franka"
                # try getting sim_view from world

                pg = ParallelGripper(
                    end_effector_prim_path=eepp,
                    joint_prim_names=jpn,
                    joint_opened_positions=jop,
                    joint_closed_positions=jcp,
                    action_deltas=ad
                )
                pg.initialize(
                    physics_sim_view=rcfg.physics_sim_view,
                    articulation_apply_action_func=art.apply_action,
                    get_joint_positions_func=art.get_joint_positions,
                    set_joint_positions_func=art.set_joint_positions,
                    dof_names=art.dof_names,
                )
                return pg

            elif grippername=="rg2": # rg2 gripper / eepp, jpn, jop,jcp, ad
            # elif self._robot_name in ["rs007n","jaka-minicobo-2","minicobo-rg2-high"]: # rg2 gripper / eepp, jpn, jop,jcp, ad
                art = rcfg._articulation
                if rcfg.robot_name == "rs007n":
                    eepp = "/World/roborg/khi_rs007n/gripper_center"
                else:
                    eepp = "/World/roborg/minicobo_parallel_onrobot_rg2/minicobo_onrobot_rg2/gripper_center"
                jpn = ["left_inner_finger_joint", "right_inner_finger_joint"]
                jop = np.array([0.05, 0.05])
                jcp = np.array([0, 0])
                ad = np.array([0.05, 0.05])
                art._policy_robot_name = "RS007N"
                pg = ParallelGripper(
                    end_effector_prim_path=eepp,
                    joint_prim_names=jpn,
                    joint_opened_positions=jop,
                    joint_closed_positions=jcp,
                    action_deltas=ad
                )
                print(f"art dof names: {art.dof_names}")
                pg.initialize(
                    physics_sim_view=None,
                    articulation_apply_action_func=art.apply_action,
                    get_joint_positions_func=art.get_joint_positions,
                    set_joint_positions_func=art.set_joint_positions,
                    dof_names=art.dof_names,
                )
                rcfg._gripper_type = "parallel"
                return pg
            elif grippername in ["short suction", "dual sucker"]:  # short suction gripper and dual sucker gripper
                art = rcfg._articulation
                rcfg._gripper_type = "suction"
                grip_direction = "x"
                grip_threshold = 0.02
                grip_translate = 0.1611

                if rcfg.robot_name == "ur10-suction-short":
                    eepp = "/World/roborg/ur10_suction_short/ee_link"
                    rcfg.grip_eeori = euler_angles_to_quat(np.array([0,np.pi/2,0]))
                elif rcfg.robot_name == "minicobo-suction":
                    # eepp = "/World/roborg/minicobo_suction/short_gripper"
                    eepp = "/World/roborg/minicobo_suction_short/minicobo_suction/short_gripper"
                elif rcfg.robot_name == "minicobo-suction-high":
                    eepp = "/World/roborg/minicobo_suction_short/minicobo_suction/short_gripper"
                elif rcfg.robot_name in ["minicobo-suction-dual","minicobo-dual-high"]:
                    eepp = "/World/roborg/minicobo_suction_dual/minicobo_suction/dual_gripper"
                    grip_direction = "y"
                    grip_threshold = 0.1
                    grip_translate = 0.17
                    rcfg.grip_eeori = euler_angles_to_quat(np.array([-np.pi/2,0,0]))
                elif rcfg.robot_name in ["jaka-minicobo-1a", "minicobo-dual-sucker"]:
                    # eepp = "/World/roborg/minicobo_v1_4/tool0"
                    eepp = f"{rcfg.robot_prim_path}/{rcfg.eeframe_name}"
                    grip_direction = "y"
                    grip_threshold = 0.01
                    # grip_translate = -0.018 # 0.002 and -0.019 does not work, but 0.001 to -0.018 do work for jaka-minicobo-1a and minicobo-dual-sucker
                    grip_translate = 0.0
                    rcfg.grip_eeori = euler_angles_to_quat(np.array([-np.pi/2,0,0]))

                elif rcfg.robot_name == "jaka-minicobo-1":
                    eepp = "/World/roborg/minicobo_v1_4/Link6/jaka_camera_endpoint/JAKA___MOTO_200mp_v4/ZPR25CNK10_06_A10_v007"
                    # eepp = f"{rcfg.robot_prim_path}/{rcfg.eeframe_name}"
                    # rcfg._end_effector = RigidPrim(prim_path=eepp, name= "jaka-minicobo-1" + "_end_effector")

                    # rcfg._end_effector.initialize(rcfg.physics_sim_view)
                else:
                    print("Unknown robot name for suction gripper")
                art._policy_robot_name = "UR10"
                rcfg._end_effector_prim_path = eepp
                sg = SurfaceGripper(
                    end_effector_prim_path=rcfg._end_effector_prim_path,
                    translate=grip_translate, # minicobo-suction works between -0.001 and 0.222 - fails at 0.223 and -0.002
                    direction=grip_direction,
                    grip_threshold=grip_threshold,  # between 0.01 and 0.5 work for minicobo-suction for the big cube
                )
                sg.initialize(
                    physics_sim_view=rcfg.physics_sim_view,
                    articulation_num_dofs=len(art.dof_names)
                )
                return sg

            else:
                carb.log_error(f"Unknown gripper type: {grippername} for robot:{rcfg.robot_name} - returning None")
                return None

    def add_pp_controllers_to_robots(self):
        for idx in range(self._nrobots):
            self.add_pp_controller(idx)

    def add_pp_controller(self,ridx=0):

        events_dt = [0.008, 0.005, 0.1,  0.1, 0.005, 0.005, 0.005, 0.1, 0.008, 0.08]
        rcfg = self.get_robot_config(ridx)

        gripper = rcfg.gripper
        if gripper is not None:
            artic = rcfg._articulation
            if rcfg.pp_controller == "franka":
            # if self._robot_name in ["fancy_franka", "franka", "rs007n"]:
                rcfg._gripper_type = "parallel"
                rcfg._controller = franka_PickPlaceController(
                    name="pick_place_controller",
                    gripper=gripper,
                    robot_articulation=artic,
                    events_dt=events_dt
                )
            elif rcfg.pp_controller in ["ur-rg2", "ur-ss"]:
            # elif self._robot_name in ["ur10-suction-short"]:
                rcfg._gripper_type = "suction" if rcfg.pp_controller == "ur10-ss" else "parallel"
                rcfg._controller = ur10_PickPlaceController(
                    name="pick_place_controller",
                    gripper=gripper,
                    robot_articulation=artic
                )
            elif rcfg.pp_controller in ["jaka-ss", "jaka-ds"]:
            #elif self._robot_name in ["minicobo-suction","minicobo-suction-high","jaka-minicobo-1",
            #                           "jaka-minicobo-1a","minicobo-dual-sucker","minicobo-suction-dual","minicobo-dual-high"]:
                rcfg._gripper_type = "suction"
                rmpconfig = {
                    "end_effector_frame_name": rcfg.eeframe_name,
                    "maximum_substep_size": rcfg.max_step_size,
                    "ignore_robot_state_updates": False,
                    "urdf_path": rcfg.urdf_path,
                    "rmpflow_config_path": rcfg.rmp_config_path,
                    "robot_description_path": rcfg.rdf_path
                }
                events_dt = [0.008, 0.005, 0.1,  0.1, 0.005, 0.005, 0.005, 0.1, 0.008, 0.08]
                rcfg._controller = jaka_PickPlaceController(
                    name="pick_place_controller",
                    gripper=gripper,
                    robot_articulation=artic,
                    rmpconfig=rmpconfig,
                    events_dt=events_dt,
                    end_effector_initial_height=0.1
                )
            elif rcfg.pp_controller == "jaka-rg2":
            # elif self._robot_name in ["jaka-minicobo-0","jaka-minicobo-2","minicobo-rg2-high"]:
                rcfg._gripper_type = "parallel"

                rmpconfig = {
                    "end_effector_frame_name": rcfg.eeframe_name,
                    "maximum_substep_size": rcfg.max_step_size,
                    "ignore_robot_state_updates": False,
                    "urdf_path": rcfg.urdf_path,
                    "rmpflow_config_path": rcfg.rmp_config_path,
                    "robot_description_path": rcfg.rdf_path
                }
                rcfg._controller = jaka_PickPlaceController(
                    name="pick_place_controller",
                    gripper=gripper,
                    robot_articulation=artic,
                    rmpconfig=rmpconfig,
                    events_dt=events_dt
                )
            else:
                print(f"add_pp_controllers - unknown controller: {rcfg.pp_controller}")
                rcfg._controller = None




    robcamlist = {}


    def add_camera_to_robcamlist(self, cam_name, cam_display_name, campath):
        self.robcamlist[cam_name] = {}
        self.robcamlist[cam_name]["name"] = cam_name
        self.robcamlist[cam_name]["display_name"] = cam_display_name
        self.robcamlist[cam_name]["usdpath"] = campath

    def add_camera_to_robot(self, ridx=0):
        rcfg = self.get_robot_config(ridx)
        if rcfg.camera_root != "":
            if rcfg.robot_name == "minicobo-dual-sucker":
                ring_rot = Gf.Vec3f([0,0,-45])
            else:
                ring_rot = Gf.Vec3f([0,0,0])
            mount_trans = Gf.Vec3f([0.011,0.147,-0.011])
            point_quat = Gf.Quatf(0.80383,Gf.Vec3f(-0.19581,-0.46288,-0.31822))

            camroot = rcfg.camera_root
            camname = f"{rcfg.robot_id}_cam"
            _, campath = add_rob_cam(camroot, ring_rot, mount_trans, point_quat, camname)
            self.add_camera_to_robcamlist(rcfg.robot_id, rcfg.robot_name, campath)

    def add_cameras_to_robots(self):
        for idx in range(self._nrobots):
            self.add_camera_to_robot(idx)

    def check_alarm_status(self, rcfg):
        art = rcfg._articulation
        pos = art.get_joint_positions()
        nalarm = 0
        for j,jn in enumerate(rcfg.dof_names):
            rcfg.dof_lamda[j] = (pos[j] - rcfg.lower_dof_lim[j])/rcfg.dof_range[j]
            toolo = pos[j] < rcfg.dof_alarm_llim[j]
            toohi = pos[j] > rcfg.dof_alarm_ulim[j]
            if toolo or toohi:
                rcfg.dof_alarm[j] = True
                nalarm += 1
            else:
                rcfg.dof_alarm[j] = False
        return nalarm

    def register_robot_articulations(self):
        for idx in range(0,self._nrobots):
            rcfg = self.get_robot_config(idx)
            self.register_articulation(rcfg._articulation, rcfg)

    def toggle_show_joints_close_to_limits(self, ridx, notoggle=False):
        rcfg = self.get_robot_config(ridx)
        if not notoggle:
            rcfg.show_joints_close_to_limits = not rcfg.show_joints_close_to_limits
        # print(f"toggle_show_joints_close_to_limits on {rcfg.robot_name} {rcfg.robot_id} - {rcfg.show_joints_close_to_limits}")
        if rcfg.show_joints_close_to_limits:
            self.assign_alarm_skin(ridx)
            self.check_alarm_status(rcfg)
            rcfg.dof_alarm_last = copy.deepcopy(rcfg.dof_alarm)
            self.realize_joint_alarms_for_one(ridx, force=True)
        else:
            if rcfg.robmatskin == "default":
                self.ensure_orimat()
                # print("Reverting to original materials (default)")
                apply_matdict_to_prim_and_children(self._stage, rcfg.orimat, rcfg.robot_prim_path)
            else:
                 #print(f"Reverting to {rcfg.robmatskin}")
                apply_material_to_prim_and_children(self._stage, self._matman, rcfg.robmatskin, rcfg.robot_prim_path)
        # print("toggle_show_joints_close_to_limits done")
        return rcfg.show_joints_close_to_limits

    def assign_alarm_skin(self, ridx):
        rcfg = self.get_robot_config(ridx)
        if rcfg.robmatskin=="Red_Glass":
            rcfg.alarmskin = "Blue_Glass"
        else:
            rcfg.alarmskin = "Red_Glass"

    def realize_joint_alarms_for_all(self,force=False):
        for ridx in range(0,self._nrobots):
            self.realize_joint_alarms_for_one(ridx)

    def realize_joint_alarms_for_one(self,ridx, force=False):
         #print(f"realize_joint_alarms force:{force}")
        rcfg = self.get_robot_config(ridx)
        if rcfg.show_joints_close_to_limits:
            self.assign_alarm_skin(ridx)
            self.check_alarm_status(rcfg)
            for j, jn in enumerate(rcfg.dof_names):
                if force or (rcfg.dof_alarm[j] != rcfg.dof_alarm_last[j]):
                    link_path = rcfg.link_paths[j]
                    joint_in_alarm = rcfg.dof_alarm[j]
                    if joint_in_alarm:
                        # print(f"   changing {link_path} to {rcfg.alarmskin} - inalarm:{joint_in_alarm}")
                        # print(f"Joint {jn} is close to limit for {rcfg.robot_name} {rcfg.robot_id} link_path:{link_path}")
                        apply_material_to_prim_and_children(self._stage, self._matman, rcfg.alarmskin, link_path)
                    else:
                        # print(f"Joint {jn} is not close to limit for {rcfg.robot_name} {rcfg.robot_id} link_path:{link_path}")
                        if rcfg.robmatskin == "default":
                            self.ensure_orimat()
                            # print(f"   changing {link_path} to rcfg.orimat - inalarm:{joint_in_alarm}")
                            apply_matdict_to_prim_and_children(self._stage, rcfg.orimat, link_path)
                        else:
                            # print(f"   changing {link_path} to {rcfg.robmatskin} - inalarm:{joint_in_alarm}")
                            apply_material_to_prim_and_children(self._stage, self._matman, rcfg.robmatskin, link_path)
            rcfg.dof_alarm_last = copy.deepcopy(rcfg.dof_alarm)

    def register_articulation(self, articulation, rcfg=None):
        # this has to happen in post_load_scenario - some initialization must be happening before this
        # probably as a result of articuation being added to the world.scene
        # TODO: once everthing uses register_robot_articulations we can get rid of the articulation parameters
        if rcfg is None:
            rcfg = self.get_robot_config(0)
        # print(f"senut.register_articulation for {rcfg.robot_name} {rcfg.robot_id}")

        art = articulation
        rcfg._articulation = art
        rcfg.dof_paths = art._prim_view._dof_paths[0] # why is this a list while the following ones are not?
        rcfg.dof_types = art._prim_view._dof_types
        rcfg.dof_names = art._prim_view._dof_names
        rcfg.link_paths = get_link_paths(rcfg.dof_paths)

        rcfg.lower_dof_lim = art.dof_properties["lower"]
        rcfg.upper_dof_lim = art.dof_properties["upper"]
        rcfg.njoints = art.num_dof
        rcfg.dof_zero_pos = np.zeros(rcfg.njoints)
        rcfg.show_joints_close_to_limits = self.show_joint_limits_for_all_robots

        pos = art.get_joint_positions()
        props = art.dof_properties

        rcfg.dof_alarm_llim = np.zeros(rcfg.njoints)
        rcfg.dof_alarm_ulim = np.zeros(rcfg.njoints)
        rcfg.orig_dof_pos =  copy.deepcopy(pos)
        lower_alarm_gap = 0.1
        upper_alarm_gap = 0.1
        rcfg.dof_alarm = np.zeros(rcfg.njoints, dtype=bool)
        rcfg.dof_range = np.zeros(rcfg.njoints)
        rcfg.dof_lamda = np.zeros(rcfg.njoints)
        for j,jn in enumerate(rcfg.dof_names):
            llim = rcfg.lower_dof_lim[j]
            ulim = rcfg.upper_dof_lim[j]
            rcfg.dof_range[j] = ulim - llim
            rcfg.dof_alarm_llim[j] = llim + lower_alarm_gap*(ulim-llim)
            rcfg.dof_alarm_ulim[j] = ulim - upper_alarm_gap*(ulim-llim)
        self.check_alarm_status(rcfg)

        self.toggle_show_joints_close_to_limits(rcfg.listidx, notoggle=True)
        # print("done senut.register_articulation")

    def setup_robot_for_pose_movement(self, gprim, rcfg, pos, rot, ska=[1, 1, 1], order="ZYX", pre_rot=[0, 0, 0]):
        pos = Gf.Vec3d(list(pos))
        rot = list(rot)

        # Pre-rotation
        if "/ring" in rcfg.robot_prim_path:
            rprimpath = f"{rcfg.root_usdpath}/ring"
            rprim = self._stage.GetPrimAtPath(rprimpath)
            if not rprim.IsValid():
                rprim = UsdGeom.Xform.Define(self._stage, rprimpath)
            rcfg.pre_rotxyzop = rprim.AddRotateXYZOp()
            rvek = Gf.Vec3f(list(np.array(pre_rot)*np.pi/180))
            rvek = Gf.Vec3f(list(pre_rot))
            rcfg.pre_rotxyzop.Set(rvek)

        rcfg.tranop = gprim.AddTranslateOp()
        rad = np.pi/180
        match order:
            case "ZYX":
                rcfg.xrotop = gprim.AddRotateXOp()
                rcfg.yrotop = gprim.AddRotateYOp()
                rcfg.zrotop = gprim.AddRotateZOp()
                rvek = np.array(rot)*rad
                quat = euler_angles_to_quat(rvek, extrinsic=True)

            case "XYZ":
                rcfg.zrotop = gprim.AddRotateZOp()
                rcfg.yrotop = gprim.AddRotateYOp()
                rcfg.xrotop = gprim.AddRotateXOp()
                rvek = np.array(rot)*rad
                quat = euler_angles_to_quat(rvek, extrinsic=True)
        rcfg.tranop.Set(pos)
        rcfg.zrotop.Set(rot[2])
        rcfg.yrotop.Set(rot[1])
        rcfg.xrotop.Set(rot[0])
        rcfg.start_robot_pos = pos
        rcfg.start_robot_rot = rot
        rcfg.robot_rotquat = quat

    def load_robot_into_scene(self, ridx=0, pos=[0, 0, 0], rot=[0, 0, 0], order="ZYX", pre_rot=[0, 0, 0]):
        stage = self._stage
        rcfg = self.get_robot_config(ridx)

        roborg = UsdGeom.Xform.Define(stage, rcfg.root_usdpath)
        self.setup_robot_for_pose_movement(roborg, rcfg, pos, rot, order=order, pre_rot=pre_rot)

        add_reference_to_stage(rcfg.robot_usd_file_path, rcfg.robot_prim_path)
        apply_convex_decomposition_to_mesh_and_children(stage, rcfg.robot_prim_path)
        apply_diable_gravity_to_rigid_bodies(stage, rcfg.robot_prim_path)

        prim = self._stage.GetPrimAtPath(rcfg.robot_prim_path)
        # UsdPhysics.RigidBodyAPI.Apply(prim)
        # UsdPhysics.CollisionAPI.Apply(prim)

        adjust_articulationAPI_location_if_needed(stage, rcfg.robot_prim_path)
        rcfg._articulation = Articulation(rcfg.artpath,f"mico-{ridx}")

        world = World.instance()
        world.scene.add(rcfg._articulation)

        return rcfg._articulation

    def teleport_robots_to_zeropos(self):
        for i in range(self._nrobots):
            rcfg = self.get_robot_config(i)
            if rcfg is not None:
                if i==0:
                    rcfg._articulation.set_joint_positions([1.5385753,-0.067451306,1.8509837,-0.017295461,-1.2601038,-1.5375034])
                else:
                    rcfg._articulation.set_joint_positions([-1.0036137,-1.5866233,1.9273415,-0.028864592,-0.86408925,1.5516889])    

    def get_robot_world_pose(self, robidx):
        rcfg = self.get_robot_config(robidx)
        prim = self._stage.GetPrimAtPath(rcfg.robot_prim_path)
        tc = Usd.TimeCode.Default()
        xf = UsdGeom.Xformable(prim)
        world_transform: Gf.Matrix4d = xf.ComputeLocalToWorldTransform(tc)
        pos = world_transform.ExtractTranslation()
        rot = world_transform.ExtractRotation().GetQuaternion()
        posar = np.array([pos[0], pos[1], pos[2]])
        # rotar = np.array([rot.GetImaginary()[0], rot.GetImaginary()[1], rot.GetImaginary()[2], rot.GetReal()])
        im = rot.GetImaginary()
        rotar = np.array([rot.GetReal(), im[0], im[1], im[2]])
        return posar, rotar


    def make_rmpflow(self, rob_idx, oblist = []):
        rcfg = self.get_robot_config(rob_idx)
        rmpflow = RmpFlow(
            robot_description_path = rcfg.rdf_path,
            urdf_path = rcfg.urdf_path,
            rmpflow_config_path = rcfg.rmp_config_path,
            end_effector_frame_name = rcfg.eeframe_name,
            maximum_substep_size = rcfg.max_step_size
        )


        pos_a = np.array(rcfg.start_robot_pos)
        quat_a = np.array(rcfg.robot_rotquat)
        pos_b,quat_b = self.get_robot_world_pose(rob_idx)
        print(f"robidx:{rob_idx}  pos:{pos_a} quat:{quat_a}")
        print(f"       {rob_idx}  pos:{pos_b} quat:{quat_b}")
        rmpflow.set_robot_base_pose(pos_b, quat_b)

        for ob in oblist:
            rmpflow.add_obstacle(ob)

        rmpflow.set_ignore_state_updates(True)
        if self._show_collision_bounds:
            rmpflow.visualize_collision_spheres()

        rcfg.articulation_rmpflow = ArticulationMotionPolicy(rcfg._articulation,rmpflow)
        rcfg._kinematics_solver = rmpflow.get_kinematics_solver()

        rcfg._articulation_kinematics_solver = ArticulationKinematicsSolver(rcfg._articulation, rcfg._kinematics_solver, rcfg.eeframe_name)
        rcfg.rmpflow = rmpflow

    def adjust_stiffness_and_damping_for_robots(self):
        for idx in range(self._nrobots):
            rcfg = self.get_robot_config(idx)
            self.set_stiffness_and_damping_for_all_joints(rcfg)

    def make_robot_mpflows(self, oblist = []):
        for i in range(self._nrobots):
            self.make_rmpflow(i, oblist)

    def reset_robot_rmpflow(self, rob_idx):
        rcfg = self.get_robot_config(rob_idx)
        # rcfg.rmpflow.reset()
        if self._show_collision_bounds:
            rcfg.rmpflow.visualize_collision_spheres()
        if self._show_endeffector_box:
            rcfg.rmpflow.visualize_end_effector_position()

    def reset_robot_rmpflows(self):
        for i in range(self._nrobots):
            self.reset_robot_rmpflow(i)

    def forward_rmpflow_step_for_robot(self, rob_idx, step_size):
        rcfg = self.get_robot_config(rob_idx)
        action = rcfg.articulation_rmpflow.get_next_articulation_action(step_size)
        rcfg._articulation.apply_action(action)

    def forward_rmpflow_step_for_robots(self, step_size):
        for i in range(self._nrobots):
            self.forward_rmpflow_step_for_robot(i, step_size)

    def rmpflow_update_world_for_all(self):
        for i in range(self._nrobots):
            rcfg = self.get_robot_config(i)
            rcfg.rmpflow.update_world()

    def set_end_effector_target_for_robot(self, rob_idx, ee_targ_pos, ee_targ_ori):
        rcfg = self.get_robot_config(rob_idx)
        rcfg.rmpflow.set_end_effector_target(ee_targ_pos, ee_targ_ori)

    def load_scenario(self, robot_name="default", ground_opt="default"):
        self._matman = MatMan(get_current_stage())

    def setup_scenario(self):
        pass

    def post_load_scenario(self):
        pass

    def reset_scenario(self):
        pass

    def teardown_scenario(self):
        pass

    def update_scenario(self):
        pass

    targXformTop = None
    def visualize_rmp_target(self):
        targname = "/World/rmp_target"
        stage = get_current_stage()
        prim = stage.GetPrimAtPath(targname)
        if prim is None or not prim.IsValid():
            targXform = UsdGeom.Xform.Define(stage, targname)
            self.targXformTop = targXform.AddTranslateOp()
            targXform.AddScaleOp().Set((0.01, 0.01, 0.01))
            spherePrim = UsdGeom.Sphere.Define(stage, targname + '/sphere')
            spherePrim.GetDisplayColorAttr().Set([(1, 1, 0)])
        if hasattr(self._controller, "mw_postarg"):
            pos = Gf.Vec3d(list(self._controller.mw_postarg))
            self.targXformTop.Set(pos)

    def realize_rmptarg_vis(self, opt):
        if hasattr(self, "_show_rmp_target" ):
           self._show_rmp_target = opt.lower() != "invisible"
           self._show_rmp_target_opt = opt
           if self._show_rmp_target:
               self.visualize_rmp_target()

    def restore_robot_skins(self):
        for rcfg in self._rcfg_list:
            if hasattr(rcfg, "orimat"):
                matdict = rcfg.orimat
                nchg = apply_matdict_to_prim_and_children(self._stage, matdict, rcfg.robot_prim_path)
                print(f"restore_robot_skins - {nchg} materials restored for {rcfg.robot_prim_path}")

    def realize_robot_skin(self, skinopt):
        match skinopt:
            case "Default"|"default":
                self.restore_robot_skins()
                return
            case "Clear Glass":
                mat1 = mat2 =  "Clear_Glass"
            case "Red Glass":
                mat1 = mat2 =  "Red_Glass"
            case "Green Glass":
                mat1 = mat2 =  "Green_Glass"
            case "Blue Glass":
                mat1 = mat2 =  "Blue_Glass"
            case "Tinted Glass":
                mat1 = mat2 =  "Tinted_Glass"
            case "Tinted Glass 75":
                mat1 = mat2 =  "Tinted_Glass_R75"
            case "Tinted Glass 85":
                mat1 = mat2 =  "Tinted_Glass_R85"
            case "Tinted Glass 98":
                mat1 = mat2 =  "Tinted_Glass_R98"
            case "Red/Green Glass":
                mat1 = "Red_Glass"
                mat2 = "Green_Glass"
            case "Red/Blue Glass":
                mat1 = "Red_Glass"
                mat2 = "Blue_Glass"
            case "Green/Blue Glass":
                mat1 = "Green_Glass"
                mat2 = "Blue_Glass"
            case "Blue Glass":
                mat1 = mat2 =  "Blue_Glass"
        print(f"realize_robot_skin robskin opt {skinopt} mat1:{mat1} mat2:{mat2}")

        didone = False
        self.ensure_orimat()
        for i,rcfg in enumerate(self._rcfg_list):
            mat = mat1 if i%2==0 else mat2
            apply_material_to_prim_and_children(self._stage, self._matman, mat, rcfg.robot_prim_path)
            rcfg = self.get_robot_config(i)
            rcfg.robmatskin = mat
            didone = True
        if not didone:
            carb.log_warn("realize_robot_skin - no robot config found")

    def change_colliders_viz(self, action):
        stage = get_current_stage()
        if self._matman is None:
            self._matman = MatMan(stage)

        if self._colprims is None:
            self._colprims: List[Usd.Prim] = find_prims_by_name("collision_sphere")
        print(f"adjust_colliders action:{action} len:{len(self._colprims)}")
        for prim in self._colprims:
            gprim = UsdGeom.Gprim(prim)
            try:
                if action == "Red Colliders":
                    material = self._matman.GetMaterial("red")
                    UsdShade.MaterialBindingAPI(gprim).Bind(material)
                elif action == "Transparent Colliders":
                    material = self._matman.GetMaterial("Clear_Glass")
                    UsdShade.MaterialBindingAPI(gprim).Bind(material)
                elif action == "Show Colliders":
                    gprim.MakeVisible()
                elif action == "Hide Colliders":
                    gprim.MakeInvisible()
            except:
                pass

    def realize_collider_vis_opt(self, opt):
        pass
    # Broken, does not work for multiple robots as written
        # stage = get_current_stage()
        # if self._matman is None:
        #     self._matman = MatMan(stage)

        # if self._colprims is None:
        #     self._colprims: List[Usd.Prim] = find_prims_by_name("collision_sphere")
        # print(f"realize_collider_vis_opt:{opt} nspheres:{len(self._colprims)}")
        # nfliped = 0
        # nexcept = 0
        # for prim in self._colprims:
        #     gprim = UsdGeom.Gprim(prim)
        #     try:
        #         if opt == "Red":
        #             gprim.MakeVisible()
        #             material = self._matman.GetMaterial("red")
        #             UsdShade.MaterialBindingAPI(gprim).Bind(material)
        #         elif opt == "Glass":
        #             gprim.MakeVisible()
        #             material = self._matman.GetMaterial("Clear_Glass")
        #             UsdShade.MaterialBindingAPI(gprim).Bind(material)
        #         else:
        #             gprim.MakeInvisible()
        #         nfliped += 1
        #     except:
        #         nexcept += 1
        #         pass
        # print(f"Realize_collider_vis_opt changed:{nfliped} exceptions:{nexcept}")

    def realize_rotate_opt(self, opt):
        if hasattr(self, "_rotate" ):
            opt = opt.lower()
            match opt:
                case "none":
                    self._rotate = False
                case "rotateforward":
                    self._rotate = True
                    self._rotate_speed = 1
                case "rotatebackward":
                    self._rotate = True
                    self._rotate_speed = -1

    def realize_eetarg_vis(self, opt):
        stage = get_current_stage()
        self._matman = MatMan(stage)

        prims = find_prims_by_name("end_effector")
        for prim in prims:
            gprim = UsdGeom.Gprim(prim)
            try:
                if opt == "Blue":
                    gprim.MakeVisible()
                    material = self._matman.GetMaterial("blue")
                    UsdShade.MaterialBindingAPI(gprim).Bind(material)
                elif opt == "Glass":
                    gprim.MakeVisible()
                    material = self._matman.GetMaterial("Clear_Glass")
                    UsdShade.MaterialBindingAPI(gprim).Bind(material)
                elif opt == "BlueGlass":
                    gprim.MakeVisible()
                    material = self._matman.GetMaterial("Blue_Glass")
                    UsdShade.MaterialBindingAPI(gprim).Bind(material)
                elif opt == "Invisible":
                    gprim.MakeInvisible()
            except:
                pass

    def set_stiffness_and_damping_for_all_joints(self, rcfg):
        if rcfg.stiffness>0:
            set_stiffness_for_joints(rcfg.dof_paths, rcfg.stiffness)
        if rcfg.damping>0:
            set_damping_for_joints(rcfg.dof_paths, rcfg.damping)

    def ensure_orimat(self):
        for rcfg in self._rcfg_list:
            if not hasattr(rcfg, "orimat"):
                rcfg.orimat = build_material_dict(self._stage, rcfg.robot_prim_path)

    def joint_check_robot(self,rcfg):
        degs = 180/np.pi
        art = rcfg._articulation
        pos = art.get_joint_positions()
        props = art.dof_properties
        for j,jn in enumerate(rcfg.dof_names):
            jpos = degs*pos[j]
            llim = degs*rcfg.lower_dof_lim[j]
            ulim = degs*rcfg.upper_dof_lim[j]
            denom = ulim - llim
            if denom == 0:
                denom = 1
            pct = 100*(jpos - llim)/denom
            if pct<10 or 90>pct:
                clr = "green"

    def show_joint_limits_for_all_robots(self, showthem):
        self.show_joint_limits_for_all_robots = showthem
        for ridx,rcfg in enumerate(self._rcfg_list):
            if rcfg.show_joints_close_to_limits != self.show_joint_limits_for_all_robots:
                self.toggle_show_joints_close_to_limits(ridx)

    def toggle_show_joint_limits(self, notoggle=False):
        if not notoggle:
            self.show_joint_limits_for_all_robots = not self.show_joint_limits_for_all_robots
        self.uibuilder._joint_alarms = self.show_joint_limits_for_all_robots
        for ridx,rcfg in enumerate(self._rcfg_list):
            if rcfg.show_joints_close_to_limits != self.show_joint_limits_for_all_robots:
                self.toggle_show_joints_close_to_limits(ridx)

    def joint_check(self):
        self.ensure_orimat()
        for rcfg in self._rcfg_list:
            self.joint_check_robot(rcfg)

    def add_spheres_to_joints(self, ribx=0):
        rcfg = self.get_robot_config(ribx)
        for j,jp in enumerate(rcfg.dof_paths):
            print(f"adding sphere to joint {j} {rcfg.dof_names[j]}")
            sphpath = f"{jp}/joint_sphere"
            prim = UsdGeom.Sphere.Define(self._stage, sphpath)
            sz = 0.01
            prim.AddScaleOp().Set(Gf.Vec3f(sz,sz,sz))
            prim.GetDisplayColorAttr().Set([(1, 1, 0)])

    def scenario_action(self, action_name, action_args):
        match action_name:
            case "Show Joint Limits":
                self.toggle_show_joint_limits()

    def get_scenario_actions(self):
        rv =  ["Show Joint Limits"]
        return rv

    def robot_action(self, action_name, action_args):
        self.current_robot_action = action_name
        match action_name:
            case _:
             pass

    def get_robot_actions(self):
        rv =  []
        return rv

    def get_scenario_action_button_text(self, action_name,action_args=None):
        match action_name:
            case "Show Joint Limits":
                ltext = "is on" if self.show_joint_limits_for_all_robots else "is off"
                msg = f"Show Joint Limits {ltext}"
                return msg
            case _:
                rv = action_name
        return rv

    def get_scenario_action_button_tooltip(self, action_name, action_args=None):
        match action_name:
            case _:
                rv = f"Tooltip for {action_name}"
        return rv

    def get_robot_action_button_text(self, action_name,action_args=None):
        match action_name:
            case _:
                rv = action_name
        return rv

    def get_robot_action_button_tooltip(self, action_name, action_args=None):
        match action_name:
            case _:
                rv = f"Tooltip for {action_name}"
        return rv