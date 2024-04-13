import os
import numpy as np
import lula
import copy

import carb
import carb.settings
from omni.isaac.motion_generation.lula.interface_helper import LulaInterfaceHelper
from .matman import MatMan
from pxr import Usd, UsdGeom, UsdShade, Gf
from typing import List
import omni
import omni.ui as ui
from omni.kit.widget.viewport import ViewportWidget

from omni.isaac.core.articulations import Articulation

from omni.isaac.core.utils.stage import get_current_stage
from omni.isaac.motion_generation import RmpFlow, ArticulationMotionPolicy
from omni.isaac.core.utils.rotations import euler_angles_to_quat

from omni.isaac.core.world import World
from omni.isaac.core.objects import GroundPlane

from .scenario_robot_configs import create_and_populate_robot_config, init_configs

from .senut import find_prims_by_name
from .senut import add_cam
from .senut import build_material_dict, apply_material_to_prim_and_children
from .senut import apply_matdict_to_prim_and_children
from .senut import set_stiffness_for_joints, set_damping_for_joints

from omni.isaac.core.utils.stage import add_reference_to_stage
from .senut import apply_convex_decomposition_to_mesh_and_children, apply_material_to_prim_and_children
from .senut import apply_diable_gravity_to_rigid_bodies, adjust_articulationAPI_location_if_needed
from .senut import add_sphere_light_to_stage, add_dome_light_to_stage
from .senut import get_link_paths


class ScenarioBase:
    rmpactive = True
    global_time = 0
    _nrobots = 0
    _rcfg_list = []
    _show_joints_close_to_limits = False

    def __init__(self):
        self._scenario_name = "empty scenario"
        self._secnario_desc = "description from ScenarioBase class"
        self._nrobots = 0
        self._rcfg_list = []
        self._stage = get_current_stage()
        self.camlist = {}
        self.rmpactive = True
        init_configs()
        pass

    @staticmethod
    def get_scenario_names():
        rv = [ "sinusoid-joint","inverse-kinematics","rmpflow","gripper","object-inspection","cage-rmpflow",
             "franka-pick-and-place","pick-and-place"]
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
            case "rmpflow":
                rv = "RMPflow - For testing robots with RMPFlow controllers"
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
            case "pick-and-place" | "rmpflow"  | "inverse-kinematics":
                rv = ["franka", "fancy_franka","rs007n", "ur10-suction-short",
                    "jaka-minicobo-0","jaka-minicobo-1","jaka-minicobo-1a", "minicobo-dual-sucker",  "jaka-minicobo-2",
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
            self._ground = GroundPlane(prim_path="/World/groundPlane", size=10, color=np.array([0.5, 0.5, 0.5]),position=[0,0,-1.03313])
            world.scene.add(self._ground)

        elif self._ground_opt == "groundplane-blue":
            self._ground = GroundPlane(prim_path="/World/groundPlane", size=10, color=np.array([0.0, 0.0, 0.5]),position=[0,0,-1.03313])
            world.scene.add(self._ground)

    camlist = {}

    def add_camera_to_camlist(self, cam_name, cam_display_name, campath):
        self.camlist[cam_name] = {}
        self.camlist[cam_name]["name"] = cam_name
        self.camlist[cam_name]["display_name"] = cam_display_name
        self.camlist[cam_name]["usdpath"] = campath

    def add_camera_to_robot(self,robot_name,robot_id,robot_prim_path):
        campath = None
        if robot_name in ["jaka-minicobo-1a","minicobo-dual-sucker"]:
            camera_root = f"{robot_prim_path}/dummy_tcp"
            campath = add_cam(robot_name, camera_root)
        return campath

    def add_cameras_to_robots(self):
        for idx in range(self._nrobots):
            rcfg = self.get_robot_config(idx)
            if rcfg.camera_root != "":
                _, campath = add_cam(rcfg.robot_name, rcfg.camera_root)
                self.add_camera_to_camlist(rcfg.robot_id, rcfg.robot_name, campath)

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

    def show_joints_close_to_limits(self):
        for idx in range(0,self._nrobots):
            rcfg = self.get_robot_config(idx)
            if not rcfg.show_joints_close_to_limits:
                continue
            nalarm = self.check_alarm_status(rcfg)
            alamat = "Red_Glass"
            if nalarm>0:
                for j,jn in enumerate(rcfg.dof_names):
                    if rcfg.dof_alarm[j]:
                        link_path = rcfg.link_paths[j]
                        print(f"Joint {jn} is close to limit for {rcfg.robot_name} {rcfg.robot_id} link_path:{link_path}")
                        apply_material_to_prim_and_children(self._stage, self._matman, alamat, link_path)


            # if hasattr(rcfg, "orimat"):
            #     matdict = rcfg.orimat
            #     nchg = apply_matdict_to_prim_and_children(self._stage, matdict, rcfg.robot_prim_path)
            #     print(f"restore_robot_skins - {nchg} materials restored for {rcfg.robot_prim_path}")


    def register_articulation(self, articulation, rcfg=None):
        # this has to happen in post_load_scenario - some initialization must be happening before this
        # probably as a result of articuation being added to the world.scene
        # self._articulation = articulation
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
        rcfg.show_joints_close_to_limits = False

        pos = art.get_joint_positions()
        props = art.dof_properties
        # stiffs = props["stiffness"]
        # print(f"stiffs for {rcfg.robot_name} {rcfg.robot_id} - {stiffs}")
        # damps = props["damping"]
        # print(f"damps for {rcfg.robot_name} {rcfg.robot_id} - {damps}")
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
        # print("done senut.register_articulation")

    def setup_robot_for_pose_movement(self, gprim, rcfg, pos, rot):
        pos = Gf.Vec3d(list(pos))
        rot = list(rot)
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

    def load_robot_into_scene(self, ridx=0, pos=[0,0,0], rot=[0,0,0]):
        stage = self._stage
        rcfg = self.get_robot_config(ridx)

        roborg = UsdGeom.Xform.Define(stage, rcfg.root_usdpath)
        self.setup_robot_for_pose_movement(roborg, rcfg, pos, rot)

        add_reference_to_stage(rcfg.robot_usd_file_path, rcfg.robot_prim_path)
        apply_convex_decomposition_to_mesh_and_children(stage, rcfg.robot_prim_path)
        apply_diable_gravity_to_rigid_bodies(stage, rcfg.robot_prim_path)

        adjust_articulationAPI_location_if_needed(stage, rcfg.robot_prim_path)
        rcfg._articulation = Articulation(rcfg.artpath,f"mico-{ridx}")

        world = World.instance()
        world.scene.add(rcfg._articulation)

        return rcfg._articulation

    def teleport_robots_to_zeropos(self):
        for i in range(self._nrobots):
            rcfg = self.get_robot_config(i)
            if rcfg is not None:
                rcfg._articulation.set_joint_positions(rcfg.dof_zero_pos)

    def make_rmpflow(self, rob_idx, oblist = []):
        rcfg = self.get_robot_config(rob_idx)
        rmpflow = RmpFlow(
            robot_description_path = rcfg.rdf_path,
            urdf_path = rcfg.urdf_path,
            rmpflow_config_path = rcfg.rmp_config_path,
            end_effector_frame_name = rcfg.eeframe_name,
            maximum_substep_size = rcfg.max_step_size
        )
        quat = euler_angles_to_quat(rcfg.robot_rotvek)
        rmpflow.set_robot_base_pose(rcfg.start_robot_pos, quat)

        for ob in oblist:
            rmpflow.add_obstacle(ob)


        rmpflow.set_ignore_state_updates(True)
        rmpflow.visualize_collision_spheres()
        articulation_rmpflow = ArticulationMotionPolicy(rcfg._articulation,rmpflow)
        rcfg.rmpflow = rmpflow
        rcfg.articulation_rmpflow = articulation_rmpflow
        return rmpflow, articulation_rmpflow

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
        rcfg.rmpflow.visualize_collision_spheres()
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
           self._show_rmp_target = opt != "invisible"
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

    _colprims = None
    _matman = None

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
        stage = get_current_stage()
        if self._matman is None:
            self._matman = MatMan(stage)

        if self._colprims is None:
            self._colprims: List[Usd.Prim] = find_prims_by_name("collision_sphere")
        print(f"realize_collider_vis_opt:{opt} nspheres:{len(self._colprims)}")
        nfliped = 0
        nexcept = 0
        for prim in self._colprims:
            gprim = UsdGeom.Gprim(prim)
            try:
                if opt == "Red":
                    gprim.MakeVisible()
                    material = self._matman.GetMaterial("red")
                    UsdShade.MaterialBindingAPI(gprim).Bind(material)
                elif opt == "Glass":
                    gprim.MakeVisible()
                    material = self._matman.GetMaterial("Clear_Glass")
                    UsdShade.MaterialBindingAPI(gprim).Bind(material)
                elif opt == "Invisible":
                    gprim.MakeInvisible()
                nfliped += 1
            except:
                nexcept += 1
                pass
        print(f"Realize_collider_vis_opt changed:{nfliped} exceptions:{nexcept}")

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

        self.camviews = camviews
        return wintitle

    def set_stiffness_and_damping_for_all_joints(self, rcfg):
        # print(f"set_stiffness_and_damping_for_all_joints - {rcfg.robot_name} - {rcfg.robot_id}")
        if rcfg.stiffness>0:
            # print(f"    setting stiffness - {rcfg.robot_name} stiffness:{rcfg.stiffness}")
            set_stiffness_for_joints(rcfg.dof_paths, rcfg.stiffness)
        if rcfg.damping>0:
            # print(f"    setting damping - {rcfg.robot_name} damping:{rcfg.damping}")
            set_damping_for_joints(rcfg.dof_paths, rcfg.damping)

    def ensure_orimat(self):
        rc = self.get_robot_config(0)
        for rcfg in self._rcfg_list:
            if not hasattr(rcfg, "orimat"):
                rcfg.orimat = build_material_dict(self._stage, rcfg.robot_prim_path)

    def joint_check_robot(self,rcfg):
        degs = 180/np.pi
        art = rcfg._articulation
        pos = art.get_joint_positions()
        props = art.dof_properties
        # stiffs = props["stiffness"]
        # damps = props["damping"]
        for j,jn in enumerate(rcfg.dof_names):
            # stiff = stiffs[j]
            # damp = damps[j]
            jpos = degs*pos[j]
            llim = degs*rcfg.lower_dof_lim[j]
            ulim = degs*rcfg.upper_dof_lim[j]
            denom = ulim - llim
            if denom == 0:
                denom = 1
            pct = 100*(jpos - llim)/denom
            if pct<10 or 90>pct:
                clr = "green"

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
            case "Camera Viewports":
                if not hasattr(self, "camlist"):
                    return
                if len(self.camlist)==0:
                    carb.log_warn("No cameras found in camlist")
                    return
                self.wtit = self.make_camera_views()
                # ui.Workspace.show_window(self.wtit,True)
            case "Camera Viewports":
                self.joint_check()

    def get_scenario_actions(self):
        return ["Camera Viewports","Joint Check"]