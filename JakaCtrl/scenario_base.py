import os
import numpy as np
import lula
from omni.isaac.motion_generation.lula.interface_helper import LulaInterfaceHelper
from .matman import MatMan
from pxr import Usd, UsdGeom, UsdShade, Gf
from typing import Tuple, List

from omni.isaac.core.utils.stage import get_current_stage

from omni.isaac.core.prims import XFormPrim

from .scenario_robot_configs import get_robot_config, init_configs

from .senut import find_prim_by_name, find_prims_by_name
from .senut import apply_convex_decomposition_to_mesh_and_children, apply_material_to_prim_and_children

import carb.settings


class ScenarioBase:
    rmpactive = True

    def __init__(self):
        self._scenario_name = "empty scenario"
        self._secnario_desc = "description from ScenarioBase class"
        self._nrobots = 0
        self._stage = get_current_stage()
        self.camlist = {}
        self.rmpactive = True
        init_configs()
        pass

    @staticmethod
    def get_scenario_names():
        rv = [ "inverse-kinematics","gripper","rmpflow","object-inspection",
            "sinusoid-joint","franka-pick-and-place","pick-and-place"]
        return rv

    @staticmethod
    def get_default_scenario():
        rv = [ "inverse-kinematics","gripper","rmpflow","object-inspection",
            "sinusoid-joint","franka-pick-and-place","pick-and-place"]
        return rv

    @staticmethod
    def get_scenario_desc(scenario_name):
        match scenario_name:
            case "sinusoid-joint":
                rv = "Move robot through its joints in a sinusoid - from Nvidia example."
            case "object-inspection":
                rv = "Two Jaka Minicobo robots - Object Inspection Scenario"
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
        robcfg = get_robot_config(robot_name, skiplula=True)
        return robcfg.desc

    @staticmethod
    def get_scenario_robots(scenario_name):
        match scenario_name:
            case "sinusoid-joint":
                rv = ["franka", "ur10e", "ur5e", "ur3e", "jaka-minicobo-0"]
            case "object-inspection":
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


    def get_robcfg(self, robot_name, ground_opt):
        robcfg = get_robot_config(robot_name)
        robcfg.ground_opt = ground_opt

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

    camlist = {}

    def add_camera_to_camlist(self, cam_name, cam_display_name, campath):
        self.camlist[cam_name] = {}
        self.camlist[cam_name]["name"] = cam_name
        self.camlist[cam_name]["display_name"] = cam_display_name
        self.camlist[cam_name]["usdpath"] = campath


    def register_articulation(self, articulation, rc=None):
        # this has to happen in post_load_scenario - some initialization must be happening before this
        # probably as a result of articuation being added to the world.scene
        # self._articulation = articulation
        if rc is None:
            rc = self._robcfg
        rc._articulation = articulation
        rc.lower_joint_limits = self._articulation.dof_properties["lower"]
        rc.upper_joint_limits = self._articulation.dof_properties["upper"]
        rc.njoints = self._articulation.num_dof
        rc.joint_names = self._articulation.dof_names
        rc.joint_zero_pos = np.zeros(self._robcfg.njoints)
        print("senut.register_articulation")
        # print(f"{self._cfg_robot_name} - njoints:{self._cfg_njoints} lower:{self._cfg_lower_joint_limits} upper:{self._cfg_upper_joint_limits}")
        # print(f"{self._cfg_robot_name} - {self._cfg_joint_names}")

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

    def realize_robot_skin(self, skinopt):
        match skinopt:
            case "Default":
                carb.log_warn("realize_robot_skin - skinopt is default - no action taken")
                return
            case "Clear Glass":
                mat1 = mat2 =  "Clear_Glass"
            case "Red Glass":
                mat1 = mat2 =  "Red_Glass"
            case "Green Glass":
                mat1 = mat2 =  "Green_Glass"
            case "Blue Glass":
                mat1 = mat2 =  "Blue_Glass"
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
        if hasattr(self, "_robcfg"):
            apply_material_to_prim_and_children(self._stage, self._matman, mat1, self._robcfg.robot_prim_path)
            didone = True
        if hasattr(self, "_robcfg1"):
            apply_material_to_prim_and_children(self._stage, self._matman, mat2, self._robcfg1.robot_prim_path)
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
                    # lula = Usd.GetPrimAtPath("lula")
                    # lula.GetVisibilityAttr().Set(False)
                    gprim.MakeInvisible()
                    # UsdGeom.Imageable(prim).MakeInvisible()
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

    def get_robot_config(self, i):
        if i == 0:
            if hasattr(self, "_robcfg"):
                return self._robcfg
        elif i == 1:
            if hasattr(self, "_robcfg1"):
                return self._robcfg1
        else:
            return None


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
                pass

    def get_scenario_actions(self):
        return ["Camera Viewports"]