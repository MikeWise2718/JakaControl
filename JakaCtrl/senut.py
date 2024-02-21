import math
import os

from pxr import Sdf, UsdLux, UsdPhysics

from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.stage import get_current_stage

from omni.isaac.core.prims import XFormPrim

from omni.isaac.core.utils.extensions import get_extension_path_from_name

import carb.settings

# Settings Utilities

_settings = None

def _init_settings():
    global _settings
    if _settings is None:
        _settings = carb.settings.get_settings()
    return _settings

SETTING_NAME = "/persistent/omni/jaka_control"

def get_setting(name, default, db=False):
    try:
        settings = _init_settings()
        key = f"{SETTING_NAME}/{name}"
        val = settings.get(key)
        if db:
            oval = val
            if oval is None:
                oval = "None"
        if val is None:
            val = default
        if db:
            print(f"get_setting {name} {oval} {val}")
    except Exception as e:
        val = default
        if db:
            print(f"Exception {e} in get_setting {name} {default} {val}")
    return val

def save_setting(name, value):
    settings = _init_settings()
    key = f"{SETTING_NAME}/{name}"
    settings.set(key, value)

# Misc Utilities

def truncf(number, digits) -> float:
    # Improve accuracy with floating point operations, to avoid truncate(16.4, 2) = 16.39 or truncate(-1.13, 2) = -1.12
    nbDecimals = len(str(number).split('.')[1])
    if nbDecimals <= digits:
        return number
    stepper = 10.0 ** digits
    return math.trunc(stepper * number) / stepper

def add_light_to_stage():
    """
    A new stage does not have a light by default.  This function creates a spherical light
    """
    sphereLight = UsdLux.SphereLight.Define(get_current_stage(), Sdf.Path("/World/SphereLight"))
    sphereLight.CreateRadiusAttr(2)
    sphereLight.CreateIntensityAttr(100000)
    XFormPrim(str(sphereLight.GetPath())).set_world_pose([6.5, 0, 12])


def get_robot_params(robot_name):
    ok = True
    match robot_name:
        case "ur3e":
            robot_prim_path = "/ur3e"
            artpath = robot_prim_path
            path_to_robot_usd = get_assets_root_path() + "/Isaac/Robots/UniversalRobots/ur3e/ur3e.usd"
        case "ur5e":
            robot_prim_path = "/ur5e"
            artpath = robot_prim_path
            path_to_robot_usd = get_assets_root_path() + "/Isaac/Robots/UniversalRobots/ur5e/ur5e.usd"
        case "ur10e":
            robot_prim_path = "/ur10e"
            artpath = robot_prim_path
            path_to_robot_usd = get_assets_root_path() + "/Isaac/Robots/UniversalRobots/ur10e/ur10e.usd"
        case "jaka-minicobo":
            robot_prim_path = "/minicobo_v1_4"
            artpath = f"{robot_prim_path}/world"
            path_to_robot_usd = "d:/nv/ov/exts/JakaControl/usd/jaka2.usda"
        case "rs007n":
            robot_prim_path = "/khi_rs007n"
            artpath = robot_prim_path
            path_to_robot_usd = get_assets_root_path() + "/Isaac/Robots/Kawasaki/RS007N/rs007n_onrobot_rg2.usd"
        case "franka":
            robot_prim_path = "/franka"
            artpath = robot_prim_path
            path_to_robot_usd = get_assets_root_path() + "/Isaac/Robots/Franka/franka.usd"
        case "fancy_franka":
            robot_prim_path = "/fancy_franka"
            artpath = robot_prim_path
            path_to_robot_usd = None
        case "jetbot":
            robot_prim_path = "/jetbot"
            artpath = robot_prim_path
            path_to_robot_usd = get_assets_root_path() + "/Isaac/Robots/Jetbot/jetbot.usd"
        case _:
            ok = False
            robot_prim_path = ""
            artpath = ""
            path_to_robot_usd = ""
    return (ok, robot_prim_path, artpath, path_to_robot_usd)

def get_robot_rmp_params(robot_name):

    mg_extension_path = get_extension_path_from_name("omni.isaac.motion_generation")
    rmp_config_dir = os.path.join(mg_extension_path, "motion_policy_configs")

    ok = True
    match robot_name:
        case "ur3e":
            rmp_mppath = rmp_config_dir + "/universal_robots/"
            rdf_path = rmp_mppath + "/ur3e/rmpflow/ur3e_robot_description.yaml"
            urdf_path = rmp_mppath + "/ur3e/ur3e.urdf"
            rmp_config_path = rmp_mppath + "/ur3e/rmpflow/ur3e_rmpflow_config.yaml"
            eeframe_name = "tool0"
            max_step_size = 0.00334
        case "ur5e":
            rmp_mppath = rmp_config_dir + "/universal_robots/"
            rdf_path = rmp_mppath + "/ur5e/rmpflow/ur5e_robot_description.yaml"
            urdf_path = rmp_mppath + "/ur5e/ur5e.urdf"
            rmp_config_path = rmp_mppath + "/ur5e/rmpflow/ur5e_rmpflow_config.yaml"
            eeframe_name = "tool0"
            max_step_size = 0.00334
        case "ur10e":
            rmp_mppath = rmp_config_dir + "/universal_robots/"
            rdf_path = rmp_mppath + "/ur10e/rmpflow/ur10e_robot_description.yaml"
            urdf_path = rmp_mppath + "/ur10e/ur10e.urdf"
            rmp_config_path = rmp_mppath + "/ur10e/rmpflow/ur10e_rmpflow_config.yaml"
            eeframe_name = "tool0"
            max_step_size = 0.00334
        case "rs007n":
            rmp_mppath = rmp_config_dir + "/Kawasaki/"
            rdf_path = rmp_mppath + "/rs007n/rmpflow/rs007n_robot_description.yaml"
            urdf_path = rmp_mppath + "/rs007n/rs007n_onrobot_rg2.urdf"
            rmp_config_path = rmp_mppath + "/rs007n/rmpflow/rs007n_rmpflow_config.yaml"
            eeframe_name = "gripper_center"
            max_step_size = 0.00334
        case "jaka-minicobo":
            # urpath = rmp_config_dir + "/Jaka/"
            rmp_mppath = "d:/nv/ov/exts/JakaControl/JakaCtrl/motion_policy_configs/Jaka/"
            rdf_path = rmp_mppath + "/minicobo/rmpflow/minicobo_robot_description.yaml"
            urdf_path = rmp_mppath + "/minicobo/minicobo_v14.urdf"
            rmp_config_path = rmp_mppath + "/minicobo/rmpflow/minicobo_rmpflow_config.yaml"
            eeframe_name = "dummy_tcp"
            max_step_size = 0.00334
        case "franka":
            rdf_path = rmp_config_dir + "/franka/rmpflow/robot_descriptor.yaml"
            urdf_path = rmp_config_dir + "/franka/lula_franka_gen.urdf"
            rmp_config_path = rmp_config_dir + "/franka/rmpflow/franka_rmpflow_common.yaml"
            eeframe_name = "right_gripper"
            max_step_size = 0.00334
        case _:
            ok = False
            rdf_path = ""
            urdf_path = ""
            rmp_config_path = ""
            eeframe_name = ""
            max_step_size = 0.00334
    return (ok, rdf_path, urdf_path, rmp_config_path, eeframe_name, max_step_size)

class ScenarioTemplate:
    def __init__(self):
        pass

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