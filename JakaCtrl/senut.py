import math
import os

from pxr import Sdf, UsdLux, UsdPhysics, Usd

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

def find_prims_by_name(prim_name: str):
    stage = get_current_stage()
    found_prims = []
    for prim in stage.Traverse():
        try:
            if prim.GetName().startswith(prim_name):
                found_prims.append(prim)
        except:
            pass
    return found_prims


def find_prim_by_name( prim_name: str) -> Usd.Prim:
    stage = get_current_stage()
    for prim in stage.Traverse():
        try:
            if prim.GetName() == prim_name:
                return prim
        except:
            pass
    return None

def set_stiffness_for_joint(joint_name, stiffness):
    stage = get_current_stage()
    prim = find_prim_by_name(joint_name)
    joint = UsdPhysics.DriveAPI.Get(prim, "angular")
    val = joint.GetStiffnessAttr()
    val.Set(stiffness)

def set_damping_for_joint(joint_name, damping):
    stage = get_current_stage()
    prim = find_prim_by_name(joint_name)
    joint = UsdPhysics.DriveAPI.Get(prim, "angular")
    val = joint.GetDampingAttr()
    val.Set(damping)

def set_stiffness_for_joints(joint_names, stiffness):
    for jp in joint_names:
        set_stiffness_for_joint(jp, stiffness)

def set_damping_for_joints(joint_names, damping):
    for jp in joint_names:
        set_damping_for_joint(jp, damping)

def adjust_joint_values(joint_names, valname, fak):
    for jp in joint_names:
        adjust_joint_value(jp, valname, fak)

def adjust_joint_value(joint_name, valname, fak):
    stage = get_current_stage()
    prim = find_prim_by_name(joint_name)
    joint = UsdPhysics.DriveAPI.Get(prim, "angular")
    if valname=="stiffness":
        val = joint.GetStiffnessAttr()
        newval = val.Get() * fak
        val.Set(newval)
    elif valname=="damping":
        val = joint.GetDampingAttr()
        newval = val.Get() * fak
        val.Set(newval)

def get_robot_params(robot_name):

    assets_root_path = get_assets_root_path()
    print("Get assets root path: ", assets_root_path)

    mg_extension_path = get_extension_path_from_name("omni.isaac.motion_generation")
    rmp_config_dir = os.path.join(mg_extension_path, "motion_policy_configs")
    jakacontrol_extension_path = get_extension_path_from_name("JakaControl")
    mopo_robot_name = ""
    path_to_robot_usd = ""
    artpath = ""
    robot_prim_path = ""

    ok = True
    match robot_name:
        case "ur3e":
            robot_prim_path = "/ur3e"
            artpath = robot_prim_path
            path_to_robot_usd = assets_root_path + "/Isaac/Robots/UniversalRobots/ur3e/ur3e.usd"
            mopo_robot_name = "UR3e"
        case "ur5e":
            robot_prim_path = "/ur5e"
            artpath = robot_prim_path
            path_to_robot_usd = assets_root_path+ "/Isaac/Robots/UniversalRobots/ur5e/ur5e.usd"
            mopo_robot_name = "UR5e"
        case "ur10e":
            robot_prim_path = "/ur10e"
            artpath = robot_prim_path
            path_to_robot_usd = assets_root_path + "/Isaac/Robots/UniversalRobots/ur10e/ur10e.usd"
            mopo_robot_name = "UR10e"
        case "ur10-suction-short":
            robot_prim_path = "/World/roborg/ur10_suction_short"
            artpath = robot_prim_path
            # path_to_robot_usd = get_assets_root_path() + "/Isaac/Robots/UR10/ur10_short_suction.usd"
            path_to_robot_usd = assets_root_path + "/Isaac/Robots/UR10/ur10_short_suction.usd"
            print("path_to_robot_usd", path_to_robot_usd)
            mopo_robot_name = "UR10-suction-short"
        case "jaka-minicobo":
            robot_prim_path = "/World/roborg/minicobo_v1_4"
            artpath = f"{robot_prim_path}/world"
            path_to_robot_usd = f"{jakacontrol_extension_path}/usd/jaka2.usda"
            mopo_robot_name = "Franka"
        case "jaka-minicobo-1":
            robot_prim_path = "/World/roborg/minicobo_v1_4"
            artpath = f"{robot_prim_path}/world"
            path_to_robot_usd = f"{jakacontrol_extension_path}/usd/jaka_v14_1.usda"
            mopo_robot_name = "Franka"
        case "rs007n":
            robot_prim_path = "/World/roborg/khi_rs007n"
            artpath = robot_prim_path
            path_to_robot_usd = assets_root_path + "/Isaac/Robots/Kawasaki/RS007N/rs007n_onrobot_rg2.usd"
            mopo_robot_name = "RS007N"
        case "franka":
            robot_prim_path = "/World/roborg/franka"
            artpath = robot_prim_path
            path_to_robot_usd = assets_root_path + "/Isaac/Robots/Franka/franka.usd"
            mopo_robot_name = "Franka"
        case "fancy_franka":
            robot_prim_path = "/fancy_franka"
            artpath = robot_prim_path
            path_to_robot_usd = None
            mopo_robot_name = "Franka"
        case "jetbot":
            robot_prim_path = "/jetbot"
            artpath = robot_prim_path
            path_to_robot_usd = assets_root_path + "/Isaac/Robots/Jetbot/jetbot.usd"
            mopo_robot_name = "Jetbot"
        case _:
            ok = False
            robot_prim_path = ""
            artpath = ""
            path_to_robot_usd = ""
            mopo_robot_name = ""
    return (ok, robot_prim_path, artpath, path_to_robot_usd, mopo_robot_name)

def get_robot_rmp_params(robot_name):

    mg_extension_path = get_extension_path_from_name("omni.isaac.motion_generation")
    rmp_config_dir = os.path.join(mg_extension_path, "motion_policy_configs")
    jakacontrol_extension_path = get_extension_path_from_name("JakaControl")

    ok = True
    match robot_name:
        case "ur3e":
            rmp_mppath = rmp_config_dir + "/universal_robots"
            rdf_path = rmp_mppath + "/ur3e/rmpflow/ur3e_robot_description.yaml"
            urdf_path = rmp_mppath + "/ur3e/ur3e.urdf"
            rmp_config_path = rmp_mppath + "/ur3e/rmpflow/ur3e_rmpflow_config.yaml"
            eeframe_name = "tool0"
            max_step_size = 0.00334
        case "ur5e":
            rmp_mppath = rmp_config_dir + "/universal_robots"
            rdf_path = rmp_mppath + "/ur5e/rmpflow/ur5e_robot_description.yaml"
            urdf_path = rmp_mppath + "/ur5e/ur5e.urdf"
            rmp_config_path = rmp_mppath + "/ur5e/rmpflow/ur5e_rmpflow_config.yaml"
            eeframe_name = "tool0"
            max_step_size = 0.00334
        case "ur10e":
            rmp_mppath = rmp_config_dir + "/universal_robots"
            rdf_path = rmp_mppath + "/ur10e/rmpflow/ur10e_robot_description.yaml"
            urdf_path = rmp_mppath + "/ur10e/ur10e.urdf"
            rmp_config_path = rmp_mppath + "/ur10e/rmpflow/ur10e_rmpflow_config.yaml"
            eeframe_name = "tool0"
            max_step_size = 0.00334
        case "ur10-suction-short":
            rmp_mppath = rmp_config_dir
            rdf_path = rmp_mppath + "/ur10/rmpflow_suction/ur10_robot_description.yaml"
            urdf_path = rmp_mppath + "/ur10/ur10_robot_suction.urdf"
            rmp_config_path = rmp_mppath + "/ur10/rmpflow_suction/ur10_rmpflow_config.yaml"
            eeframe_name = "ee_link"
            max_step_size = 0.00334
        case "rs007n":
            rmp_mppath = rmp_config_dir + "/Kawasaki"
            rdf_path = rmp_mppath + "/rs007n/rmpflow/rs007n_robot_description.yaml"
            urdf_path = rmp_mppath + "/rs007n/rs007n_onrobot_rg2.urdf"
            rmp_config_path = rmp_mppath + "/rs007n/rmpflow/rs007n_rmpflow_config.yaml"
            eeframe_name = "gripper_center"
            max_step_size = 0.00334
        case "jaka-minicobo":
            # urpath = rmp_config_dir + "/Jaka/"
            rmp_mppath = f"{jakacontrol_extension_path}/JakaCtrl/motion_policy_configs/Jaka"
            rdf_path = rmp_mppath + "/minicobo/rmpflow/minicobo_robot_description.yaml"
            urdf_path = rmp_mppath + "/minicobo/minicobo_v14.urdf"
            rmp_config_path = rmp_mppath + "/minicobo/rmpflow/minicobo_rmpflow_config.yaml"
            eeframe_name = "dummy_tcp"
            max_step_size = 0.00334
        case "m0609":
            # urpath = rmp_config_dir + "/Jaka/"
            rmp_mppath = f"{jakacontrol_extension_path}/JakaCtrl/motion_policy_configs/Doosan"
            rdf_path = rmp_mppath + "/m0609/rmpflow/m0609_robot_description.yaml"
            urdf_path = rmp_mppath + "/m0609/minicobo_v14.urdf"
            rmp_config_path = rmp_mppath + "/m0609/rmpflow/m0609_rmpflow_config.yaml"
            eeframe_name = "dummy_tcp"
            max_step_size = 0.00334
        case "jaka-minicobo-1":
            # urpath = rmp_config_dir + "/Jaka/"
            rmp_mppath = f"{jakacontrol_extension_path}/JakaCtrl/motion_policy_configs/Jaka"
            rdf_path = rmp_mppath + "/minicobo/rmpflow/minicobo_robot_description.yaml"
            urdf_path = rmp_mppath + "/minicobo/minicobo_v14_1.urdf"
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

    def get_robot_config(self, robot_name, ground_opt):
        self._cfg_robot_name = robot_name
        self._cfg_ground_opt = ground_opt
        (ok, robot_prim_path, artpath, path_to_robot_usd, mopo_robot_name) = get_robot_params(robot_name)
        self._cfg_robot_params_ok = ok
        self._cfg_robot_prim_path = robot_prim_path
        self._cfg_artpath = artpath
        self._cfg_path_to_robot_usd = path_to_robot_usd
        self._cfg_mopo_robot_name = mopo_robot_name

        self._cfg_mg_extension_path = get_extension_path_from_name("omni.isaac.motion_generation")
        self._cfg_rmp_config_dir = os.path.join(self._cfg_mg_extension_path, "motion_policy_configs")
        self._cfg_jc_extension_path = get_extension_path_from_name("JakaControl")

        (ok, rdf_path, urdf_path, rmp_config_path, eeframe_name, max_step_size) = get_robot_rmp_params(robot_name)
        self._cfg_rdf_path = rdf_path
        self._cfg_urdf_path = urdf_path
        self._cfg_eeframe_name = eeframe_name
        self._cfg_max_step_size = max_step_size

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

    def action(self):
        pass

    def get_actions(self):
        return [""]
