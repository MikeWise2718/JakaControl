import math
import os
import numpy as np
import lula
from omni.isaac.motion_generation.lula.interface_helper import LulaInterfaceHelper
from .matman import MatMan
from pxr import Usd, UsdGeom, UsdShade, Gf
from typing import Tuple, List



from pxr import Sdf, UsdLux, UsdPhysics, Usd

from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.stage import get_current_stage

from omni.isaac.core.prims import XFormPrim

from omni.isaac.core.utils.extensions import get_extension_path_from_name

from omni.isaac.core.utils.nucleus import get_assets_root_path

from .senut import get_extension_path_from_name, cleanup_path
from .senut import find_prim_by_name, find_prims_by_name

import carb.settings

class robcfg:
    def __init__(self):
        pass

def get_robot_params_robcfg(robot_name):
    global robcfg

    assets_root_path = get_assets_root_path()
    mg_extension_path = get_extension_path_from_name("omni.isaac.motion_generation")
    jakacontrol_extension_path = cleanup_path(get_extension_path_from_name("JakaControl"))
    # robsjaka_extension_path = cleanup_path(get_extension_path_from_name("robs.jaka"))
    asimovjaka_extension_path = cleanup_path(get_extension_path_from_name("omni.asimov.jaka"))
    rmp_config_dir = cleanup_path(mg_extension_path + "/motion_policy_configs")

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

            rmp_mppath = rmp_config_dir + "/universal_robots"
            rdf_path = rmp_mppath + "/ur3e/rmpflow/ur3e_robot_description.yaml"
            urdf_path = rmp_mppath + "/ur3e/ur3e.urdf"
            rmp_config_path = rmp_mppath + "/ur3e/rmpflow/ur3e_rmpflow_config.yaml"
            eeframe_name = "tool0"
            max_step_size = 0.00334

        case "ur5e":
            robot_prim_path = "/ur5e"
            artpath = robot_prim_path
            path_to_robot_usd = assets_root_path+ "/Isaac/Robots/UniversalRobots/ur5e/ur5e.usd"
            mopo_robot_name = "UR5e"

            rmp_mppath = rmp_config_dir + "/universal_robots"
            rdf_path = rmp_mppath + "/ur5e/rmpflow/ur5e_robot_description.yaml"
            urdf_path = rmp_mppath + "/ur5e/ur5e.urdf"
            rmp_config_path = rmp_mppath + "/ur5e/rmpflow/ur5e_rmpflow_config.yaml"
            eeframe_name = "tool0"
            max_step_size = 0.00334

        case "ur10e":
            robot_prim_path = "/ur10e"
            artpath = robot_prim_path
            path_to_robot_usd = assets_root_path + "/Isaac/Robots/UniversalRobots/ur10e/ur10e.usd"
            mopo_robot_name = "UR10e"

            rmp_mppath = rmp_config_dir + "/universal_robots"
            rdf_path = rmp_mppath + "/ur10e/rmpflow/ur10e_robot_description.yaml"
            urdf_path = rmp_mppath + "/ur10e/ur10e.urdf"
            rmp_config_path = rmp_mppath + "/ur10e/rmpflow/ur10e_rmpflow_config.yaml"
            eeframe_name = "tool0"
            max_step_size = 0.00334
        case "ur10e-gripper":
            robot_prim_path = "/ur10e"
            artpath = robot_prim_path
            path_to_robot_usd = assets_root_path + "/Isaac/Robots/UniversalRobots/ur10e/ur10e.usd"
            mopo_robot_name = "UR10e"

            rmp_mppath = rmp_config_dir + "/universal_robots"
            rdf_path = rmp_mppath + "/ur10e/rmpflow/ur10e_robot_description.yaml"
            urdf_path = rmp_mppath + "/ur10e/ur10e.urdf"
            rmp_config_path = rmp_mppath + "/ur10e/rmpflow/ur10e_rmpflow_config.yaml"
            eeframe_name = "tool0"
            max_step_size = 0.00334
        case "ur10-suction-short":
            robot_prim_path = "/World/roborg/ur10_suction_short"
            artpath = robot_prim_path
            path_to_robot_usd = assets_root_path + "/Isaac/Robots/UR10/ur10_short_suction.usd"
            mopo_robot_name = "UR10-suction-short"

            rmp_mppath = rmp_config_dir
            rdf_path = rmp_mppath + "/ur10/rmpflow_suction/ur10_robot_description.yaml"
            urdf_path = rmp_mppath + "/ur10/ur10_robot_suction.urdf"
            rmp_config_path = rmp_mppath + "/ur10/rmpflow_suction/ur10_rmpflow_config.yaml"
            eeframe_name = "ee_link"
            max_step_size = 0.00334

        case "m0609":
            robot_prim_path = "/World/roborg/minicobo_v1_4"
            artpath = f"{robot_prim_path}/world"
            path_to_robot_usd = f"{jakacontrol_extension_path}/usd/jaka2.usda"
            mopo_robot_name = "Franka"

            rmp_mppath = f"{jakacontrol_extension_path}/JakaCtrl/motion_policy_configs/Doosan"
            rdf_path = rmp_mppath + "/m0609/rmpflow/m0609_robot_description.yaml"
            urdf_path = rmp_mppath + "/m0609/minicobo_v14.urdf"
            rmp_config_path = rmp_mppath + "/m0609/rmpflow/m0609_rmpflow_config.yaml"
            eeframe_name = "dummy_tcp"
            max_step_size = 0.00334


        case "jaka-minicobo-0":
            robot_prim_path = "/World/roborg/minicobo_v1_4"
            artpath = f"{robot_prim_path}/world"
            path_to_robot_usd = f"{jakacontrol_extension_path}/usd/jaka2.usda"
            mopo_robot_name = "Franka"

            rmp_mppath = f"{jakacontrol_extension_path}/JakaCtrl/motion_policy_configs/Jaka"
            rdf_path = rmp_mppath + "/minicobo/rmpflow/minicobo_robot_description.yaml"
            urdf_path = rmp_mppath + "/minicobo/minicobo_v14.urdf"
            rmp_config_path = rmp_mppath + "/minicobo/rmpflow/minicobo_rmpflow_config.yaml"
            eeframe_name = "dummy_tcp"
            max_step_size = 0.00334

        case "jaka-minicobo-2":
            robot_prim_path = "/World/roborg/minicobo_parallel_onrobot_rg2"
            artpath = f"{robot_prim_path}/minicobo_onrobot_rg2/world"
            path_to_robot_usd = f"{asimovjaka_extension_path}/usd/minicobo-parallel-onrobot-rg2-6.usda"
            mopo_robot_name = "RS007N"

            rmp_mppath = f"{jakacontrol_extension_path}/JakaCtrl/motion_policy_configs/Jaka"
            rdf_path = rmp_mppath + "/minicobo/rmpflow/minicobo_robot_description.yaml"
            urdf_path = rmp_mppath + "/minicobo/minicobo_v14_onrobot_rg2.urdf"
            rmp_config_path = rmp_mppath + "/minicobo/rmpflow/minicobo_rmpflow_config.yaml"
            eeframe_name = "gripper_center"
            max_step_size = 0.00334

        case "minicobo-rg2-high":
            robot_prim_path = "/World/roborg/minicobo_parallel_onrobot_rg2"
            artpath = f"{robot_prim_path}/minicobo_onrobot_rg2/world"
            path_to_robot_usd = f"{asimovjaka_extension_path}/usd/minicobo-parallel-onrobot-rg2-6.usda"
            mopo_robot_name = "RS007N"

            rmp_mppath = f"{asimovjaka_extension_path}/JakaCtrl/motion_policy_configs/Jaka"
            rdf_path = f"{asimovjaka_extension_path}/rdf/minicobo_robot_description.yaml"
            rmp_config_path = f"{asimovjaka_extension_path}/rdf/minicobo_rmpflow_config.yaml"
            urdf_path = f"{asimovjaka_extension_path}/urdf/minicobo_v14_onrobot_rg2.urdf"
            eeframe_name = "gripper_center"
            max_step_size = 0.00334

        case "jaka-minicobo-1":
            robot_prim_path = "/World/roborg/minicobo_v1_4"
            artpath = f"{robot_prim_path}/world"
            path_to_robot_usd = f"{jakacontrol_extension_path}/usd/jaka_v14_1.usda"
            mopo_robot_name = "RS007N"

            rmp_mppath = f"{jakacontrol_extension_path}/JakaCtrl/motion_policy_configs/Jaka"
            rdf_path = rmp_mppath + "/minicobo/rmpflow/minicobo_robot_description.yaml"
            urdf_path = rmp_mppath + "/minicobo/minicobo_v14_1.urdf"
            rmp_config_path = rmp_mppath + "/minicobo/rmpflow/minicobo_rmpflow_config.yaml"
            eeframe_name = "dummy_tcp"
            max_step_size = 0.00334

        case "minicobo-suction-dual" | "minicobo-dual-high":
            robot_prim_path = "/World/roborg/minicobo_suction_dual"
            artpath = f"{robot_prim_path}/minicobo_suction/world"
            path_to_robot_usd = f"{asimovjaka_extension_path}/usd/minicobo-suction-dual-4.usda"
            mopo_robot_name = "RS007N"

            rmp_mppath = f"{asimovjaka_extension_path}/JakaCtrl/motion_policy_configs/Jaka"
            rdf_path = f"{asimovjaka_extension_path}/rdf/minicobo_robot_description.yaml"
            rmp_config_path = f"{asimovjaka_extension_path}/rdf/minicobo_rmpflow_config.yaml"
            urdf_path = f"{asimovjaka_extension_path}/urdf/minicobo_v14_onrobot_rg2.urdf"
            eeframe_name = "gripper_center"
            max_step_size = 0.00334

        case "minicobo-suction":
            robot_prim_path = "/World/roborg/minicobo_suction_short"
            artpath = f"{robot_prim_path}/minicobo_suction/world"
            path_to_robot_usd = f"{asimovjaka_extension_path}/usd/minicobo-suction-2.usda"
            mopo_robot_name = "RS007N"
            rmp_mppath = f"{asimovjaka_extension_path}/JakaCtrl/motion_policy_configs/Jaka"
            rdf_path = f"{asimovjaka_extension_path}/rdf/minicobo_robot_description.yaml"
            rmp_config_path = f"{asimovjaka_extension_path}/rdf/minicobo_rmpflow_config.yaml"
            urdf_path = f"{asimovjaka_extension_path}/urdf/minicobo_v14_onrobot_rg2.urdf"
            eeframe_name = "gripper_center"
            max_step_size = 0.00334


        case "minicobo-suction-high":
            robot_prim_path = "/World/roborg/minicobo_suction_short"
            artpath = f"{robot_prim_path}/minicobo_suction/world"
            path_to_robot_usd = f"{asimovjaka_extension_path}/usd/minicobo-suction-2.usda"
            mopo_robot_name = "RS007N"
            rmp_mppath = f"{asimovjaka_extension_path}/JakaCtrl/motion_policy_configs/Jaka"
            rdf_path = f"{asimovjaka_extension_path}/rdf/minicobo_robot_description.yaml"
            rmp_config_path = f"{asimovjaka_extension_path}/rdf/minicobo_rmpflow_config.yaml"
            urdf_path = f"{asimovjaka_extension_path}/urdf/minicobo_v14_onrobot_rg2.urdf"
            eeframe_name = "gripper_center"
            max_step_size = 0.00334

        case "rs007n":
            robot_prim_path = "/World/roborg/khi_rs007n"
            artpath = robot_prim_path
            path_to_robot_usd = assets_root_path + "/Isaac/Robots/Kawasaki/RS007N/rs007n_onrobot_rg2.usd"
            mopo_robot_name = "RS007N"
            rmp_mppath = rmp_config_dir + "/Kawasaki"
            rdf_path = rmp_mppath + "/rs007n/rmpflow/rs007n_robot_description.yaml"
            urdf_path = rmp_mppath + "/rs007n/rs007n_onrobot_rg2.urdf"
            rmp_config_path = rmp_mppath + "/rs007n/rmpflow/rs007n_rmpflow_config.yaml"
            eeframe_name = "gripper_center"
            max_step_size = 0.00334

        case "franka":
            robot_prim_path = "/World/roborg/franka"
            artpath = robot_prim_path
            path_to_robot_usd = assets_root_path + "/Isaac/Robots/Franka/franka.usd"
            mopo_robot_name = "Franka"

            rdf_path = rmp_config_dir + "/franka/rmpflow/robot_descriptor.yaml"
            urdf_path = rmp_config_dir + "/franka/lula_franka_gen.urdf"
            rmp_config_path = rmp_config_dir + "/franka/rmpflow/franka_rmpflow_common.yaml"
            eeframe_name = "right_gripper"
            max_step_size = 0.00334

        case "fancy_franka":
            robot_prim_path = "/World/roborg/Fancy_Franka"
            artpath = robot_prim_path
            path_to_robot_usd = assets_root_path + "/Isaac/Robots/Franka/franka.usd"
            mopo_robot_name = "Franka"

            rdf_path = rmp_config_dir + "/franka/rmpflow/robot_descriptor.yaml"
            urdf_path = rmp_config_dir + "/franka/lula_franka_gen.urdf"
            rmp_config_path = rmp_config_dir + "/franka/rmpflow/franka_rmpflow_common.yaml"
            eeframe_name = "right_gripper"
            max_step_size = 0.00334

        case _:
            print("Bad robot type name", robot_name)
    rc = robcfg()
    rc.robot_name = robot_name
    rc.robot_prim_path = robot_prim_path
    rc.artpath = artpath
    rc.path_to_robot_usd = path_to_robot_usd
    rc.mopo_robot_name = mopo_robot_name
    rc.rdf_path = rdf_path
    rc.urdf_path = urdf_path
    rc.rmp_config_path = rmp_config_path
    rc.eeframe_name = eeframe_name
    rc.max_step_size = max_step_size

    return rc

def get_robot_params(robot_name):

    assets_root_path = get_assets_root_path()
    mg_extension_path = get_extension_path_from_name("omni.isaac.motion_generation")
    jakacontrol_extension_path = cleanup_path(get_extension_path_from_name("JakaControl"))
    # robsjaka_extension_path = cleanup_path(get_extension_path_from_name("robs.jaka"))
    asimovjaka_extension_path = cleanup_path(get_extension_path_from_name("omni.asimov.jaka"))
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
        case "ur10e-gripper":
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
        case "jaka-minicobo-0":
            robot_prim_path = "/World/roborg/minicobo_v1_4"
            artpath = f"{robot_prim_path}/world"
            path_to_robot_usd = f"{jakacontrol_extension_path}/usd/jaka2.usda"
            mopo_robot_name = "Franka"
        case "jaka-minicobo-2":
            robot_prim_path = "/World/roborg/minicobo_parallel_onrobot_rg2"
            artpath = f"{robot_prim_path}/minicobo_onrobot_rg2/world"
            path_to_robot_usd = f"{asimovjaka_extension_path}/usd/minicobo-parallel-onrobot-rg2-6.usda"
            mopo_robot_name = "RS007N"
        case "minicobo-rg2-high":
            robot_prim_path = "/World/roborg/minicobo_parallel_onrobot_rg2"
            artpath = f"{robot_prim_path}/minicobo_onrobot_rg2/world"
            path_to_robot_usd = f"{asimovjaka_extension_path}/usd/minicobo-parallel-onrobot-rg2-6.usda"
            mopo_robot_name = "RS007N"
        case "jaka-minicobo-1":
            robot_prim_path = "/World/roborg/minicobo_v1_4"
            artpath = f"{robot_prim_path}/world"
            path_to_robot_usd = f"{jakacontrol_extension_path}/usd/jaka_v14_1.usda"
            mopo_robot_name = "RS007N"
        case "minicobo-suction-dual" | "minicobo-dual-high":
            # robot_prim_path = "/World/roborg/minicobo_suction"
            # artpath = f"{robot_prim_path}/world"
            # path_to_robot_usd = f"{robsjaka_extension_path}/usd/minicobo-suction.usda"
            robot_prim_path = "/World/roborg/minicobo_suction_dual"
            artpath = f"{robot_prim_path}/minicobo_suction/world"
            path_to_robot_usd = f"{asimovjaka_extension_path}/usd/minicobo-suction-dual-4.usda"
            mopo_robot_name = "RS007N"
        case "minicobo-suction":
            # robot_prim_path = "/World/roborg/minicobo_suction"
            # artpath = f"{robot_prim_path}/world"
            # path_to_robot_usd = f"{robsjaka_extension_path}/usd/minicobo-suction.usda"
            robot_prim_path = "/World/roborg/minicobo_suction_short"
            artpath = f"{robot_prim_path}/minicobo_suction/world"
            path_to_robot_usd = f"{asimovjaka_extension_path}/usd/minicobo-suction-2.usda"
            mopo_robot_name = "RS007N"
        case "minicobo-suction-high":
            robot_prim_path = "/World/roborg/minicobo_suction_short"
            artpath = f"{robot_prim_path}/minicobo_suction/world"
            path_to_robot_usd = f"{asimovjaka_extension_path}/usd/minicobo-suction-2.usda"
            mopo_robot_name = "RS007N"
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
            # robot_prim_path = "/World/roborg/fancy_franka"
            robot_prim_path = "/World/roborg/Fancy_Franka"
            artpath = robot_prim_path
            path_to_robot_usd = assets_root_path + "/Isaac/Robots/Franka/franka.usd"
            # path_to_robot_usd = None
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
    rmp_config_dir = cleanup_path(mg_extension_path + "/motion_policy_configs")
    jakacontrol_extension_path = cleanup_path(get_extension_path_from_name("JakaControl"))
    asimovjaka_extension_path = cleanup_path(get_extension_path_from_name("omni.asimov.jaka"))

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
        case "ur10e-gripper":
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
        case "m0609":
            # urpath = rmp_config_dir + "/Jaka/"
            rmp_mppath = f"{jakacontrol_extension_path}/JakaCtrl/motion_policy_configs/Doosan"
            rdf_path = rmp_mppath + "/m0609/rmpflow/m0609_robot_description.yaml"
            urdf_path = rmp_mppath + "/m0609/minicobo_v14.urdf"
            rmp_config_path = rmp_mppath + "/m0609/rmpflow/m0609_rmpflow_config.yaml"
            eeframe_name = "dummy_tcp"
            max_step_size = 0.00334
        case "jaka-minicobo-0":
            # urpath = rmp_config_dir + "/Jaka/"
            rmp_mppath = f"{jakacontrol_extension_path}/JakaCtrl/motion_policy_configs/Jaka"
            rdf_path = rmp_mppath + "/minicobo/rmpflow/minicobo_robot_description.yaml"
            urdf_path = rmp_mppath + "/minicobo/minicobo_v14.urdf"
            rmp_config_path = rmp_mppath + "/minicobo/rmpflow/minicobo_rmpflow_config.yaml"
            eeframe_name = "dummy_tcp"
            max_step_size = 0.00334
        case "jaka-minicobo-2":
            # urpath = rmp_config_dir + "/Jaka/"
            rmp_mppath = f"{jakacontrol_extension_path}/JakaCtrl/motion_policy_configs/Jaka"
            rdf_path = rmp_mppath + "/minicobo/rmpflow/minicobo_robot_description.yaml"
            urdf_path = rmp_mppath + "/minicobo/minicobo_v14_onrobot_rg2.urdf"
            rmp_config_path = rmp_mppath + "/minicobo/rmpflow/minicobo_rmpflow_config.yaml"
            eeframe_name = "gripper_center"
            max_step_size = 0.00334
        case "minicobo-rg2-high":
            # urpath = rmp_config_dir + "/Jaka/"
            # rmp_mppath = f"{jakacontrol_extension_path}/JakaCtrl/motion_policy_configs/Jaka"
            # rdf_path = rmp_mppath + "/minicobo/rmpflow/minicobo_robot_description.yaml"
            # urdf_path = rmp_mppath + "/minicobo/minicobo_v14_onrobot_rg2.urdf"
            rmp_mppath = f"{asimovjaka_extension_path}/JakaCtrl/motion_policy_configs/Jaka"
            rdf_path = f"{asimovjaka_extension_path}/rdf/minicobo_robot_description.yaml"
            rmp_config_path = f"{asimovjaka_extension_path}/rdf/minicobo_rmpflow_config.yaml"
            urdf_path = f"{asimovjaka_extension_path}/urdf/minicobo_v14_onrobot_rg2.urdf"
            eeframe_name = "gripper_center"
            max_step_size = 0.00334
        case "jaka-minicobo-1":
            # urpath = rmp_config_dir + "/Jaka/"
            rmp_mppath = f"{jakacontrol_extension_path}/JakaCtrl/motion_policy_configs/Jaka"
            rdf_path = rmp_mppath + "/minicobo/rmpflow/minicobo_robot_description.yaml"
            urdf_path = rmp_mppath + "/minicobo/minicobo_v14_1.urdf"
            rmp_config_path = rmp_mppath + "/minicobo/rmpflow/minicobo_rmpflow_config.yaml"
            eeframe_name = "dummy_tcp"
            max_step_size = 0.00334
        case "minicobo-suction-dual" |"minicobo-dual-high":
            # rmp_mppath = f"{jakacontrol_extension_path}/JakaCtrl/motion_policy_configs/Jaka"
            # rdf_path = rmp_mppath + "/minicobo/rmpflow/minicobo_robot_description.yaml"
            # urdf_path = rmp_mppath + "/minicobo/minicobo_v14_onrobot_rg2.urdf"
            # rmp_config_path = rmp_mppath + "/minicobo/rmpflow/minicobo_rmpflow_config.yaml"
            # eeframe_name = "dual_gripper"
            rmp_mppath = f"{asimovjaka_extension_path}/JakaCtrl/motion_policy_configs/Jaka"
            rdf_path = f"{asimovjaka_extension_path}/rdf/minicobo_robot_description.yaml"
            rmp_config_path = f"{asimovjaka_extension_path}/rdf/minicobo_rmpflow_config.yaml"
            urdf_path = f"{asimovjaka_extension_path}/urdf/minicobo_v14_onrobot_rg2.urdf"
            eeframe_name = "gripper_center"
            # eeframe_name = "dual_gripper"
            # eeframe_name = "DUAL_SMC_ADAPTER_v002"

            max_step_size = 0.00334
        case "minicobo-suction" | "minicobo-suction-high":
            # urpath = rmp_config_dir + "/Jaka/"
            # rmp_mppath = f"{jakacontrol_extension_path}/JakaCtrl/motion_policy_configs/Jaka"
            # rdf_path = rmp_mppath + "/minicobo/rmpflow/minicobo_robot_description.yaml"
            # urdf_path = rmp_mppath + "/minicobo/minicobo_v14_onrobot_rg2.urdf"
            # rmp_config_path = rmp_mppath + "/minicobo/rmpflow/minicobo_rmpflow_config.yaml"
            rmp_mppath = f"{asimovjaka_extension_path}/JakaCtrl/motion_policy_configs/Jaka"
            rdf_path = f"{asimovjaka_extension_path}/rdf/minicobo_robot_description.yaml"
            rmp_config_path = f"{asimovjaka_extension_path}/rdf/minicobo_rmpflow_config.yaml"
            urdf_path = f"{asimovjaka_extension_path}/urdf/minicobo_v14_onrobot_rg2.urdf"
            eeframe_name = "gripper_center"
            max_step_size = 0.00334
        case "rs007n":
            rmp_mppath = rmp_config_dir + "/Kawasaki"
            rdf_path = rmp_mppath + "/rs007n/rmpflow/rs007n_robot_description.yaml"
            urdf_path = rmp_mppath + "/rs007n/rs007n_onrobot_rg2.urdf"
            rmp_config_path = rmp_mppath + "/rs007n/rmpflow/rs007n_rmpflow_config.yaml"
            eeframe_name = "gripper_center"
            max_step_size = 0.00334
        case "fancy_franka" | "franka":
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

def get_scenario_robots(scenario_name):
    match scenario_name:
        case "sinusoid-joint":
            rv = ["franka", "ur10e", "ur5e", "ur3e", "jaka-minicobo-0"]
        case "object-inspection":
            rv = ["franka", "ur10e", "ur5e", "ur3e", "jaka-minicobo-0","minicobo-dual-high"]
        case "franka-pick-and-place":
            rv = ["franka", "fancy_franka"]
        case "pick-and-place" | "rmpflow" | "object-inspection" | "inverse-kinematics":
            rv = ["franka", "fancy_franka","rs007n", "ur10-suction-short",
                  "jaka-minicobo-0", "jaka-minicobo-1",  "jaka-minicobo-2",
                  "minicobo-rg2-high", "minicobo-suction-dual", "minicobo-suction", "minicobo-suction-high", "minicobo-dual-high"]
        case "gripper":
            rv = ["cone","inverted-cone","sphere","cube","cube-yrot","cylinder","suction-short","suction-dual","suction-dual-0"]
        case _:
            rv = ["ur3e", "ur5e", "ur10e", "ur10e-gripper", "ur10-suction-short",
                  "jaka-minicobo-0","jaka-minicobo-1", "jaka-minicobo-2",
                  "minicobo-rg2-high","minicobo-suction-dual","minicobo-suction","minicobo-suction-high","minicobo-dual-high",
                  "rs007n", "franka", "fancy_franka", "m0609",
                  "cone","inverted-cone","sphere","cube","cube-yrot","cylinder","suction-short","suction-dual","suction-dual-0"]
    return rv

def can_handle_robot(scenario_name, robot_name):
    robs = get_scenario_robots(scenario_name)
    rv = robot_name in robs
    return rv

class ScenarioBase:
    def __init__(self):
        pass

    def get_robcfg(self, robot_name, ground_opt):
        robcfg = get_robot_params_robcfg(robot_name)
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

    def get_robot_config(self, robot_name, ground_opt):
        print(f"senut.get_robot_config robot_name:{robot_name} ground_opt:{ground_opt} ")
        self._cfg_robot_name = robot_name
        self._cfg_ground_opt = ground_opt
        (ok, robot_prim_path, artpath, path_to_robot_usd, mopo_robot_name) = get_robot_params(robot_name)
        config = get_robot_params_robcfg(robot_name)
        if (not ok):
            msg = f"Robot {robot_name} not found for get_robot_params in senut.py"
            carb.log_error(msg)
            print(msg)
            return
        self._cfg_robot_params_ok = ok
        self._cfg_robot_prim_path = robot_prim_path
        self._cfg_artpath = artpath
        self._cfg_path_to_robot_usd = path_to_robot_usd
        self._cfg_mopo_robot_name = mopo_robot_name

        self._cfg_mg_extension_path = get_extension_path_from_name("omni.isaac.motion_generation")
        self._cfg_rmp_config_dir = os.path.join(self._cfg_mg_extension_path, "motion_policy_configs")
        self._cfg_jc_extension_path = get_extension_path_from_name("JakaControl")

        (ok, rdf_path, urdf_path, rmp_config_path, eeframe_name, max_step_size) = get_robot_rmp_params(robot_name)
        if (not ok):
            msg = f"Robot {robot_name} not found for get_robot_rmp_params in senut.py"
            carb.log_error(msg)
            print(msg)
            return
        self._cfg_rdf_path = rdf_path
        self._cfg_urdf_path = urdf_path
        self._cfg_rmp_config_path = rmp_config_path
        self._cfg_eeframe_name = eeframe_name
        self._cfg_max_step_size = max_step_size

        if self._cfg_rdf_path=="" or self._cfg_urdf_path=="":
            msg = f"Robot {robot_name} rdf_path or urdf_path not specified"
            carb.log_warn(msg)
            print(msg)
            return

        if not os.path.isfile(self._cfg_rdf_path):
            msg = f"Robot {robot_name} rdf_path bad - file not found:{self._cfg_rdf_path}"
            carb.log_error(msg)
            print(msg)
            return

        if not os.path.isfile(self._cfg_urdf_path):
            msg = f"Robot {robot_name} urdf_path bad - file not found:{self._cfg_urdf_path}"
            carb.log_error(msg)
            print(msg)
            return

        if not os.path.isfile(self._cfg_rmp_config_path):
            msg = f"Robot {robot_name} rmp_config_path bad - file not found:{self._cfg_rmp_config_path}"
            carb.log_error(msg)
            print(msg)
            return

        try:
            self._cfg_robot_description = lula.load_robot(self._cfg_rdf_path, self._cfg_urdf_path)
            self._cfg_lulaHelper = LulaInterfaceHelper(self._cfg_robot_description)
        except Exception as e:
            msg = f"ScenarioBase - Robot {robot_name} lula.load_robot of rdf and urdf failed:{e}"
            carb.log_error(msg)
            print(msg)
        return

    def register_articulation(self, articulation):
        # this has to happen in post_load_scenario - some initialization must be happening before this
        # probably as a result of articuation being added to the world.scene
        self._articulation = articulation
        self._cfg_lower_joint_limits = self._articulation.dof_properties["lower"]
        self._cfg_upper_joint_limits = self._articulation.dof_properties["upper"]
        self._cfg_joint_names = self._articulation.dof_names
        self._cfg_njoints = self._articulation.num_dof
        self._cfg_joint_zero_pos = np.zeros(self._cfg_njoints)
        print("senut.register_articulation")
        print(f"{self._cfg_robot_name} - njoints:{self._cfg_njoints} lower:{self._cfg_lower_joint_limits} upper:{self._cfg_upper_joint_limits}")
        print(f"{self._cfg_robot_name} - {self._cfg_joint_names}")

    def can_handle_robot(self, robot_name):
        return True

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

    def scenario_action(self):
        pass

    def get_scenario_actions(self):
        return ["--None--"]

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

        prim = find_prim_by_name("end_effector")
        if prim is None:
            # msg = 'realize_eetarg_vis:prim "end_effector" not found'
            # carb.log_warn(msg)
            return
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
