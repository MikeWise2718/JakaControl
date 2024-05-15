import os
import numpy as np
import lula
from omni.isaac.motion_generation.lula.interface_helper import LulaInterfaceHelper

from omni.isaac.core.utils.nucleus import get_assets_root_path


from omni.isaac.core.utils.extensions import get_extension_path_from_name

from omni.isaac.core.utils.nucleus import get_assets_root_path

from .senut import get_extension_path_from_name, cleanup_path

import carb.settings

class robcfg:
    def __init__(self):
        pass

def create_and_populate_robot_config(robot_name, robot_root_usdpath="/world/roborg", skiplula=False):
    global robcfg

    assets_root_dir = get_assets_root_path()
    mg_extension_dir = get_extension_path_from_name("omni.isaac.motion_generation")
    current_extension_dir = cleanup_path(get_extension_path_from_name("JakaControl"))
    # robsjaka_extension_path = cleanup_path(get_extension_path_from_name("robs.jaka"))
    asimovjaka_extension_dir = cleanup_path(get_extension_path_from_name("omni.asimov.jaka"))
    rmp_config_dir = cleanup_path(mg_extension_dir + "/motion_policy_configs")

    mopo_robot_name = ""
    robot_usd_file_path = ""
    artpath = ""
    robot_prim_path = ""
    stiffness = -1
    damping = -1
    desc = "no description"
    camera_root = ""
    grippername = "none"
    prefered_target = "cuboid"
    pp_controller = "none"

    ok = True
    match robot_name:
        case "cone"|"inverted-cone"|"sphere"|"cube"|"cube-yrot"|"cylinder"|"suction-short"|"suction-dual"|"suction-dual-0":
            robot_prim_path = f"/{robot_name}"
            artpath = robot_prim_path
            robot_usd_file_path = "None"
            mopo_robot_name = robot_name

            rmp_param_dir = "None"
            rdf_path = "None"
            urdf_path = "None"
            rmp_config_path = "None"
            eeframe_name = robot_prim_path
            max_step_size = 0.00334

            grippername = "none"

            mfg = "None"
            model = "None"
            desc = "Gripper testing proxy robot"

        case "ur3e":
            robot_prim_path = "/ur3e"
            artpath = robot_prim_path
            robot_usd_file_path = assets_root_dir + "/Isaac/Robots/UniversalRobots/ur3e/ur3e.usd"
            mopo_robot_name = "UR3e"

            rmp_param_dir = rmp_config_dir + "/universal_robots"
            rdf_path = rmp_param_dir + "/ur3e/rmpflow/ur3e_robot_description.yaml"
            urdf_path = rmp_param_dir + "/ur3e/ur3e.urdf"
            rmp_config_path = rmp_param_dir + "/ur3e/rmpflow/ur3e_rmpflow_config.yaml"
            eeframe_name = "tool0"
            max_step_size = 0.00334

            grippername = "none"

            mfg = "Universal Robots"
            model = "UR3e"
            desc = "Universal Robots UR3e"

        case "ur5e":
            robot_prim_path = "/ur5e"
            artpath = robot_prim_path
            robot_usd_file_path = assets_root_dir+ "/Isaac/Robots/UniversalRobots/ur5e/ur5e.usd"
            mopo_robot_name = "UR5e"

            rmp_param_dir = rmp_config_dir + "/universal_robots"
            rdf_path = rmp_param_dir + "/ur5e/rmpflow/ur5e_robot_description.yaml"
            urdf_path = rmp_param_dir + "/ur5e/ur5e.urdf"
            rmp_config_path = rmp_param_dir + "/ur5e/rmpflow/ur5e_rmpflow_config.yaml"
            eeframe_name = "tool0"
            max_step_size = 0.00334

            grippername = "none"

            mfg = "Universal Robots"
            model = "UR5e"
            grippername = "none"
            desc = "Universal Robots UR5e"

        case "ur10e":
            robot_prim_path = "/ur10e"
            artpath = robot_prim_path
            robot_usd_file_path = assets_root_dir + "/Isaac/Robots/UniversalRobots/ur10e/ur10e.usd"
            mopo_robot_name = "UR10e"

            rmp_param_dir = rmp_config_dir + "/universal_robots"
            rdf_path = rmp_param_dir + "/ur10e/rmpflow/ur10e_robot_description.yaml"
            urdf_path = rmp_param_dir + "/ur10e/ur10e.urdf"
            rmp_config_path = rmp_param_dir + "/ur10e/rmpflow/ur10e_rmpflow_config.yaml"
            eeframe_name = "tool0"
            max_step_size = 0.00334

            grippername = "none"

            mfg = "Universal Robots"
            model = "UR10e"
            desc = "Universal Robots UR10e"

        case "ur10e-gripper":
            robot_prim_path = "/ur10e"
            artpath = robot_prim_path
            robot_usd_file_path = assets_root_dir + "/Isaac/Robots/UniversalRobots/ur10e/ur10e.usd"
            mopo_robot_name = "UR10e"

            rmp_param_dir = rmp_config_dir + "/universal_robots"
            rdf_path = rmp_param_dir + "/ur10e/rmpflow/ur10e_robot_description.yaml"
            urdf_path = rmp_param_dir + "/ur10e/ur10e.urdf"
            rmp_config_path = rmp_param_dir + "/ur10e/rmpflow/ur10e_rmpflow_config.yaml"
            eeframe_name = "tool0"
            max_step_size = 0.00334

            grippername = "rg2"
            pp_controller = "ur-rg2"


            mfg = "Universal Robots"
            model = "UR10e"
            desc = "Universal Robots UR10e with Parallel Gripper"

        case "ur10-suction-short":
            robot_prim_path = "/World/roborg/ur10_suction_short"
            artpath = robot_prim_path
            robot_usd_file_path = assets_root_dir + "/Isaac/Robots/UR10/ur10_short_suction.usd"
            mopo_robot_name = "UR10-suction-short"

            rmp_param_dir = rmp_config_dir
            rdf_path = rmp_param_dir + "/ur10/rmpflow_suction/ur10_robot_description.yaml"
            urdf_path = rmp_param_dir + "/ur10/ur10_robot_suction.urdf"
            rmp_config_path = rmp_param_dir + "/ur10/rmpflow_suction/ur10_rmpflow_config.yaml"
            eeframe_name = "ee_link"
            max_step_size = 0.00334

            grippername = "short suction"
            pp_controller = "ur-ss"


            mfg = "Universal Robots"
            model = "UR10"
            desc = "Universal Robots UR10 with Suction Gripper"

        case "m0609":
            robot_prim_path = "/World/roborg/minicobo_v1_4"
            artpath = f"{robot_prim_path}/world"
            robot_usd_file_path = f"{current_extension_dir}/usd/jaka2.usda"
            mopo_robot_name = "Franka"

            rmp_param_dir = f"{current_extension_dir}/JakaCtrl/motion_policy_configs/Doosan"
            rdf_path = rmp_param_dir + "/m0609/rmpflow/m0609_robot_description.yaml"
            urdf_path = rmp_param_dir + "/m0609/minicobo_v14.urdf"
            rmp_config_path = rmp_param_dir + "/m0609/rmpflow/m0609_rmpflow_config.yaml"
            eeframe_name = "dummy_tcp"
            max_step_size = 0.00334

            grippername = "none"

            mfg = "Doosan"
            model = "M0609"
            desc = "Doosan M0609"

        case "jaka-minicobo-0":
            robot_prim_path = "/World/roborg/minicobo_v1_4"
            artpath = f"{robot_prim_path}/world"
            robot_usd_file_path = f"{current_extension_dir}/usd/jaka2.usda"
            mopo_robot_name = "Franka"

            rmp_param_dir = f"{current_extension_dir}/JakaCtrl/motion_policy_configs/Jaka"
            rdf_path = rmp_param_dir + "/minicobo/rmpflow/minicobo_robot_description_0.yaml"
            urdf_path = rmp_param_dir + "/minicobo/minicobo_v14.urdf"
            rmp_config_path = rmp_param_dir + "/minicobo/rmpflow/minicobo_rmpflow_config.yaml"
            eeframe_name = "dummy_tcp"
            stiffness = 400
            damping = 40
            max_step_size = 0.00334

            grippername = "none"

            mfg = "Jaka"
            model = "Minicobo"
            desc = "Jaka Minicobo without a gripper"

        case "jaka-minicobo-1":
            # robot_prim_path = "/World/roborg/minicobo_v1_4"
            robot_prim_path = f"{robot_root_usdpath}/minicobo_v1_4"
            artpath = f"{robot_prim_path}/world"
            robot_usd_file_path = f"{current_extension_dir}/usd/jaka_v14_1.usda"
            mopo_robot_name = "RS007N"

            rmp_param_dir = f"{current_extension_dir}/JakaCtrl/motion_policy_configs/Jaka"
            rdf_path = rmp_param_dir + "/minicobo/rmpflow/minicobo_robot_description_0.yaml"
            urdf_path = rmp_param_dir + "/minicobo/minicobo_v14_1.urdf"
            rmp_config_path = rmp_param_dir + "/minicobo/rmpflow/minicobo_rmpflow_config.yaml"
            eeframe_name = "dummy_tcp"
            stiffness = 400
            damping = 40
            max_step_size = 0.00334

            grippername = "dual sucker"
            prefered_target = "phone_slab"
            pp_controller = "jaka-ds"

            mfg = "Jaka"
            model = "Minicobo"
            desc = "Jaka Minicobo with a dual sucker gripper (old)"

        case "jaka-minicobo-1a":
            # robot_prim_path = "/World/roborg/minicobo_v1_4"
            robot_prim_path = f"{robot_root_usdpath}/ring/minicobo_v1_4"
            artpath = f"{robot_prim_path}/world"
            # robot_usd_file_path = f"{jakacontrol_extension_dir}/usd/jaka_v14_1.usda"
            robot_usd_file_path = f"{current_extension_dir}/JakaCtrl/motion_policy_configs/Jaka/minicobo/minicobo_v14_1a/minicobo_v14_1a.usda"

            mopo_robot_name = "RS007N"

            rmp_param_dir = f"{current_extension_dir}/JakaCtrl/motion_policy_configs/Jaka"
            rdf_path = rmp_param_dir + "/minicobo/rmpflow/minicobo_robot_description_0.yaml"
            urdf_path = rmp_param_dir + "/minicobo/minicobo_v14_1a.urdf"
            rmp_config_path = rmp_param_dir + "/minicobo/rmpflow/minicobo_rmpflow_config.yaml"
            eeframe_name = "tool0"
            stiffness = 400
            damping = 40
            max_step_size = 0.00334

            camera_root = f"{robot_prim_path}/dummy_tcp"

            grippername = "dual sucker"
            prefered_target = "moto50mp"
            pp_controller = "jaka-ds"


            mfg = "Jaka"
            model = "Minicobo"
            desc = "Jaka Minicobo with a dual sucker gripper"

        case "minicobo-dual-sucker":
            # robot_prim_path = "/World/roborg/minicobo_v1_4"
            robot_prim_path = f"{robot_root_usdpath}/ring/minicobo_v1_4"
            artpath = f"{robot_prim_path}/world"
            # robot_usd_file_path = f"{jakacontrol_extension_dir}/usd/jaka_v14_1.usda"
            robot_usd_file_path = f"{current_extension_dir}/JakaCtrl/motion_policy_configs/Jaka/minicobo/minicobo_dual_sucker/minicobo_dual_sucker1.usda"

            mopo_robot_name = "RS007N"

            rmp_param_dir = f"{current_extension_dir}/JakaCtrl/motion_policy_configs/Jaka"
            rdf_path = rmp_param_dir + "/minicobo/rmpflow/minicobo_robot_description_dual_sucker.yaml"
            urdf_path = rmp_param_dir + "/minicobo/minicobo_dual_sucker.urdf"
            rmp_config_path = rmp_param_dir + "/minicobo/rmpflow/minicobo_rmpflow_config_dual_sucker.yaml"
            eeframe_name = "tool0"
            stiffness = 400
            damping = 40
            max_step_size = 0.00334

            camera_root = f"{robot_prim_path}/dummy_tcp"

            grippername = "dual sucker"
            prefered_target = "phone_slab"
            pp_controller = "jaka-ds"

            mfg = "Jaka"
            model = "Minicobo"
            desc = "Jaka Minicobo with a dual sucker gripper"

        case "jaka-minicobo-2":
            robot_prim_path = "/World/roborg/minicobo_parallel_onrobot_rg2"
            artpath = f"{robot_prim_path}/minicobo_onrobot_rg2/world"
            robot_usd_file_path = f"{asimovjaka_extension_dir}/usd/minicobo-parallel-onrobot-rg2-6.usda"
            mopo_robot_name = "RS007N"

            rmp_param_dir = f"{current_extension_dir}/JakaCtrl/motion_policy_configs/Jaka"
            rdf_path = rmp_param_dir + "/minicobo/rmpflow/minicobo_robot_description.yaml"
            # urdf_path = rmp_param_dir + "/minicobo/minicobo_v14_onrobot_rg2.urdf"
            urdf_path = rmp_param_dir + "/minicobo/minicobo_v14.urdf"
            rmp_config_path = rmp_param_dir + "/minicobo/rmpflow/minicobo_rmpflow_config.yaml"
            eeframe_name = "gripper_center"
            stiffness = 400
            damping = 40
            max_step_size = 0.00334

            grippername = "rg2"
            pp_controller = "jaka-rg2"

            mfg = "Jaka"
            model = "Minicobo"
            desc = "Jaka Minicobo with a ?"

        case "minicobo-rg2-high":
            robot_prim_path = "/World/roborg/minicobo_parallel_onrobot_rg2"
            artpath = f"{robot_prim_path}/minicobo_onrobot_rg2/world"
            robot_usd_file_path = f"{asimovjaka_extension_dir}/usd/minicobo-parallel-onrobot-rg2-6.usda"
            mopo_robot_name = "RS007N"

            rmp_param_dir = f"{asimovjaka_extension_dir}/JakaCtrl/motion_policy_configs/Jaka"
            rdf_path = f"{asimovjaka_extension_dir}/rdf/minicobo_robot_description.yaml"
            rmp_config_path = f"{asimovjaka_extension_dir}/rdf/minicobo_rmpflow_config.yaml"
            urdf_path = f"{asimovjaka_extension_dir}/urdf/minicobo_v14_onrobot_rg2.urdf"
            eeframe_name = "gripper_center"
            max_step_size = 0.00334
            stiffness = 400
            damping = 40

            grippername = "rg2"
            pp_controller = "jaka-rg2"

            mfg = "Jaka"
            model = "Minicobo"
            desc = "Jaka Minicobo with a RG2 parallel gripper"


        case "minicobo-suction-dual" | "minicobo-dual-high":
            robot_prim_path = "/World/roborg/minicobo_suction_dual"
            artpath = f"{robot_prim_path}/minicobo_suction/world"
            robot_usd_file_path = f"{asimovjaka_extension_dir}/usd/minicobo-suction-dual-4.usda"
            mopo_robot_name = "RS007N"

            rmp_param_dir = f"{asimovjaka_extension_dir}/JakaCtrl/motion_policy_configs/Jaka"
            rdf_path = f"{asimovjaka_extension_dir}/rdf/minicobo_robot_description.yaml"
            urdf_path = f"{asimovjaka_extension_dir}/urdf/minicobo_v14_onrobot_rg2.urdf"
            rmp_config_path = f"{asimovjaka_extension_dir}/rdf/minicobo_rmpflow_config.yaml"
            eeframe_name = "gripper_center"
            stiffness = 400
            damping = 40
            max_step_size = 0.00334

            grippername = "dual sucker"
            prefered_target = "phone_slab"
            pp_controller = "jaka-ds"

            mfg = "Jaka"
            model = "Minicobo"
            desc = "Jaka Minicobo with a dual suction gripper"

        case "minicobo-suction":
            robot_prim_path = "/World/roborg/minicobo_suction_short"
            artpath = f"{robot_prim_path}/minicobo_suction/world"
            robot_usd_file_path = f"{asimovjaka_extension_dir}/usd/minicobo-suction-2.usda"
            mopo_robot_name = "RS007N"
            rmp_param_dir = f"{asimovjaka_extension_dir}/JakaCtrl/motion_policy_configs/Jaka"
            rdf_path = f"{asimovjaka_extension_dir}/rdf/minicobo_robot_description.yaml"
            urdf_path = f"{asimovjaka_extension_dir}/urdf/minicobo_v14_onrobot_rg2.urdf"
            rmp_config_path = f"{asimovjaka_extension_dir}/rdf/minicobo_rmpflow_config.yaml"
            eeframe_name = "gripper_center"
            stiffness = 400
            damping = 40
            max_step_size = 0.00334

            grippername = "short suction"
            pp_controller = "jaka-ss"

            mfg = "Jaka"
            model = "Minicobo"
            desc = "Jaka Minicobo with a short suction gripper"


        case "minicobo-suction-high":
            robot_prim_path = "/World/roborg/minicobo_suction_short"
            artpath = f"{robot_prim_path}/minicobo_suction/world"
            robot_usd_file_path = f"{asimovjaka_extension_dir}/usd/minicobo-suction-2.usda"
            mopo_robot_name = "RS007N"
            rmp_param_dir = f"{asimovjaka_extension_dir}/JakaCtrl/motion_policy_configs/Jaka"
            rdf_path = f"{asimovjaka_extension_dir}/rdf/minicobo_robot_description.yaml"
            urdf_path = f"{asimovjaka_extension_dir}/urdf/minicobo_v14_onrobot_rg2.urdf"
            rmp_config_path = f"{asimovjaka_extension_dir}/rdf/minicobo_rmpflow_config.yaml"
            eeframe_name = "gripper_center"
            stiffness = 400
            damping = 40
            max_step_size = 0.00334

            grippername = "short suction"
            pp_controller = "jaka-ss"

            mfg = "Jaka"
            model = "Minicobo"
            desc = "Jaka Minicobo with a short suction gripper - mounted high"

        case "rs007n":
            robot_prim_path = "/World/roborg/khi_rs007n"
            artpath = robot_prim_path
            robot_usd_file_path = assets_root_dir + "/Isaac/Robots/Kawasaki/RS007N/rs007n_onrobot_rg2.usd"
            mopo_robot_name = "RS007N"
            rmp_param_dir = rmp_config_dir + "/Kawasaki"
            rdf_path = rmp_param_dir + "/rs007n/rmpflow/rs007n_robot_description.yaml"
            urdf_path = rmp_param_dir + "/rs007n/rs007n_onrobot_rg2.urdf"
            rmp_config_path = rmp_param_dir + "/rs007n/rmpflow/rs007n_rmpflow_config.yaml"
            eeframe_name = "gripper_center"
            max_step_size = 0.00334

            grippername = "rg2"
            pp_controller = "jaka-rg2"

            mfg = "Kawasaki"
            model = "RS007N"
            desc = "Kawasaki RS007N"

        case "franka":
            robot_prim_path = "/World/roborg/franka"
            artpath = robot_prim_path
            robot_usd_file_path = assets_root_dir + "/Isaac/Robots/Franka/franka.usd"
            mopo_robot_name = "Franka"

            rdf_path = rmp_config_dir + "/franka/rmpflow/robot_descriptor.yaml"
            urdf_path = rmp_config_dir + "/franka/lula_franka_gen.urdf"
            rmp_config_path = rmp_config_dir + "/franka/rmpflow/franka_rmpflow_common.yaml"
            eeframe_name = "right_gripper"
            max_step_size = 0.00334

            grippername = "franka gripper"
            pp_controller = "franka"

            mfg = "Franka"
            model = "Panda"
            desc = "Franka Panda"

        case "fancy_franka":
            robot_prim_path = "/World/roborg/Fancy_Franka"
            artpath = robot_prim_path
            robot_usd_file_path = assets_root_dir + "/Isaac/Robots/Franka/franka.usd"
            mopo_robot_name = "Franka"

            rdf_path = rmp_config_dir + "/franka/rmpflow/robot_descriptor.yaml"
            urdf_path = rmp_config_dir + "/franka/lula_franka_gen.urdf"
            rmp_config_path = rmp_config_dir + "/franka/rmpflow/franka_rmpflow_common.yaml"
            eeframe_name = "right_gripper"
            max_step_size = 0.00334

            grippername = "franka gripper"
            pp_controller = "franka"

            mfg = "Franka"
            model = "Panda"
            desc = "Franka Panda with some fancy initialization"

        case _:
            print("Bad robot type name", robot_name)

    if rdf_path != "None":
        if rdf_path=="" or urdf_path=="":
            msg = f"Robot {robot_name} rdf_path or urdf_path not specified"
            carb.log_warn(msg)
            print(msg)
            return

        if not os.path.isfile(rdf_path):
            msg = f"Robot {robot_name} rdf_path bad - file not found:{rdf_path}"
            carb.log_error(msg)
            print(msg)
            return

        if not os.path.isfile(urdf_path):
            msg = f"Robot {robot_name} urdf_path bad - file not found:{urdf_path}"
            carb.log_error(msg)
            print(msg)
            return

        if not os.path.isfile(rmp_config_path):
            msg = f"Robot {robot_name} rmp_config_path bad - file not found:{rmp_config_path}"
            carb.log_error(msg)
            print(msg)
            return

    rc = robcfg()
    rc.robot_name = robot_name
    if not skiplula:
        # we don't want to useup an id if we are skipping lula
        rc.robot_id = get_robot_id(robot_name)
    rc.robot_prim_path = robot_prim_path
    rc.eeframe_name = eeframe_name
    rc.max_step_size = max_step_size
    rc.stiffness = stiffness
    rc.damping = damping
    rc.mopo_robot_name = mopo_robot_name
    rc.robmatskin = "default"

    rc.mg_extension_dir = mg_extension_dir
    rc.rmp_config_dir = rmp_config_dir
    rc.jc_extension_dir = current_extension_dir
    rc.asv_extension_dir = asimovjaka_extension_dir

    rc.artpath = artpath

    rc.camera_root = camera_root

    rc.urdf_path = urdf_path
    rc.rdf_path = rdf_path
    rc.rmp_config_path = rmp_config_path
    rc.robot_usd_file_path = robot_usd_file_path

    rc.manufacturer = mfg
    rc.model = model
    rc.grippername = grippername
    rc.pp_controller = pp_controller
    rc.prefered_target = prefered_target
    rc.desc = desc

    rc.root_usdpath = robot_root_usdpath
    rc.current_robot_action = "None"

    if not skiplula:
        try:
            rc.robot_description = lula.load_robot(rdf_path, urdf_path)
            rc.lulaHelper = LulaInterfaceHelper(rc.robot_description)
        except Exception as e:
            msg = f"ScenarioBase - Robot {robot_name} lula.load_robot of rdf and urdf failed:{e}"
            carb.log_error(msg)
            print(msg)

    return rc

ids = {}
def get_robot_id(robot_name):
    global ids
    i = 0
    while True:
        robid = f"robot_{i}"
        if robid not in ids:
            ids[robid] = True
            return robid
        i += 1

def init_configs():
    global ids
    ids = {}