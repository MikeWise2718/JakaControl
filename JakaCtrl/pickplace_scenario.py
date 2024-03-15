import numpy as np

from pxr import UsdPhysics, Usd, UsdGeom, Gf, Sdf

import carb

from omni.isaac.core.utils.stage import add_reference_to_stage,  get_current_stage
import omni.timeline

from omni.isaac.core.articulations import Articulation
from omni.isaac.core.objects.cuboid import DynamicCuboid
from omni.isaac.core.objects import GroundPlane
from omni.isaac.manipulators.grippers import ParallelGripper
from .franka.controllers import PickPlaceController as franka_PickPlaceController
from .universal_robots.omni.isaac.universal_robots.controllers import PickPlaceController as ur10_PickPlaceController
from robs.jaka.controllers.pick_place_controller import PickPlaceController as jaka_PickPlaceController


from omni.isaac.core.world import World

from .senut import add_light_to_stage, get_robot_params
from .senut import ScenarioTemplate
from omni.isaac.manipulators.grippers.surface_gripper import SurfaceGripper
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.prims.rigid_prim import RigidPrim
from omni.isaac.motion_generation.lula.interface_helper import LulaInterfaceHelper
from omni.isaac.core.utils.rotations import euler_angles_to_quat

from omni.isaac.core.utils.prims import delete_prim, is_prim_path_valid
from omni.isaac.core.utils.string import find_unique_string_name
from omni.isaac.core import objects

# Copyright (c) 2022-2023, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

class PickAndPlaceScenario(ScenarioTemplate):
    _running_scenario = False
    _rmpflow = None
    _show_collision_bounds = True
    _gripper_type = "none"
    _controller = None

    def __init__(self):
        pass

    # def set_robot_pose(self, robot_name, pos, xang, yang, zang):
    #     stage = get_current_stage()
    #     roborg = UsdGeom.Xform.Define(stage, "/World/roborg")
    #     gfpos = Gf.Vec3d(pos)
    #     roborg.AddTranslateOp().Set(gfpos)
    #     roborg.AddRotateXOp().Set(xang)
    #     roborg.AddRotateYOp().Set(yang)
    #     roborg.AddRotateZOp().Set(zang)
    #     lulaprim = UsdGeom.Xform.Define(stage, "/lula")
    #     lulaprim.AddTranslateOp().Set(gfpos)
    #     lulaprim.AddRotateXOp().Set(xang)
    #     lulaprim.AddRotateYOp().Set(yang)
    #     lulaprim.AddRotateZOp().Set(zang)

    def load_scenario(self, robot_name, ground_opt):
        self.get_robot_config(robot_name, ground_opt)

        self._robot_name = robot_name
        self._ground_opt = ground_opt

        add_light_to_stage()

       # print("Assets root path: ", get_assets_root_path())
        need_to_add_articulation = False
        self._robot_name = robot_name
        self._ground_opt = ground_opt



        self._start_robot_pos = Gf.Vec3d([0, 0, 0])
        self._start_robot_rot = [0, 0, 0]
        if self._robot_name == "ur10-suction-short":
            self._start_robot_pos = Gf.Vec3d([0, 0, 0.4])
            self._start_robot_rot = [0, 0, 0]
        elif self._robot_name == "jaka-minicobo-3":
            self._start_robot_pos = Gf.Vec3d([0, 0, 0.85])
            self._start_robot_rot = [0, 130, 0]


        quat = euler_angles_to_quat(np.array([0,0,0]))

        stage = get_current_stage()
        roborg = UsdGeom.Xform.Define(stage, "/World/roborg")
        roborg.AddTranslateOp().Set(self._start_robot_pos)
        roborg.AddRotateXOp().Set(self._start_robot_rot[0])
        roborg.AddRotateYOp().Set(self._start_robot_rot[1])
        roborg.AddRotateZOp().Set(self._start_robot_rot[2])

        add_reference_to_stage(self._cfg_path_to_robot_usd, self._cfg_robot_prim_path)

        if need_to_add_articulation:
            prim = get_current_stage().GetPrimAtPath(self._cfg_artpath)
            UsdPhysics.ArticulationRootAPI.Apply(prim)

        if self._robot_name == "fancy_franka":
            from omni.isaac.franka import Franka
            self._articulation= Franka(prim_path="/World/Fancy_Franka", name="fancy_franka")
        else:
            self._articulation = Articulation(self._cfg_artpath, position=self._start_robot_pos, orientation=quat)
            # if self._robot_name == "jaka-minicobo-3":
            #     self._cfg_njoints = self._articulation.num_dof
            #     self._cfg_joint_zero_pos = np.zeros(self._cfg_njoints)
            #     self._cfg_joint_zero_pos[2] = 0.9
            #     self._cfg_joint_zero_pos[4] = 0.9
            #     self._articulation.set_joints_default_state(self._cfg_joint_zero_pos)


        # mode specific initialization
        if self._robot_name == "ur10-suction-short":
            # target_pos = np.array([1.16, 0.5, 0.15])
            target_pos = np.array([1.00, 0.5, 0.15])
        else:
            target_pos = np.array([0.25, 0.25, 0.15])
        self._cuboid = DynamicCuboid(
            "/Scenario/cuboid", position=target_pos, size=0.05, color=np.array([128, 0, 128])
        )

        # Add user-loaded objects to the World
        world = World.instance()
        if self._articulation is not None:
            world.scene.add(self._articulation)
        if self._cuboid is not None:
            world.scene.add(self._cuboid)

        if self._ground_opt == "default":
            world.scene.add_default_ground_plane()

        elif self._ground_opt == "groundplane":
            ground = GroundPlane(prim_path="/World/groundPlane", size=10, color=np.array([0.5, 0.5, 0.5]))
            world.scene.add(ground)

        elif self._ground_opt == "groundplane-blue":
            ground = GroundPlane(prim_path="/World/groundPlane", size=10, color=np.array([0.0, 0.0, 0.5]))
            world.scene.add(ground)

        self._object = self._cuboid
        self._fancy_cube = self._cuboid
        self._world = world
        self._mopo_robot_name = self._cfg_mopo_robot_name



        print("load_scenario done")



    def post_load_scenario(self):
        print("post_load_scenario - start")

        # self.lulaHelper = LulaInterfaceHelper(self._kinematics_solver._robot_description)

        self.register_articulation(self._articulation) # this has to happen in post_load_scenario

        if self._robot_name == "jaka-minicobo-3":
            self._cfg_joint_zero_pos[2] = 0.9
            self._cfg_joint_zero_pos[4] = 0.9
            self._articulation.set_joints_default_state(self._cfg_joint_zero_pos)
            self._articulation.initialize()

        # self._articulation.set_joint_positions(self._cfg_joint_zero_pos)

        gripper = self.get_gripper()
        if gripper is not None:
            if self._robot_name in ["fancy_franka", "franka", "rs007n"]:
                self._gripper_type = "parallel"
                self._controller = franka_PickPlaceController(
                    name="pick_place_controller",
                    gripper=gripper,
                    robot_articulation=self._articulation
                )
            elif self._robot_name in ["ur10-suction-short"]:
                self._gripper_type = "suction"
                self._controller = ur10_PickPlaceController(
                    name="pick_place_controller",
                    gripper=gripper,
                    robot_articulation=self._articulation
                )
            elif self._robot_name in ["jaka-minicobo-0","jaka-minicobo-1","jaka-minicobo-2","jaka-minicobo-3"]:
                self._gripper_type = "parallel"
                rmpconfig = {
                    "end_effector_frame_name": self._cfg_eeframe_name,
                    "maximum_substep_size": self._cfg_max_step_size,
                    "ignore_robot_state_updates": False,
                    "urdf_path": self._cfg_urdf_path,
                    "rmpflow_config_path": self._cfg_rmp_config_path,
                    "robot_description_path": self._cfg_rdf_path
                }
                self._controller = jaka_PickPlaceController(
                    name="pick_place_controller",
                    gripper=gripper,
                    robot_articulation=self._articulation,
                    rmpconfig=rmpconfig
                )
            if self._show_collision_bounds:
                self._rmpflow = self._controller._cspace_controller.rmp_flow
                    # self._rmpflow.reset()
                self._rmpflow.visualize_collision_spheres()

        self._timeline = omni.timeline.get_timeline_interface()
        # self._timeline.set_auto_update(False)
        self._timeline.forward_one_frame()
        # self._timeline.set_auto_update(True)

        print("post_load_scenario - done")

    def reset_scenario(self):
        self.nsteps = 0
        gripper = self.get_gripper()

        if self._controller is not None:
            self._controller.reset()

        if self._show_collision_bounds:
            if self._rmpflow is not None:
                self._rmpflow.reset()
                self._rmpflow.visualize_collision_spheres()

        if gripper is not None:
            if self._gripper_type == "parallel":
                gripper.open()
            elif self._gripper_type == "suction":
                if gripper.is_closed():
                    gripper.open()

        if self._robot_name == "jaka-minicobo-3":
            self._cfg_joint_zero_pos[2] = 0.9
            self._cfg_joint_zero_pos[4] = 0.9
            self._articulation.set_joint_positions(self._cfg_joint_zero_pos)

    def get_gripper(self):
        art = self._articulation
        if not hasattr(art, "_policy_robot_name"):
            art._policy_robot_name = self._mopo_robot_name
        if hasattr(self._articulation,"gripper"):
            gripper = art.gripper
            return gripper
        else:
            art = self._articulation
            self._gripper_type = "parallel"
            art._policy_robot_name = self._mopo_robot_name
            self.physics_sim_view = self._world.physics_sim_view

            if self._robot_name in ["franka","fancy_franka"]:
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
                    physics_sim_view=self.physics_sim_view,
                    articulation_apply_action_func=art.apply_action,
                    get_joint_positions_func=art.get_joint_positions,
                    set_joint_positions_func=art.set_joint_positions,
                    dof_names=art.dof_names,
                )
                return pg
            elif self._robot_name in ["rs007n","jaka-minicobo-2","jaka-minicobo-3"]:
                art = self._articulation
                self._gripper_type = "parallel"
                if self._robot_name == "rs007n":
                    eepp = "/World/roborg/khi_rs007n/gripper_center"
                else:
                    eepp = "/World/roborg/minicobo_parallel_onrobot_rg2/minicobo_onrobot_rg2/gripper_center"
                jpn = ["left_inner_finger_joint", "right_inner_finger_joint"]
                jop = np.array([0.15, 0.15])
                jcp = np.array([0, 0])
                ad = np.array([0.15, 0.15])
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
                return pg
            elif self._robot_name == "ur10-suction-short":
                art = self._articulation
                self._gripper_type = "suction"
                # eepp = "/World/roborg/ur10_suction_short/ee_link/gripper_base/xf"
                # UsdGeom.Xform.Define(get_current_stage(), eepp)
                # self._end_effector = RigidPrim(prim_path=eepp, name= "ur10" + "_end_effector")
                # self._end_effector.initialize(None)
                eepp = "/World/roborg/ur10_suction_short/ee_link"
                jpn = ["left_inner_finger_joint", "right_inner_finger_joint"]
                jop = np.array([0.05, 0.05])
                jcp = np.array([0, 0])
                ad = np.array([0.05, 0.05])
                art._policy_robot_name = "UR10"
                self._end_effector_prim_path = eepp
                sg = SurfaceGripper(
                    end_effector_prim_path=self._end_effector_prim_path, translate=0.1611, direction="x"
                )
                # self._end_effector = RigidPrim(prim_path=eeppgb, name= "ur10" + "_end_effector")
                # self._end_effector.initialize(None)
                sg.initialize(
                    physics_sim_view=self.physics_sim_view,
                    articulation_num_dofs=len(art.dof_names)
                )
                return sg

            elif self._robot_name == "jaka-minicobo":
                art = self._articulation
                self._gripper_type = "suction"
                eepp = "/World/roborg/minicobo_v1_4/dummy_tcp"
                jpn = ["left_inner_finger_joint", "right_inner_finger_joint"]
                jop = np.array([0.05, 0.05])
                jcp = np.array([0, 0])
                ad = np.array([0.05, 0.05])
                art._policy_robot_name = "Franka"
                self._end_effector_prim_path = eepp
                assets_root_path = get_assets_root_path()
                if assets_root_path is None:
                    carb.log_error("Could not find Isaac Sim assets folder")
                    return
                # gripper_usd = assets_root_path + "/Isaac/Robots/UR10/Props/short_gripper.usd"
                # add_reference_to_stage(usd_path=gripper_usd, prim_path=self._end_effector_prim_path)
                # self._end_effector = RigidPrim(prim_path=eeppgb, name= "ur10" + "_end_effector")
                sg = SurfaceGripper(
                    end_effector_prim_path=self._end_effector_prim_path, translate=0.1611, direction="z"
                )
                sg.initialize(
                    physics_sim_view=None,
                    articulation_num_dofs=len(art.dof_names)
                )
                return sg
            else:
                return None

    def mw_create_collision_sphere_prims(self, is_visible):
        print("mwdb - _create_collision_sphere_prims")
        self._robot_description = self._rmpflow._robot_description
        self._policy = self._rmpflow._policy
        self._robot_joint_positions = self._rmpflow._robot_joint_positions
        self._meters_per_unit = self._rmpflow._meters_per_unit
        if self._robot_joint_positions is None:
            joint_positions = self._robot_description.default_c_space_configuration()
        else:
            joint_positions = self._robot_joint_positions.astype(np.float64)

        lih = self._rmpflow

        sphere_poses = self._policy.collision_sphere_positions(joint_positions)
        sphere_radii = self._policy.collision_sphere_radii()
        nsph = len(sphere_poses)
        print(f"mwdb - mw_create_collision_sphere_prims - found {nsph} configured spheres")
        for i, (sphere_pose, sphere_rad) in enumerate(zip(sphere_poses, sphere_radii)):
            prim_path = find_unique_string_name("/lula/collision_sphere" + str(i), lambda x: not is_prim_path_valid(x))
            self._rmpflow._collision_spheres.append(
                objects.sphere.VisualSphere(prim_path, radius=sphere_rad / self._meters_per_unit)
            )
        j = 0
        for sphere, sphere_pose in zip(self._rmpflow._collision_spheres, sphere_poses):
            new_pose = lih._robot_rot @ sphere_pose + lih._robot_pos
            if j<4:
                print(f"mwdb - mw_create_collision_sphere_prims - {j} sphere_pose: {sphere_pose} new_pose: {new_pose}")
            sphere.set_world_pose(new_pose / self._meters_per_unit)
            sphere.set_visibility(is_visible)
            j += 1


    nsteps = 0
    def physics_step(self, step_size):
        if self.nsteps==0:
            robot_base_translation,robot_base_orientation = self._articulation.get_world_pose()
            print(f"robot_base_translation: {robot_base_translation}, robot_base_orientation: {robot_base_orientation}")
            self._rmpflow.set_robot_base_pose(robot_base_translation,robot_base_orientation)

        # self._rmpflow.delete_collision_sphere_prims()
        # self.mw_create_collision_sphere_prims(True)
            # self._rmpflow._create_collision_sphere_prims(True)
            # self._rmpflow.visualize_collision_spheres()

        cube_position, _ = self._fancy_cube.get_world_pose()
        goal_position = np.array([+0.3, -0.3, 0.0515 / 2.0])
        current_joint_positions = self._articulation.get_joint_positions()
        if self._controller is not None:
            actions = self._controller.forward(
                picking_position=cube_position,
                placing_position=goal_position,
                current_joint_positions=current_joint_positions,
            )
            if self._articulation is not None:
                    self._articulation.apply_action(actions)
        self.nsteps += 1
        # Only for the pick and place controller, indicating if the state
        # machine reached the final state.
        if self._controller is not None:
            if self._controller.is_done():
                self._world.pause()
        return

    def setup_scenario(self):
        pass

    def teardown_scenario(self):
        pass

    def update_scenario(self, step: float):
        if not self._running_scenario:
            return
        self.physics_step(step)
