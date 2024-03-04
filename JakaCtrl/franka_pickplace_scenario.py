import numpy as np

from pxr import UsdPhysics, Usd, UsdGeom, Gf

import carb

from omni.isaac.core.utils.stage import add_reference_to_stage,  get_current_stage

from omni.isaac.core.articulations import Articulation
from omni.isaac.core.objects.cuboid import DynamicCuboid
from omni.isaac.core.objects import GroundPlane
from omni.isaac.manipulators.grippers import ParallelGripper
from .franka.controllers import PickPlaceController as franka_PickPlaceController
from .universal_robots.omni.isaac.universal_robots.controllers import PickPlaceController as ur10_PickPlaceController

from omni.isaac.core.world import World

from .senut import add_light_to_stage, get_robot_params
from .senut import ScenarioTemplate
from omni.isaac.manipulators.grippers.surface_gripper import SurfaceGripper
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.prims.rigid_prim import RigidPrim

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

    def __init__(self):
        pass

    def load_scenario(self, robot_name, ground_opt):

        self._robot_name = robot_name
        self._ground_opt = ground_opt

        add_light_to_stage()

       # print("Assets root path: ", get_assets_root_path())
        need_to_add_articulation = False
        self._robot_name = robot_name
        self._ground_opt = ground_opt
        (ok, robot_prim_path, artpath, path_to_robot_usd, mopo_robot_name) = get_robot_params(self._robot_name)
        if not ok:
            print(f"Unknown robot name {self._robot_name}")
            return

        if path_to_robot_usd is not None:
            if self._robot_name == "franka":
                stage = get_current_stage()
                pos = Gf.Vec3d([0, 0, 1.1])
                ang = 180
                roborg = UsdGeom.Xform.Define(stage, "/World/roborg")
                roborg.AddTranslateOp().Set(pos)
                roborg.AddRotateXOp().Set(ang)
                lula = UsdGeom.Xform.Define(stage, "/lula")
                lula.AddTranslateOp().Set(pos)
                lula.AddRotateXOp().Set(ang)
            add_reference_to_stage(path_to_robot_usd, robot_prim_path)

        if need_to_add_articulation:
            prim = get_current_stage().GetPrimAtPath(artpath)
            UsdPhysics.ArticulationRootAPI.Apply(prim)

        if self._robot_name == "fancy_franka":
            from omni.isaac.franka import Franka
            self._articulation= Franka(prim_path="/World/Fancy_Franka", name="fancy_franka")
        else:
            self._articulation = Articulation(artpath)


        # mode specific initialization
        self._cuboid = DynamicCuboid(
            "/Scenario/cuboid", position=np.array([0.3, 0.3, 0.15]), size=0.05, color=np.array([128, 0, 128])
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
        self._mopo_robot_name = mopo_robot_name
        print("load_scenario done - self._object", self._object)

    def get_gripper(self):
        if hasattr(self._articulation,"gripper"):
            gripper = self._articulation.gripper
            return gripper
        else:
            art = self._articulation
            art._policy_robot_name = self._mopo_robot_name
            if self._robot_name == "franka":
                # eepp = "/World/cobotta/onrobot_rg6_base_link"
                # jpn = ["finger_joint", "right_outer_knuckle_joint"]
                # jop = np.array([0, 0])
                # jcp = np.array([0.628, -0.628])
                # ad = np.array([-0.628, 0.628])
                eepp = "/World/roborg/franka/panda_rightfinger"
                jpn = ["panda_finger_joint1", "panda_finger_joint2"]
                jop = np.array([0.05, 0.05])
                jcp = np.array([0, 0])
                ad = np.array([0.05, 0.05])
                art._policy_robot_name = "Franka"
                pg = ParallelGripper(
                    end_effector_prim_path=eepp,
                    joint_prim_names=jpn,
                    joint_opened_positions=jop,
                    joint_closed_positions=jcp,
                    action_deltas=ad
                )
                pg.initialize(
                    physics_sim_view=None,
                    articulation_apply_action_func=art.apply_action,
                    get_joint_positions_func=art.get_joint_positions,
                    set_joint_positions_func=art.set_joint_positions,
                    dof_names=art.dof_names,
                )
                return pg
            elif self._robot_name == "rs007n":
                art = self._articulation
                eepp = "/World/roborg/khi_rs007n/gripper_center"
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
                eepp = "/World/roborg/ur10_suction_short/ee_link/gripper_base"
                jpn = ["left_inner_finger_joint", "right_inner_finger_joint"]
                jop = np.array([0.05, 0.05])
                jcp = np.array([0, 0])
                ad = np.array([0.05, 0.05])
                art._policy_robot_name = "UR10"
                self._end_effector_prim_path = eepp
                sg = SurfaceGripper(
                    end_effector_prim_path=self._end_effector_prim_path, translate=0.1611, direction="x"
                )
                self._end_effector = RigidPrim(prim_path=self._end_effector_prim_path, name= "ur10" + "_end_effector")
                self._end_effector.initialize(None)
                sg.initialize(
                    physics_sim_view=None,
                    articulation_num_dofs=len(art.dof_names)
                )
                return sg

                #
                #
                #
            elif self._robot_name == "jaka-minicobo":
                art = self._articulation
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
                gripper_usd = assets_root_path + "/Isaac/Robots/UR10/Props/short_gripper.usd"
                add_reference_to_stage(usd_path=gripper_usd, prim_path=self._end_effector_prim_path)
                sg = SurfaceGripper(
                    end_effector_prim_path=self._end_effector_prim_path, translate=0.1611, direction="z"
                )
                return sg
            else:
                return None

            # art.gripper = pg
            #define the manipulator
            # my_denso = self._world.scene.add(SingleManipulator(prim_path="/franka", name="robot",
            #                                     end_effector_prim_name="panda_rightfinger", gripper=pg))
            #set the default positions of the other gripper joints to be opened so
            #that its out of the way of the joints we want to control when gripping an object for instance.
            # joints_default_positions = np.zeros(12)
            # joints_default_positions[7] = 0.628
            # joints_default_positions[8] = 0.628
            # my_denso.set_joints_default_state(positions=joints_default_positions)
            return None


    def post_load_scenario(self):
        gripper = self.get_gripper()
        if gripper is not None:
            if self._robot_name in ["fancy_franka", "franka", "rs007n"]:
                self._controller = franka_PickPlaceController(
                    name="pick_place_controller",
                    gripper=gripper,
                    robot_articulation=self._articulation
                )
            elif self._robot_name in ["jaka-minicobo", "ur10-suction-short"]:
                self._controller = ur10_PickPlaceController(
                    name="pick_place_controller",
                    gripper=gripper,
                    robot_articulation=self._articulation
                )
            print("gripper.joint_opened_positions",gripper.joint_opened_positions)
            if self._show_collsion_bounds:
                self._rmpflow = self._controller._cspace_controller.rmp_flow
                # self._rmpflow.reset()
                self._rmpflow.visualize_collision_spheres()

            gripper.set_joint_positions(gripper.joint_opened_positions)
        self._world.add_physics_callback("sim_step", callback_fn=self.physics_step)

    def reset_scenario(self):
        gripper = self.get_gripper()
        if self._robot_name in ["fancy_franka", "franka", "rs007n"]:
            self._controller = franka_PickPlaceController(
                name="pick_place_controller",
                gripper=gripper,
                robot_articulation=self._articulation
            )
        elif self._robot_name in ["jaka-minicobo", "ur10-suction-short"]:
            self._controller = ur10_PickPlaceController(
                name="pick_place_controller",
                gripper=gripper,
                robot_articulation=self._articulation
            )
        if self._show_collsion_bounds:
            if self._rmpflow is not None:
                # self._rmpflow.reset()
                self._rmpflow.visualize_collision_spheres()
        gripper.set_joint_positions(gripper.joint_opened_positions)

    def physics_step(self, step_size):
        cube_position, _ = self._fancy_cube.get_world_pose()
        goal_position = np.array([-0.3, -0.3, 0.0515 / 2.0])
        current_joint_positions = self._articulation.get_joint_positions()
        actions = self._controller.forward(
            picking_position=cube_position,
            placing_position=goal_position,
            current_joint_positions=current_joint_positions,
        )
        self._articulation.apply_action(actions)
        # Only for the pick and place controller, indicating if the state
        # machine reached the final state.
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
