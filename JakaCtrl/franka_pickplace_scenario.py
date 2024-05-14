import numpy as np

from pxr import UsdPhysics, Usd, UsdGeom, Gf

import carb

from omni.isaac.core.utils.stage import add_reference_to_stage,  get_current_stage

from omni.isaac.core.articulations import Articulation
from omni.isaac.core.objects.cuboid import DynamicCuboid
from omni.isaac.core.objects import GroundPlane
from omni.asimov.manipulators.grippers import ParallelGripper
from .franka.controllers import PickPlaceController as franka_PickPlaceController
from .universal_robots.omni.isaac.universal_robots.controllers import PickPlaceController as ur10_PickPlaceController

from omni.isaac.core.world import World

from .senut import add_sphere_light_to_stage
from .scenario_base import ScenarioBase

from omni.asimov.manipulators.grippers.surface_gripper import SurfaceGripper
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.prims.rigid_prim import RigidPrim
from omni.isaac.core.utils.rotations import euler_angles_to_quat

# Copyright (c) 2022-2023, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#


class FrankaPickAndPlaceScenario(ScenarioBase):
    _running_scenario = False
    _rmpflow = None
    _rob_base_pos = Gf.Vec3d([0, 0, 0])
    _rob_base_xang = 0
    _gripper_type = "parallel"
    _show_rmp_target = False
    _show_rmp_target_opt = "invisible"

    def __init__(self, uibuilder=None):
        super().__init__()
        self._scenario_name = "franka-pick-and-place"
        self._scenario_description = ScenarioBase.get_scenario_desc(self._scenario_name)
        self._nrobots = 1
        self.uibuilder = uibuilder


    def load_scenario(self, robot_name, ground_opt):
        super().load_scenario(robot_name, ground_opt)
        self._robcfg = self.create_robot_config(robot_name, ground_opt)



        # self.get_robot_config(robot_name, ground_opt)

        self._robot_name = robot_name
        self._ground_opt = ground_opt

        add_sphere_light_to_stage()

       # print("Assets root path: ", get_assets_root_path())
        need_to_add_articulation = False
        self._robot_name = robot_name
        self._ground_opt = ground_opt

        if self._robot_name in ["fancy_franka"]:
            self._rob_base_pos = Gf.Vec3d([0, 0, 1.1])
            self._rob_base_xang = 180
        else:
            self._rob_base_pos = Gf.Vec3d([0, 0, 0])
            self._rob_base_xang = 0
        stage = get_current_stage()
        roborg = UsdGeom.Xform.Define(stage, "/World/roborg")
        roborg.AddTranslateOp().Set(self._rob_base_pos)
        roborg.AddRotateXOp().Set(self._rob_base_xang)
            # lula = UsdGeom.Xform.Define(stage, "/lula")
            # lula.AddTranslateOp().Set(pos)
            # lula.AddRotateXOp().Set(ang)


        if self._robcfg.robot_usd_file_path is not None:
            add_reference_to_stage(self._robcfg.robot_usd_file_path, self._robcfg.robot_prim_path)

        if need_to_add_articulation:
            prim = get_current_stage().GetPrimAtPath(self._robcfg.artpath)
            UsdPhysics.ArticulationRootAPI.Apply(prim)

        if self._robot_name in ["fancy_franka"]:
            from omni.isaac.franka import Franka
            self._articulation= Franka(prim_path="/World/roborg/Fancy_Franka", name="fancy_franka")
        else:
            self._articulation = Articulation(self._robcfg.artpath)


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
        print("load_scenario done - self._object", self._object)

    def get_franka_gripper(self):
        art = self._articulation
        art._policy_robot_name = self._robcfg.mopo_robot_name
        if hasattr(art,"gripper"):
            gripper = art.gripper
            gripper._policy_robot_name = self._robcfg.mopo_robot_name
            return gripper
        else:
            self.physics_sim_view = self._world.physics_sim_view

            if self._robot_name in ["franka","fancy_franka"]:
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
                    physics_sim_view=self.physics_sim_view,
                    articulation_apply_action_func=art.apply_action,
                    get_joint_positions_func=art.get_joint_positions,
                    set_joint_positions_func=art.set_joint_positions,
                    dof_names=art.dof_names,
                )
                self._gripper_type = "parallel"
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
                self._gripper_type = "parallel"
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
                self._gripper_type = "suction"
                return sg

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
                self._gripper_type = "suction"
                sg.initialize(
                    physics_sim_view=None,
                    articulation_num_dofs=len(art.dof_names)
                )
                return sg
            else:
                return None

            return None


    def post_load_scenario(self):

        self.register_articulation(self._articulation) # this has to happen in post_load_scenario

        gripper = self.get_franka_gripper()
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
            xqrot = euler_angles_to_quat(np.array([self._rob_base_xang*np.pi/180, 0, 0]))
            self._controller._cspace_controller._motion_policy.set_robot_base_pose(self._rob_base_pos, xqrot)
            self._rmpflow = self._controller._cspace_controller.rmp_flow

            if self._show_collision_bounds:
                self._rmpflow.visualize_collision_spheres()
            if self._show_endeffector_box:
                self._rmpflow.visualize_end_effector_position()
            if self._show_rmp_target:
                self.realize_rmptarg_vis(self._show_rmp_target_opt)


            gripper.set_joint_positions(gripper.joint_opened_positions)

            robot_base_translation,robot_base_orientation = self._articulation.get_world_pose()
            self._rmpflow.set_robot_base_pose(robot_base_translation,robot_base_orientation)


    def reset_scenario(self):
        gripper = self.get_franka_gripper()

        self._controller.reset()

        if self._show_collision_bounds:
            if self._rmpflow is not None:
                self._rmpflow.reset()
                if self._show_collision_bounds:
                    self._rmpflow.visualize_collision_spheres()
                if self._show_endeffector_box:
                    self._rmpflow.visualize_end_effector_position()
                if self._show_rmp_target:
                    self.realize_rmptarg_vis(self._show_rmp_target_opt)


        if gripper is not None:
            if self._gripper_type == "parallel":
                gripper.open()
            elif self._gripper_type == "suction":
                if gripper.is_closed():
                    gripper.open()

    nsteps = 0
    def physics_step(self, step_size):
        if self.nsteps==0:
            robot_base_translation,robot_base_orientation = self._articulation.get_world_pose()
            self._rmpflow.set_robot_base_pose(robot_base_translation,robot_base_orientation)
            # self._rmpflow.delete_collision_sphere_prims()
        # self._rmpflow._create_collision_sphere_prims(True)


        cube_position, _ = self._fancy_cube.get_world_pose()
        goal_position = np.array([-0.3, -0.3, 0.0515 / 2.0])
        current_joint_positions = self._articulation.get_joint_positions()
        actions = self._controller.forward(
            picking_position=cube_position,
            placing_position=goal_position,
            current_joint_positions=current_joint_positions,
        )
        self._articulation.apply_action(actions)
        self.nsteps += 1

        # Only for the pick and place controller, indicating if the state
        # machine reached the final state.
        # if self._controller.is_done():
        #     self._world.pause()
        return

    def setup_scenario(self):
        pass

    def teardown_scenario(self):
        pass

    def update_scenario(self, step: float):
        if not self._running_scenario:
            return
        self.physics_step(step)



    def can_handle_robot(self, robot_name):
        if robot_name in ["fancy_franka", "franka"]:
            return True
        return False
