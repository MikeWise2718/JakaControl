import numpy as np

from pxr import UsdPhysics, Usd, UsdGeom, Gf, Sdf

import carb

from omni.isaac.core.utils.stage import add_reference_to_stage,  get_current_stage
import omni.timeline

from omni.isaac.core.articulations import Articulation
from omni.isaac.core.objects.cuboid import DynamicCuboid, VisualCuboid
from omni.isaac.core.objects import cuboid, sphere, capsule
from omni.isaac.core.objects import GroundPlane
# from .franka.controllers import PickPlaceController as franka_PickPlaceController
from .franka.controllers import PickPlaceController as franka_PickPlaceController
from omni.asimov.jaka.controllers.pick_place_controller import PickPlaceController as jaka_PickPlaceController
from .universal_robots.omni.isaac.universal_robots.controllers import PickPlaceController as ur10_PickPlaceController
# from robs.jaka.controllers.pick_place_controller import PickPlaceController as jaka_PickPlaceController
from omni.isaac.franka import Franka

from omni.isaac.motion_generation import ArticulationMotionPolicy
from omni.isaac.motion_generation import ArticulationKinematicsSolver

from omni.isaac.core.world import World

from .senut import add_light_to_stage
from .senut import adjust_joint_values, set_stiffness_for_joints, set_damping_for_joints
from .scenario_base import ScenarioBase

from .senut import apply_convex_decomposition_to_mesh_and_children, apply_material_to_prim_and_children
from .senut import apply_diable_gravity_to_rigid_bodies, adjust_articulation

from omni.asimov.manipulators.grippers.parallel_gripper import ParallelGripper
from omni.asimov.manipulators.grippers.surface_gripper import SurfaceGripper
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.rotations import euler_angles_to_quat

from omni.isaac.core.prims.rigid_prim import RigidPrim
from .senut import calc_robot_circle_pose, interp, GetXformOps, GetXformOpsFromPath, deg_euler_to_quatd, deg_euler_to_quatf
from .senut import add_camera_to_robot

# Copyright (c) 2022-2023, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

class PickAndPlaceScenario(ScenarioBase):
    _running_scenario = False
    _rmpflow = None
    _show_collision_bounds = True
    _gripper_type = "none"
    _controller = None
    _rotate = False
    _rotate_speed = 1
    _show_rmp_target = False

    def __init__(self):
        super().__init__()
        self._scenario_name = "pick-and-place"
        self._scenario_desc = ScenarioBase.get_scenario_desc(self._scenario_name)
        self._nrobots = 1


    def load_scenario(self, robot_name, ground_opt):
        super().load_scenario(robot_name, ground_opt)

        self._robcfg = self.get_robcfg(robot_name, ground_opt)

        # self.get_robot_config(robot_name, ground_opt)


        self.nphysstep_calls = 0
        self.global_time = 0
        self.global_ang = 0

        self._robot_name = robot_name
        self._ground_opt = ground_opt

        add_light_to_stage()

       # print("Assets root path: ", get_assets_root_path())
        self._robot_name = robot_name
        self._ground_opt = ground_opt

        stage = get_current_stage()
        roborg = UsdGeom.Xform.Define(stage, "/World/roborg")
        self._rob = roborg
        self._rob_tranop = roborg.AddTranslateOp()
        self._rob_zrotop = roborg.AddRotateZOp()
        self._rob_yrotop = roborg.AddRotateYOp()
        self._rob_xrotop = roborg.AddRotateXOp()


        self._rob_ang = 0
        self._start_robot_pos = Gf.Vec3d([0, 0, 0])
        self._start_robot_rot = [0, 0, 0]
        if self._robot_name == "ur10-suction-short":
            self._start_robot_pos = Gf.Vec3d([0, 0, 0.4])
            self._start_robot_rot = [0, 0, 0]
        elif self._robot_name in ["minicobo-rg2-high","minicobo-suction-high","minicobo-dual-high"]:
            # self._start_robot_pos = Gf.Vec3d([-0.35, 0, 0.80])
            # self._start_robot_rot = [0, 130, 0]
            pos, rot = calc_robot_circle_pose(self._rob_ang)
            self._start_robot_pos = pos
            self._start_robot_rot = rot
        elif self._robot_name in ["fancy_franka"]:
            self._start_robot_pos = Gf.Vec3d([0, 0, 1.1])
            self._start_robot_rot = [180, 0, 0]

            print(f"load_scenario {self._robot_name} - start_robot_pos: {self._start_robot_pos} start_robot_rot: {self._start_robot_rot}")

        self.set_robot_circle_pose(self._start_robot_pos, self._start_robot_rot)

        add_reference_to_stage(self._robcfg.robot_usd_file_path, self._robcfg.robot_prim_path)
        apply_convex_decomposition_to_mesh_and_children(stage, self._robcfg.robot_prim_path)
        apply_diable_gravity_to_rigid_bodies(stage, self._robcfg.robot_prim_path)
        adjust_articulation(stage, self._robcfg.robot_prim_path)


        if self._robot_name == "fancy_franka":
            self._articulation= Franka(prim_path="/World/roborg/Fancy_Franka", name="fancy_franka")
        else:
            # quat = euler_angles_to_quat(np.array([0,0,0]))
            quat = euler_angles_to_quat(self._start_robot_rot)
            self._articulation = Articulation(self._robcfg.artpath, position=self._start_robot_pos, orientation=quat)


        # mode specific initialization
        if self._robot_name == "ur10-suction-short":
            # target_pos = np.array([1.16, 0.5, 0.15])
            self._target_pos = np.array([1.00, 0.5, 0.15])
            self._goal_position = np.array([+0.3, -0.3, 0.0515 / 2.0])
        elif self._robot_name in ["minicobo-rg2-high","minicobo-suction-high","minicobo-dual-high"]:
            self._target_pos = np.array([0.1, 0.1, 0.15])
            # self._target_pos = np.array([0.0, 0.25, 0.15])
            self._goal_position = np.array([0.0, -0.3, 0.025])
        else:
            self._target_pos = np.array([0.25, 0.25, 0.15])
            self._goal_position = np.array([+0.3, -0.3, 0.0515 / 2.0])

        world = World.instance()

        # world.scene.add(cuboid.DynamicCuboid(
        #     "/visualcube",
        #     # prim_path="/World/spawn_region",
        #     position=[0.7, 0, 0.45],
        #     scale=np.array([0.4, 1.0, 0.3]),
        #     color=np.array([1, 0, 1]),
        # ))

        orient = euler_angles_to_quat(np.array([0, 0, 44*np.pi/180]))
        cuboid_path = "/Scenario/cuboid"
        cuboid_thickness = 0.01
        self.cuboid_rmp_off = cuboid_thickness*interp(cuboid_thickness, 0.01, 0.02, 1.2, 0.9)
        self._cuboid = cuboid.DynamicCuboid(
            cuboid_path,
            position=self._target_pos,
            orientation=orient,
            scale=np.array([0.08, 0.16, cuboid_thickness]), # 0.7
            # scale=np.array([0.08, 0.16, 0.02]), # 0.7
            color=np.array([128, 0, 128]
            )
        )
        apply_material_to_prim_and_children(stage, self._matman, "Blue_Glass", cuboid_path )
        # self._cuboid = DynamicCuboid(
        #     "/Scenario/cuboid", position=self._target_pos, size=0.05, color=np.array([128, 0, 128])
        # )

        # Add user-loaded objects to the World
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

        # self._object = self._cuboid
        # self._fancy_cube = self._cuboid
        self._world = world
        self._mopo_robot_name = self._robcfg.mopo_robot_name

        print("load_scenario done")

    def set_robot_circle_pose(self, pos, rot):
        self._rob_tranop.Set(pos)
        self._rob_zrotop.Set(rot[2])
        self._rob_yrotop.Set(rot[1])
        self._rob_xrotop.Set(rot[0])
        self._start_robot_pos = pos
        self._start_robot_rot = rot

    def set_stiffness_for_all_joints(self, stiffness):
        joint_names = self._rmpflow.get_active_joints()
        set_stiffness_for_joints(joint_names, stiffness)

    def set_damping_for_all_joints(self, damping):
        joint_names = self._rmpflow.get_active_joints()
        set_damping_for_joints(joint_names, damping)

    def post_load_scenario(self):
        print("post_load_scenario - start")

        # self.lulaHelper = LulaInterfaceHelper(self._kinematics_solver._robot_description)

        self.register_articulation(self._articulation) # this has to happen in post_load_scenario

        if self._robot_name in ["minicobo-rg2-high","minicobo-suction-high"]:
            self._robcfg.joint_zero_pos[2] = 0.9
            self._robcfg.joint_zero_pos[4] = 0.9
            self._articulation.set_joints_default_state(self._robcfg.joint_zero_pos)
            self._articulation.initialize()

        # self._articulation.set_joint_positions(self._robcfg.joint_zero_pos)



        self._articulation.gripper = self.get_gripper()
        self._robot_id = self._robcfg.robot_id

        add_camera_to_robot(self._robot_name, self._robot_id, self._robcfg.robot_prim_path)

        self.add_controllers()

        if self._show_collision_bounds:
            self._rmpflow = self._controller._cspace_controller.rmp_flow
                # self._rmpflow.reset()
            self._rmpflow.visualize_collision_spheres()
            self._rmpflow.visualize_end_effector_position()


        self._timeline = omni.timeline.get_timeline_interface()
        # print("post_load_scenario - pre-forward_one_frame time: ", self._timeline.get_current_time())
        self._timeline.forward_one_frame()
        # print("post_load_scenario - post-forward_one_frame time: ", self._timeline.get_current_time())

        if self._robot_name in ["jaka-minicobo","jaka-minicobo-1","jaka-minicobo-2","minicobo-rg2-high","minicobo-suction","minicobo-suction-high"]:
            self.set_stiffness_for_all_joints(10000000.0 / 200) # 1e8 or 10 million seems too high
            self.set_damping_for_all_joints(100000.0 / 20) # 1e5 or 100 thousand seems too high


        self._articulation_rmpflow = ArticulationMotionPolicy(self._articulation,self._rmpflow)
        self._kinematics_solver = self._rmpflow.get_kinematics_solver()

        self._articulation_kinematics_solver = ArticulationKinematicsSolver(self._articulation,self._kinematics_solver, self._robcfg.eeframe_name)
        ee_pos, ee_rot_mat = self._articulation_kinematics_solver.compute_end_effector_pose()

        self._ee_pos = ee_pos
        self._ee_rot = ee_rot_mat

        print("post_load_scenario - done")

    def reset_scenario(self):
        self.nphysstep_calls = 0
        self.global_time = 0
        self.global_ang = 0
        gripper = self.get_gripper()

        if self._controller is not None:
            self._controller.reset()

        if self._show_collision_bounds:
            if self._rmpflow is not None:
                self._rmpflow.reset()
                self._rmpflow.visualize_collision_spheres()
                self._rmpflow.visualize_end_effector_position()

        if gripper is not None:
            if self._gripper_type == "parallel":
                gripper.open()
            elif self._gripper_type == "suction":
                if gripper.is_closed():
                    gripper.open()

        if self._robot_name in ["minicobo-rg2-high","minicobo-suction-high"]:
            self._robcfg.joint_zero_pos[2] = 0.9
            self._robcfg.joint_zero_pos[4] = 0.9
            self._articulation.set_joint_positions(self._robcfg.joint_zero_pos)



    def add_controllers(self):

        events_dt = [0.008, 0.005, 0.1,  0.1, 0.005, 0.005, 0.005, 0.1, 0.008, 0.08]

        gripper = self.get_gripper()
        if gripper is not None:
            if self._robot_name in ["fancy_franka", "franka", "rs007n"]:
                self._gripper_type = "parallel"
                self._controller = franka_PickPlaceController(
                    name="pick_place_controller",
                    gripper=gripper,
                    robot_articulation=self._articulation,
                    events_dt=events_dt
                )
            elif self._robot_name in ["ur10-suction-short"]:
                self._gripper_type = "suction"
                self._controller = ur10_PickPlaceController(
                    name="pick_place_controller",
                    gripper=gripper,
                    robot_articulation=self._articulation
                )
            elif self._robot_name in ["minicobo-suction","minicobo-suction-high","jaka-minicobo-1",
                                      "jaka-minicobo-1a","minicobo-dual-sucker","minicobo-suction-dual","minicobo-dual-high"]:
                self._gripper_type = "suction"
                rmpconfig = {
                    "end_effector_frame_name": self._robcfg.eeframe_name,
                    "maximum_substep_size": self._robcfg.max_step_size,
                    "ignore_robot_state_updates": False,
                    "urdf_path": self._robcfg.urdf_path,
                    "rmpflow_config_path": self._robcfg.rmp_config_path,
                    "robot_description_path": self._robcfg.rdf_path
                }
                events_dt = [0.008, 0.005, 0.1,  0.1, 0.005, 0.005, 0.005, 0.1, 0.008, 0.08]
                self._controller = jaka_PickPlaceController(
                    name="pick_place_controller",
                    gripper=gripper,
                    robot_articulation=self._articulation,
                    rmpconfig=rmpconfig,
                    events_dt=events_dt
                )
            elif self._robot_name in ["jaka-minicobo-0","jaka-minicobo-2","minicobo-rg2-high"]:
                self._gripper_type = "parallel"
                rmpconfig = {
                    "end_effector_frame_name": self._robcfg.eeframe_name,
                    "maximum_substep_size": self._robcfg.max_step_size,
                    "ignore_robot_state_updates": False,
                    "urdf_path": self._robcfg.urdf_path,
                    "rmpflow_config_path": self._robcfg.rmp_config_path,
                    "robot_description_path": self._robcfg.rdf_path
                }
                self._controller = jaka_PickPlaceController(
                    name="pick_place_controller",
                    gripper=gripper,
                    robot_articulation=self._articulation,
                    rmpconfig=rmpconfig,
                    events_dt=events_dt
                )


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
            self.grip_eeori = euler_angles_to_quat(np.array([0,0,0]))
            self.grip_eeoff = np.array([0,0,0])

            if self._robot_name in ["franka","fancy_franka"]:   # franka gripper
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

            elif self._robot_name in ["rs007n","jaka-minicobo-2","minicobo-rg2-high"]: # rg2 gripper / eepp, jpn, jop,jcp, ad
                art = self._articulation
                if self._robot_name == "rs007n":
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
                self._gripper_type = "parallel"
                return pg
            elif self._robot_name in ["ur10-suction-short","jaka-minicobo-1","jaka-minicobo-1a",
                                      "minicobo-suction-dual","minicobo-suction","minicobo-dual-sucker",
                                      "minicobo-dual-high","minicobo-suction-high"]:  # short suction gripper and dual sucker gripper
                art = self._articulation
                self._gripper_type = "suction"
                # eepp = "/World/roborg/ur10_suction_short/ee_link/gripper_base/xf"
                # UsdGeom.Xform.Define(get_current_stage(), eepp)
                # self._end_effector = RigidPrim(prim_path=eepp, name= "ur10" + "_end_effector")
                # self._end_effector.initialize(None)
                grip_direction = "x"
                grip_threshold = 0.02
                grip_translate = 0.1611

                if self._robot_name == "ur10-suction-short":
                    eepp = "/World/roborg/ur10_suction_short/ee_link"
                    self.grip_eeori = euler_angles_to_quat(np.array([0,np.pi/2,0]))
                elif self._robot_name == "minicobo-suction":
                    # eepp = "/World/roborg/minicobo_suction/short_gripper"
                    eepp = "/World/roborg/minicobo_suction_short/minicobo_suction/short_gripper"
                elif self._robot_name == "minicobo-suction-high":
                    eepp = "/World/roborg/minicobo_suction_short/minicobo_suction/short_gripper"
                elif self._robot_name in ["minicobo-suction-dual","minicobo-dual-high"]:
                    eepp = "/World/roborg/minicobo_suction_dual/minicobo_suction/dual_gripper"
                    # eepp = "/World/roborg/minicobo_suction_dual/minicobo_suction/dual_gripper/JAKA___MOTO_200mp_v4"

                    # self._end_effector = RigidPrim(prim_path=eepp, name= "minicobo_dual_gripper" + "_end_effector")
                    # self._end_effector.initialize(self.physics_sim_view)
                    grip_direction = "y"
                    grip_threshold = 0.1
                    grip_translate = 0.17
                    self.grip_eeori = euler_angles_to_quat(np.array([-np.pi/2,0,0]))
                elif self._robot_name in ["jaka-minicobo-1a","minicobo-dual-sucker"]:
                    eepp = "/World/roborg/minicobo_v1_4/tool0"
                    # eepp = "/World/roborg/minicobo_suction_dual/minicobo_suction/dual_gripper/JAKA___MOTO_200mp_v4"

                    # self._end_effector = RigidPrim(prim_path=eepp, name= "minicobo_dual_gripper" + "_end_effector")
                    # self._end_effector.initialize(self.physics_sim_view)
                    grip_direction = "y"
                    grip_threshold = 0.01
                    # grip_translate = -0.018 # 0.002 and -0.019 does not work, but 0.001 to -0.018 do work for jaka-minicobo-1a and minicobo-dual-sucker
                    grip_translate = 0.0
                    self.grip_eeori = euler_angles_to_quat(np.array([-np.pi/2,0,0]))

                elif self._robot_name == "jaka-minicobo-1":
                    eepp = "/World/roborg/minicobo_v1_4/Link6/jaka_camera_endpoint/JAKA___MOTO_200mp_v4/ZPR25CNK10_06_A10_v007"
                    self._end_effector = RigidPrim(prim_path=eepp, name= "jaka-minicobo-1" + "_end_effector")
                    self._end_effector.initialize(self.physics_sim_view)
                else:
                    print("Unknown robot name for suction gripper")
                # jpn = ["left_inner_finger_joint", "right_inner_finger_joint"]
                # jop = np.array([0.05, 0.05])
                # jcp = np.array([0, 0])
                # ad = np.array([0.05, 0.05])
                art._policy_robot_name = "UR10"
                self._end_effector_prim_path = eepp
                sg = SurfaceGripper(
                    end_effector_prim_path=self._end_effector_prim_path,
#                     translate=0.1611,
#                    translate=0.223, # minicobo-suction works between -0.001 and 0.222 - fails at 0.223 and -0.002
                    translate=grip_translate, # minicobo-suction works between -0.001 and 0.222 - fails at 0.223 and -0.002
                    direction=grip_direction,
                    grip_threshold=grip_threshold,  # between 0.01 and 0.5 work for minicobo-suction for the big cube
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

    nphysstep_calls = 0
    global_time = 0
    global_ang = 0
    def physics_step(self, step_size):
        if self.nphysstep_calls==0:
            robot_base_translation,robot_base_orientation = self._articulation.get_world_pose()
            print(f"physics step zero: robot_base_translation: {robot_base_translation}, robot_base_orientation: {robot_base_orientation}")
            self._rmpflow.set_robot_base_pose(robot_base_translation,robot_base_orientation)

        if self._robot_name in ["minicobo-rg2-high","minicobo-suction-high"] and self._rotate:
            angvel = 20
            phase = self._controller._event
            if phase in [1,2]:
                angvel = 10
            elif phase in [3,4]:
                angvel = 10
            elif phase in [5,6]:
                angvel = 10
            self.global_ang += self._rotate_speed*angvel*step_size
            pos, rot = calc_robot_circle_pose(self.global_ang)
            self.set_robot_circle_pose(pos, rot)
            rrot = np.array(rot)*np.pi/180
            quat = euler_angles_to_quat(rrot)
            self._rmpflow.set_robot_base_pose(pos ,quat)
            n = self.nphysstep_calls
            # print(f"physics_step {n} rotate - step_size: {step_size:.4f} ang: {self.global_ang} phase:{phase}")

        if self._show_rmp_target:
            self.visualize_rmp_target()

        cp, _ = self._cuboid.get_world_pose()
        cube_position = np.array([cp[0],cp[1],cp[2]+self.cuboid_rmp_off])
        goal_position = self._goal_position
        current_joint_positions = self._articulation.get_joint_positions()
        if self._controller is not None:
            # eeoff = np.array([0,0,-0.03]) # -0.03 works for minicobo-suction-dual
            eeoff = np.array([0,0,-0.01])
            if self.rmpactive:
                actions = self._controller.forward(
                    picking_position=cube_position,
                    placing_position=goal_position,
                    current_joint_positions=current_joint_positions,
                    end_effector_offset=eeoff,
                    end_effector_orientation=self.grip_eeori
                )
                self._articulation.apply_action(actions)

        ee_pos, ee_rot_mat = self._articulation_kinematics_solver.compute_end_effector_pose()

        self._ee_pos = ee_pos
        self._ee_rot = ee_rot_mat
        # print(f"ee_pos:{ee_pos}")


        self.nphysstep_calls += 1
        self.global_time += step_size
        # Only for the pick and place controller, indicating if the state
        # machine reached the final state.
        if self._controller is not None:
            if self._controller.is_done():
                self._world.pause()
        return

    def scenario_action(self, action_name: str, action_args):
        if self._controller is not None:
            if action_name == "rotate":
                self._rotate = not self._rotate
                print(f"scenario_action - rotate changed to: {self._rotate}  param: {param}")
                return
            elif action_name == "show_rmp_target":
                self._show_rmp_target = not self._show_rmp_target
                print(f"scenario_action - _show_rmp_target changed to: {self._show_rmp_target}  param: {param}")
                return
        if action_name in self.base_actions:
            rv = super().scenario_action(action_name, action_args)
            return rv
        return

    def setup_scenario(self):
        pass

    def teardown_scenario(self):
        pass

    def update_scenario(self, step: float):
        if not self._running_scenario:
            return
        self.physics_step(step)


    def get_scenario_actions(self):
        self.base_actions = super().get_scenario_actions()
        combo  = self.base_actions + ["rotate","show_rmp_target"]
        return combo