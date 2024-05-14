import numpy as np
import time

from pxr import UsdPhysics, Usd, UsdGeom, Gf, Sdf
from omni.isaac.core.world import World

import carb

from omni.isaac.core.utils.stage import add_reference_to_stage,  get_current_stage
import omni.timeline

from omni.isaac.core.articulations import Articulation
from omni.isaac.core.objects import cuboid, sphere, capsule
from omni.isaac.core.objects import GroundPlane
# from .franka.controllers import PickPlaceController as franka_PickPlaceController
# from robs.jaka.controllers.pick_place_controller import PickPlaceController as jaka_PickPlaceController
from omni.isaac.franka import Franka

from omni.isaac.motion_generation import ArticulationMotionPolicy
from omni.isaac.motion_generation import ArticulationKinematicsSolver


from .senut import add_sphere_light_to_stage
from .senut import adjust_joint_values, set_stiffness_for_joints, set_damping_for_joints
from .scenario_base import ScenarioBase

from .senut import apply_convex_decomposition_to_mesh_and_children, apply_material_to_prim_and_children
from .senut import apply_diable_gravity_to_rigid_bodies, adjust_articulationAPI_location_if_needed

from omni.asimov.manipulators.grippers.parallel_gripper import ParallelGripper
from omni.asimov.manipulators.grippers.surface_gripper import SurfaceGripper
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.rotations import euler_angles_to_quat

from .senut import calc_robot_circle_pose, interp, GetXformOps, GetXformOpsFromPath, deg_euler_to_quatd, deg_euler_to_quatf
from .motomod import MotoMan


# Copyright (c) 2022-2023, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

class PickAndPlaceNewScenario(ScenarioBase):
    _running_scenario = False
    _rmpflow = None
    _gripper_type = "none"
    _controller = None
    _rotate = False
    _rotate_speed = 1
    _show_rmp_target = False
    _show_rmp_target_opt = "invisible" # don't delete
    _show_collision_bounds = False
    _show_collision_bounds_opt = "invisible" # don't delete
    _show_endeffector_box = False


    def __init__(self, uibuilder=None):
        super().__init__()
        self._scenario_name = "pick-and-place-new"
        self._scenario_desc = ScenarioBase.get_scenario_desc(self._scenario_name)
        self._nrobots = 1
        self._uibuilder = uibuilder


    def load_scenario(self, robot_name, ground_opt):
        super().load_scenario(robot_name, ground_opt)

        self._robcfg = self.create_robot_config(robot_name, "/World/roborg", ground_opt)
        rcfg = self.get_robot_config()

        # self.get_robot_config(robot_name, ground_opt)


        self.nphysstep_calls = 0
        self.global_time = 0
        self.global_ang = 0

        self._robot_name = robot_name
        self._ground_opt = ground_opt

        add_sphere_light_to_stage()

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

        add_reference_to_stage(rcfg.robot_usd_file_path, rcfg.robot_prim_path)
        apply_convex_decomposition_to_mesh_and_children(stage, rcfg.robot_prim_path)
        apply_diable_gravity_to_rigid_bodies(stage, rcfg.robot_prim_path)
        adjust_articulationAPI_location_if_needed(stage, rcfg.robot_prim_path)


        if self._robot_name == "fancy_franka":
            rcfg._articulation= Franka(prim_path="/World/roborg/Fancy_Franka", name="fancy_franka")
        else:
            # quat = euler_angles_to_quat(np.array([0,0,0]))
            quat = euler_angles_to_quat(self._start_robot_rot)
            rcfg._articulation = Articulation(rcfg.artpath, position=self._start_robot_pos, orientation=quat)


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

        cuboid_path = "/Scenario/cuboid"
        prefered_target = rcfg.prefered_target
        if prefered_target == "cuboid":
            self._cuboid = cuboid.DynamicCuboid(
                cuboid_path,
                position=[0.3, 0.3, 0.15],
                scale=np.array([0.05, 0.05, 0.05]),
                color=np.array([0.0, 0, 0.5]),
            )
            self.cuboid_rmp_off = 0
        #  self._cuboid = DynamicCuboid(
        #     "/Scenario/cuboid", position=np.array([0.3, 0.3, 0.15]), size=0.05, color=np.array([128, 0, 128])
        # )

        elif prefered_target == "phone_slab":
            orient = euler_angles_to_quat(np.array([0, 0, 44*np.pi/180]))
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
        elif prefered_target == "moto50mp":
            a90 = np.pi/2
            rot = np.array([-a90, 0, 44*np.pi/180])
            mm = MotoMan(self._stage, self._matman)
            self.moto = mm.AddMoto50mp("moto2", rot=rot, pos=self._target_pos)
            self._cuboid = None
            self.cuboid_rmp_off = 0
        else:
            carb.log_error(f"Unknown prefered_target: {prefered_target}")

        # self._cuboid = DynamicCuboid(
        #     "/Scenario/cuboid", position=self._target_pos, size=0.05, color=np.array([128, 0, 128])
        # )

        # Add user-loaded objects to the World
        if rcfg._articulation is not None:
            world.scene.add(rcfg._articulation)
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
        self._mopo_robot_name = rcfg.mopo_robot_name

        print("load_scenario done")

    def set_robot_circle_pose(self, pos, rot):
        self._rob_tranop.Set(pos)
        self._rob_zrotop.Set(rot[2])
        self._rob_yrotop.Set(rot[1])
        self._rob_xrotop.Set(rot[0])
        self._start_robot_pos = pos
        self._start_robot_rot = rot

    def set_stiffness_for_all_joints(self, stiffness):
        rcfg = self.get_robot_config()
        active_joints = rcfg._rmpflow.get_active_joints()
        set_stiffness_for_joints(active_joints, stiffness)

    def set_damping_for_all_joints(self, damping):
        rcfg = self.get_robot_config()
        active_joints = rcfg._rmpflow.get_active_joints()
        set_damping_for_joints(active_joints, damping)

    def post_load_scenario(self):
        print("post_load_scenario - start")

        rcfg = self.get_robot_config()

        # self.lulaHelper = LulaInterfaceHelper(self._kinematics_solver._robot_description)


        self.register_articulation(rcfg._articulation) # this has to happen in post_load_scenario

        if self._robot_name in ["minicobo-rg2-high","minicobo-suction-high"]:
            rcfg.dof_zero_pos[2] = 0.9
            rcfg.dof_zero_pos[4] = 0.9
            rcfg._articulation.set_joints_default_state(rcfg.dof_zero_pos)
            rcfg._articulation.initialize()

        # rcfg._articulation.set_joint_positions(rcfg.dof_zero_pos)

        # these always need to exist
        self.grip_eeori = euler_angles_to_quat(np.array([0,0,0]))
        self.grip_eeoff = np.array([0,0,0])

        art = rcfg._articulation
        if not hasattr(art, "_policy_robot_name"):
            art._policy_robot_name = self._mopo_robot_name #ugly hack, should remove at some point


        if not hasattr(rcfg._articulation, "gripper"):
            rcfg.gripper = self.get_or_create_gripper(0)

        self._robot_id = rcfg.robot_id

        # add_camera_to_robot(self._robot_name, self._robot_id, rcfg.robot_prim_path)
        self.add_camera_to_robot(0)

        self.add_pp_controller(0)

        rcfg._rmpflow = rcfg._controller._cspace_controller.rmp_flow


        self.realize_rmptarg_vis(self._show_rmp_target_opt)
        if self._show_collision_bounds:
            rcfg._rmpflow.visualize_collision_spheres()
            self.realize_collider_vis_opt(self._show_collision_bounds_opt)
        if self._show_endeffector_box:
            rcfg._rmpflow.visualize_end_effector_position()


        self._timeline = omni.timeline.get_timeline_interface()
        # print("post_load_scenario - pre-forward_one_frame time: ", self._timeline.get_current_time())
        self._timeline.forward_one_frame()
        # print("post_load_scenario - post-forward_one_frame time: ", self._timeline.get_current_time())

        if self._robot_name in ["jaka-minicobo","jaka-minicobo-1","jaka-minicobo-2","minicobo-rg2-high","minicobo-suction","minicobo-suction-high"]:
            self.set_stiffness_for_all_joints(10000000.0 / 200) # 1e8 or 10 million seems too high
            self.set_damping_for_all_joints(100000.0 / 20) # 1e5 or 100 thousand seems too high

        rcfg._articulation_rmpflow = ArticulationMotionPolicy(rcfg._articulation, rcfg._rmpflow)
        rcfg._kinematics_solver = rcfg._rmpflow.get_kinematics_solver()

        rcfg._articulation_kinematics_solver = ArticulationKinematicsSolver(rcfg._articulation, rcfg._kinematics_solver, rcfg.eeframe_name)
        ee_pos, ee_rot_mat = rcfg._articulation_kinematics_solver.compute_end_effector_pose()

        print(f"post_load_scenario done")
        self._ee_pos = ee_pos
        self._ee_rot = ee_rot_mat

        self.activate_ee_collision( 0, False)

        print(f"post_load_scenario done")

    def reset_scenario(self):
        print(f"reset_scenario start")
        self.nphysstep_calls = 0
        self.global_time = 0
        self.global_ang = 0
        rcfg = self.get_robot_config()
        rcfg.gripper = self.get_or_create_gripper(0)
        print(f"reset_scenario after get_gripper")

        if rcfg._controller is not None:
            rcfg._controller.reset()

        if rcfg._rmpflow is not None:
            rcfg._rmpflow.reset()
            if self._show_collision_bounds:
                rcfg._rmpflow.visualize_collision_spheres()
                self.realize_collider_vis_opt(self._show_collision_bounds_opt)
            if self._show_endeffector_box:
                rcfg._rmpflow.visualize_end_effector_position()
            self.realize_rmptarg_vis(self._show_rmp_target_opt)


        if rcfg.gripper is not None:
            if rcfg._gripper_type == "parallel":
                rcfg.gripper.open()
            elif rcfg._gripper_type == "suction":
                if rcfg.gripper.is_closed():
                    rcfg.gripper.open()

        if self._robot_name in ["minicobo-rg2-high","minicobo-suction-high"]:
            rcfg.dof_zero_pos[2] = 0.9
            rcfg.dof_zero_pos[4] = 0.9
            rcfg._articulation.set_joint_positions(rcfg.dof_zero_pos)
        print(f"reset_scenario done - eeori: {self.grip_eeori}")



    nphysstep_calls = 0
    global_time = 0
    global_ang = 0
    def physics_step(self, step_size):
        npc = self.nphysstep_calls
        # print(f"physics_step {npc} start - time: {self.global_time:.4f} eeori: {self.grip_eeori} ")
        rcfg = self.get_robot_config()

        if npc==0:
            self.lasttime = self.global_time
            robot_base_translation,robot_base_orientation = rcfg._articulation.get_world_pose()
            print(f"physics step zero: robot_base_translation: {robot_base_translation}, robot_base_orientation: {robot_base_orientation}")
            rcfg._rmpflow.set_robot_base_pose(robot_base_translation,robot_base_orientation)

        phase = rcfg._controller._event

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
            rcfg._rmpflow.set_robot_base_pose(pos ,quat)

        if self._show_rmp_target:
            self.visualize_rmp_target()

        if rcfg.prefered_target in ["cuboid","phone-slab"]:
            cp, _ = self._cuboid.get_world_pose()
            cube_position = np.array([cp[0],cp[1],cp[2]+self.cuboid_rmp_off])
        else:
            cp, _ = self.moto.get_world_pose()
            rmpoff = 0.0
            cube_position = np.array([cp[0],cp[1],0.01+rmpoff])
        goal_position = self._goal_position
        current_joint_positions = rcfg._articulation.get_joint_positions()
        if rcfg._controller is not None:
            # eeoff = np.array([0,0,-0.03]) # -0.03 works for minicobo-suction-dual
            eeoff = np.array([0,0,-0.01])
            if self.rmpactive:
                if rcfg.prefered_target == "cuboid":
                    args = dict(
                        picking_position=cube_position,
                        placing_position=goal_position,
                        current_joint_positions=current_joint_positions
                    )
                else:
                    args = dict(
                        picking_position=cube_position,
                        placing_position=goal_position,
                        current_joint_positions=current_joint_positions,
                        end_effector_offset=eeoff,
                        end_effector_orientation=rcfg.grip_eeori
                    )
                actions = rcfg._controller.forward(**args)
                rcfg._articulation.apply_action(actions)

        ee_pos, ee_rot_mat = rcfg._articulation_kinematics_solver.compute_end_effector_pose()

        self._ee_pos = ee_pos
        self._ee_rot = ee_rot_mat
        # print(f"ee_pos:{ee_pos}")

        self.global_time += step_size
        self.nphysstep_calls += 1

        elap = self.global_time - self.lasttime
        if elap>0.5:
            print(f"physics_step {npc} rotate - time: {self.global_time:.4f} phase:{phase} eeori: {rcfg.grip_eeori} ")
            self.lasttime = self.global_time


        # Only for the pick and place controller, indicating if the state
        # machine reached the final state.
        # if rcfg._controller is not None:
        #     if rcfg._controller.is_done():
        #         self._world.pause()
        return

    def scenario_action(self, action_name: str, action_args):
        rcfg = self.get_robot_config()
        if rcfg._controller is not None:
            if action_name == "rotate":
                self._rotate = not self._rotate
                print(f"scenario_action - rotate changed to: {self._rotate}  param: {action_args}")
                return
            elif action_name == "show_rmp_target":
                self._show_rmp_target = not self._show_rmp_target
                print(f"scenario_action - _show_rmp_target changed to: {self._show_rmp_target}  param: {action_args}")
                return
        if action_name in self.base_scenario_actions:
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
        self.base_scenario_actions = super().get_scenario_actions()
        combo  = self.base_scenario_actions + ["rotate", "show_rmp_target"]
        return combo