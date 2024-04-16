import numpy as np
from pxr import Usd, UsdGeom, Gf

from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.objects.cuboid import FixedCuboid
from omni.isaac.core.prims import XFormPrim

from omni.isaac.core.utils.stage import get_current_stage
from omni.isaac.core.world import World

from omni.isaac.core.utils.extensions import get_extension_path_from_name
from omni.isaac.core.utils.rotations import euler_angles_to_quat
from omni.isaac.core.utils.viewports import set_camera_view

from .senut import apply_material_to_prim_and_children, GetXformOps, GetXformOpsFromPath

from .scenario_base import ScenarioBase

# Copyright (c) 2022-2023, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

class CageRmpflowScenario(ScenarioBase):

    _running_scenario = False
    _colorScheme = "transparent"
    rotate_target0 = False
    rotate_target1 = False
    target_rot_speed = 2*np.pi/10 # 10 seconds for a full rotation



    def __init__(self):
        super().__init__()
        self._scenario_name = "cage-rmpflow"
        self._scenario_description = ScenarioBase.get_scenario_desc(self._scenario_name)
        self._nrobots = 2
        pass

    def load_scenario(self, robot_name, ground_opt, light_opt="dome_light"):
        super().load_scenario(robot_name, ground_opt)

        self.create_robot_config(robot_name,"/World/roborg0")
        self.create_robot_config(robot_name,"/World/roborg1")

        self.add_light(light_opt)
        self.add_ground(ground_opt)

        # Robots
        (pos0, rot0) = ([0.14, 0, 0.77], [0, 150, 0])
        self.load_robot_into_scene(0, pos0, rot0)

        (pos1, rot1) = ([-0.08, 0, 0.77], [0, -150, 0])
        self.load_robot_into_scene(1, pos1, rot1)

        self.add_cameras_to_robots()

        # tagets
        quat = euler_angles_to_quat([-np.pi/2,0,0])
        t0path = "/World/target0"
        self._target0 = XFormPrim(t0path, scale=[.04,.04,.04], position=[0.15, 0.00, 0.02], orientation=quat)
        (self.targ0top,_,_,_) = GetXformOpsFromPath(t0path)
        add_reference_to_stage(get_assets_root_path() + "/Isaac/Props/UIElements/frame_prim.usd", t0path)

        quat = euler_angles_to_quat([-np.pi/2,0,0])
        t1path = "/World/target1"
        self._target1 = XFormPrim(t1path, scale=[.04,.04,.04], position=[-0.15, 0.00, 0.02], orientation=quat)
        (self.targ1top,_,_,_) = GetXformOpsFromPath(t1path)
        add_reference_to_stage(get_assets_root_path() + "/Isaac/Props/UIElements/frame_prim.usd", t1path)


        # obstacles
        self._obstacle = FixedCuboid("/World/obstacle",size=.05,position=np.array([0.4, 0.0, 1.65]),color=np.array([0.,0.,1.]))

        # cage
        jakacontrol_extension_path = get_extension_path_from_name("JakaControl")
        path_to_cage_usd = f"{jakacontrol_extension_path}/usd/cage_v1.usd"
        add_reference_to_stage(path_to_cage_usd, "/World/cage_v1")

        cagepath = "/World/cage_v1"
        self._cage = XFormPrim(cagepath, scale=[1,1,1], position=[0,0,0])
        if self._colorScheme == "default":
            self._cage.set_color([0.5, 0.5, 0.5, 1.0])
        elif self._colorScheme == "transparent":
            apply_material_to_prim_and_children(self._stage, self._matman, "Steel_Blued", cagepath)

    def setup_scenario(self):
        self.register_robot_articulations()
        self.adjust_stiffness_and_damping_for_robots()
        self.teleport_robots_to_zeropos()

        self.make_robot_mpflows([self._obstacle])

        set_camera_view(eye=[0.0, 2.5, 1.0], target=[0,0,0], camera_prim_path="/OmniverseKit_Persp")

        self._running_scenario = True

    def reset_scenario(self):
        self.reset_robot_rmpflows()

    gang = 0
    def rotate_target(self, target, top, cen, radius, step_size):
        # pos, ori = target.get_world_pose()
        cen = np.array(cen)
        (xp,yp,zp) = cen
        # newpos = np.array([radius*np.cos(ang), radius*np.sin(ang), zp])
        self.gang += self.target_rot_speed*step_size
        newpos = Gf.Vec3d([xp+radius*np.cos(self.gang), yp+radius*np.sin(self.gang), zp])
        top.Set(newpos)

    def physics_step(self, step_size):
        self.global_time += step_size

        if self.rmpactive:
            self.rmpflow_update_world_for_all()

        if self.rotate_target0:
            self.rotate_target(self._target0, self.targ0top, [+0.3, 0.00, 0.02], 0.15, step_size)
        if self.rotate_target1:
            self.rotate_target(self._target1, self.targ1top,  [-0.3, 0.00, 0.02], 0.15, step_size)

        target0_position, target0_orientation = self._target0.get_world_pose()
        target1_position, target1_orientation = self._target1.get_world_pose()

        if self.rmpactive:
            self.set_end_effector_target_for_robot(0, target0_position, target0_orientation)
            self.set_end_effector_target_for_robot(1, target1_position, target1_orientation)
            self.forward_rmpflow_step_for_robots(step_size)

    def update_scenario(self, step: float):
        if not self._running_scenario:
            return
        self.physics_step(step)

    def scenario_action(self, action_name, action_args):
        if action_name in self.base_actions:
            rv = super().scenario_action(action_name, action_args)
            return rv
        match action_name:
            case "RotateTarget0":
                self.rotate_target0 = not self.rotate_target0
            case "RotateTarget1":
                self.rotate_target1 = not self.rotate_target1
            case "FasterTargetSpeed":
                self.target_rot_speed *= 2
            case "SlowerTargetSpeed":
                self.target_rot_speed /= 2
            case "ReverseTargetSpeed":
                self.target_rot_speed *= -1
            case _:
                print(f"Action {action_name} not implemented")
                return False

    def scenario_action(self, action_name, action_args):
        if action_name in self.base_actions:
            rv = super().scenario_action(action_name, action_args)
            return rv
        match action_name:
            case "RotateTarget0":
                self.rotate_target0 = not self.rotate_target0
            case "RotateTarget1":
                self.rotate_target1 = not self.rotate_target1
            case "FasterTargetSpeed":
                self.target_rot_speed *= 2
            case "SlowerTargetSpeed":
                self.target_rot_speed /= 2
            case "ReverseTargetSpeed":
                self.target_rot_speed *= -1
            case _:
                print(f"Action {action_name} not implemented")
                return False

    def get_action_button_text(self, action_name, action_args=None):
        if action_name in self.base_actions:
            rv = super().get_action_button_text(action_name, action_args)
            return rv
        match action_name:
            case "RotateTarget0":
                rv = "Rotate Target 0"
            case "RotateTarget1":
                rv = "Rotate Target 1"
            case "FasterTargetSpeed":
                rv = "Double Target Speed"
            case "SlowerTargetSpeed":
                rv = "Half Target Speed"
            case "ReverseTargetSpeed":
                rv = "Reverse Target Direction"
            case _:
                print(f"Action {action_name} not implemented")
                return False
        return rv

    def get_scenario_actions(self):
        self.base_actions = super().get_scenario_actions()
        combo  = self.base_actions + ["RotateTarget0", "RotateTarget1",
                                      "FasterTargetSpeed","SlowerTargetSpeed","ReverseTargetSpeed"]
        return combo
