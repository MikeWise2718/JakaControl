import numpy as np

from pxr import Sdf, UsdLux, UsdPhysics

from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.stage import add_reference_to_stage, create_new_stage, get_current_stage

from omni.isaac.core.articulations import Articulation
from omni.isaac.core.objects.cuboid import FixedCuboid
from omni.isaac.core.objects import GroundPlane
import omni.isaac.franka.controllers as franka_controllers

from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.core.world import World


# Copyright (c) 2022-2023, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

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

from pxr import Sdf, UsdLux
from omni.isaac.core.prims import XFormPrim
from omni.isaac.core.utils.stage import create_new_stage, get_current_stage


class ScenarioTemplate:
    def __init__(self):
        pass

    def setup_scenario(self):
        pass

    def teardown_scenario(self):
        pass

    def update_scenario(self):
        pass



class PickAndPlaceScenario(ScenarioTemplate):
    _running_scenario = False
    def __init__(self):
        pass

    def load_scenario(self, robot_name, ground_opt):

        self._robot_name = robot_name
        self._ground_opt = ground_opt

        add_light_to_stage()

        world = World.instance()
        if self._ground_opt == "default":
            world.scene.add_default_ground_plane()

        elif self._ground_opt == "groundplane":
            ground = GroundPlane(prim_path="/World/groundPlane", size=10, color=np.array([0.5, 0.5, 0.5]))
            world.scene.add(ground)

        elif self._ground_opt == "groundplane-blue":
            ground = GroundPlane(prim_path="/World/groundPlane", size=10, color=np.array([0.0, 0.0, 0.5]))
            world.scene.add(ground)

        pass

    def setup_scenario(self):
        pass

    def teardown_scenario(self):
        pass

    def update_scenario(self, step: float):
        if not self._running_scenario:
            return


"""
This scenario takes in a robot Articulation and makes it move through its joint DOFs.
Additionally, it adds a cuboid prim to the stage that moves in a circle around the robot.

The particular framework under which this scenario operates should not be taken as a direct
recomendation to the user about how to structure their code.  In the simple example put together
in this template, this particular structure served to improve code readability and separate
the logic that runs the example from the UI design.
"""



class SinusoidJointScenario(ScenarioTemplate):
    def __init__(self):
        self._object = None
        self._articulation = None

        self._running_scenario = False

        self._time = 0.0  # s

        self._object_radius = 0.5  # m
        self._object_height = 0.5  # m
        self._object_frequency = 0.25  # Hz

        self._joint_index = 0
        self._max_joint_speed = 4  # rad/sec
        self._lower_joint_limits = None
        self._upper_joint_limits = None

        self._joint_time = 0
        self._path_duration = 0
        self._calculate_position = lambda t, x: 0
        self._calculate_velocity = lambda t, x: 0


    def load_scenario(self, robot_name, ground_opt):

        add_light_to_stage()

        # print("Assets root path: ", get_assets_root_path())
        need_to_add_articulation = False
        self._robot_name = robot_name
        self._ground_opt = ground_opt
        match self._robot_name:
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
            case "jaka":
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
                self.error(f"Unknown robot name {self._robot_name}")

        if path_to_robot_usd is not None:
            add_reference_to_stage(path_to_robot_usd, robot_prim_path)

        if need_to_add_articulation:
            prim = get_current_stage().GetPrimAtPath(artpath)
            UsdPhysics.ArticulationRootAPI.Apply(prim)

        if self._robot_name == "fancy_franka":
            from omni.isaac.franka import Franka
            self._fancy_robot = Franka(prim_path="/World/Fancy_Franka", name="fancy_franka")
            self._articulation = self._fancy_robot
        else:
            self._articulation = Articulation(artpath)


        # mode specific initialization
        self._cuboid = FixedCuboid(
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
        print("load_scenario done - self._object", self._object)



    def setup_scenario(self):
        # self._articulation = articulation
        # self._object = object_prim


        print("setup_scenario - self._object", self._object)

        self._initial_object_position = self._object.get_world_pose()[0]
        self._initial_object_phase = np.arctan2(self._initial_object_position[1], self._initial_object_position[0])
        self._object_radius = np.linalg.norm(self._initial_object_position[:2])

        self._running_scenario = True

        self._joint_index = 0
        self._lower_joint_limits = self._articulation.dof_properties["lower"]
        self._upper_joint_limits = self._articulation.dof_properties["upper"]
        self._zeros = np.zeros(len(self._lower_joint_limits))
        self._njoints = len(self._lower_joint_limits)
        print(f"jaka - njoints:{self._njoints} lower:{self._lower_joint_limits} upper:{self._upper_joint_limits}")

        # teleport robot to lower joint range
        epsilon = 0.001
        # articulation.set_joint_positions(self._lower_joint_limits + epsilon)
        self._articulation.set_joint_positions(self._zeros + epsilon)

        self._derive_sinusoid_params(0)

    def teardown_scenario(self):
        self._time = 0.0
        # self._object = None
        # self._articulation = None
        self._running_scenario = False

        self._joint_index = 0
        self._lower_joint_limits = None
        self._upper_joint_limits = None

        self._joint_time = 0
        self._path_duration = 0
        self._calculate_position = lambda t, x: 0
        self._calculate_velocity = lambda t, x: 0

    def update_scenario(self, step: float):
        if not self._running_scenario:
            return

        self._time += step

        x = self._object_radius * np.cos(self._initial_object_phase + self._time * self._object_frequency * 2 * np.pi)
        y = self._object_radius * np.sin(self._initial_object_phase + self._time * self._object_frequency * 2 * np.pi)
        z = self._initial_object_position[2]

        self._object.set_world_pose(np.array([x, y, z]))

        self._update_sinusoidal_joint_path(step)

    def _derive_sinusoid_params(self, joint_index: int):
        # Derive the parameters of the joint target sinusoids for joint {joint_index}
        start_position = self._lower_joint_limits[joint_index]
        start_position = 0
        llim = self._lower_joint_limits[joint_index]
        ulim = self._upper_joint_limits[joint_index]
        mjs = self._max_joint_speed

        print(f"jaka - jidx:{joint_index} start_position:{start_position:.3f} llim:{llim:.3f} ulim:{ulim:.3f}")

        P_max = self._upper_joint_limits[joint_index] - start_position
        V_max = self._max_joint_speed
        T = P_max * np.pi / V_max
        print(f"jaka - P_max:{P_max:.3f} V_max:{V_max:.3f} path_duration (T):{T:.3f}")

        # T is the expected time of the joint path

        self._path_duration = T
        self._path_duration = 10
        self._calculate_position = (
            lambda time, path_duration: start_position
            + -P_max / 2 * np.cos(time * 2 * np.pi / path_duration)
            + P_max / 2
        )
        self.lastprint_time = 0
        self._calculate_velocity = lambda time, path_duration: V_max * np.sin(2 * np.pi * time / path_duration)


    def _calculate_position_new(self, time, path_duration):
        start_position = self._lower_joint_limits[self._joint_index]
        P_max = self._upper_joint_limits[self._joint_index] - start_position
        t1 = start_position
        t2 = -P_max / 2 * np.cos(time * 2 * np.pi / path_duration)
        t3 = P_max / 2
        rv = start_position + -P_max / 2 * np.cos(time * 2 * np.pi / path_duration)+ P_max / 2
        rv = t1 + t2 + t3

        # print(f"jaka - t1 {t1:.3f}  \\  t2 {t2:.3f}    \\    t3 {t3:.3f}      \\      rv {rv:.3f} time {time:.3f} path_duration {path_duration:.3f}")

        return rv


    def _update_sinusoidal_joint_path(self, step):
        # Update the target for the robot joints
        self._joint_time += step

        if self._joint_time > self._path_duration:
            self._joint_time = 0
            ojidx = self._joint_index
            self._joint_index = (self._joint_index + 1) % self._articulation.num_dof
            print(f"Changing to joint {self._joint_index} at time {self._time:.3f}")
            self._derive_sinusoid_params(self._joint_index)
            action = ArticulationAction(
                np.array([0]),
                np.array([0]),
                joint_indices=np.array([ojidx])
            )

        joint_position_target = self._calculate_position_new(self._joint_time, self._path_duration, )
        joint_velocity_target = self._calculate_velocity(self._joint_time, self._path_duration)

        if self._joint_time - self.lastprint_time > 1:
           self.lastprint_time = self._joint_time
           print(f"jaka - idx:{self._joint_index} joint time: {self._joint_time:.3f} path duration: {self._path_duration:.3f}")
           print(f"jaka - joint_position_target:{joint_position_target:.3f} joint_velocity_target:{joint_velocity_target:.3f}")


        action = ArticulationAction(
            np.array([joint_position_target]),
            np.array([joint_velocity_target]),
            joint_indices=np.array([self._joint_index])
        )
        self._articulation.apply_action(action)
