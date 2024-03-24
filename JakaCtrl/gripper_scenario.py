import math
import numpy as np
import os

import carb

from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdPhysics, UsdShade,  PhysxSchema

import omni
import omni.physx as _physx
from omni.isaac.core.utils.stage import add_reference_to_stage,  get_current_stage
from omni.isaac.surface_gripper._surface_gripper import Surface_Gripper, Surface_Gripper_Properties
from omni.isaac.core.utils.viewports import set_camera_view
from omni.isaac.dynamic_control import _dynamic_control as dc

from omni.isaac.core.articulations import Articulation
from omni.isaac.core.objects.cuboid import FixedCuboid
from omni.isaac.core.objects import GroundPlane

from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.core.world import World

from .senut import add_light_to_stage, deg_euler_to_quat
from .senut import find_prim_by_name, find_prims_by_name, GetXformOps

from .scenario_base import ScenarioBase
from omni.isaac.core.objects import cuboid, sphere, capsule
from omni.isaac.core.utils.rotations import euler_angles_to_quat

from omni.isaac.core.utils.nucleus import get_assets_root_path



# Copyright (c) 2022-2023, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#



"""
This scenario takes in a robot Articulation and makes it move through its joint DOFs.
Additionally, it adds a cuboid prim to the stage that moves in a circle around the robot.

The particular framework under which this scenario operates should not be taken as a direct
recomendation to the user about how to structure their code.  In the simple example put together
in this template, this particular structure served to improve code readability and separate
the logic that runs the example from the UI design.
"""



class GripperScenario(ScenarioBase):
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
        # self._lower_joint_limits = None
        # self._upper_joint_limits = None

        self._joint_time = 0
        self._path_duration = 0
        self._calculate_position = lambda t, x: 0
        self._calculate_velocity = lambda t, x: 0

        # Gripper Cone experiment
        # self.sgp.offset.p.z = 0.1001  # does not close -  no error
        #  self.sgp.offset.p.z = 0.1 # does not close - gripper is inside the parent rigid body please move it forwward 0.001000
        # self.sgp.offset.p.z = 0.05 # does not close - gripper is inside the parent rigid body please move it forwward 0.051000
        # self.sgp.offset.p.z = 0.0   # does not close -  gripper is inside the parent rigid body please move it forwward 0.101000
        # self.sgp.offset.p.z = -0.099 #  does not close - gripper is inside the parent rigid body please move it forwward 0.200000
        # self.sgp.offset.p.z = -0.100 #  does not close - gripper is inside the parent rigid body please move it forwward 0.201000
        # self.sgp.offset.p.z = -0.1001 # closes - no error


    def load_scenario(self, robot_name, ground_opt):

        self._dc = dc.acquire_dynamic_control_interface()
        self._usd_context = omni.usd.get_context()
        self._stage = get_current_stage()


        add_light_to_stage()

        self._robot_name = robot_name
        self._ground_opt = ground_opt

        world = World.instance()

        self._target_pos = np.array([0.25, 0.25, 0.15])
        orient = euler_angles_to_quat(np.array([0, 0, 44*np.pi/180]))
        self._cuboid = cuboid.DynamicCuboid(
            "/Scenario/cuboid",
            position=self._target_pos,
            orientation=orient,
            scale=np.array([0.04, 0.1, 0.11]),
            color=np.array([128, 0, 128]
            )
        )

        self.color_closed = Gf.Vec3f(255, 0.0, 0.0)
        self.color_open = Gf.Vec3f(0.0, 255, 0.0)
        self.mat_closed = "Red_Glass"
        self.mat_open = "Green_Glass"
        grippername = "Gripper"
        self.gripperActorPath = f"/{grippername}"

        # Cone that will represent the gripper
        make_rigid_shape = False
        load_asset = False
        assets_root_path = get_assets_root_path()
        disable_gravity = True
        if self._robot_name == "cone":
            start_pt = [0, 0, 0.301]
            start_rot = [0, 0, 0]
            gripper_pt = [0, 0, -0.1001]
            make_rigid_shape = True
            form = UsdGeom.Cone
            disable_gravity = False
            mass = 0.1
        elif self._robot_name == "inverted-cone":
            start_pt = [0, 0, 0.301]
            start_rot = [0, 180, 0]
            gripper_pt = [0, 0, -0.1001]
            make_rigid_shape = True
            form = UsdGeom.Cone
            disable_gravity = False
            mass = 0.2
        elif self._robot_name == "sphere":
            # start_pt = [0, 0, 0.301]
            start_pt = [0, 0, 0.4]
            start_rot = [0, 0, 0]
            gripper_pt = [0, 0, -0.100]
            make_rigid_shape = True
            form = UsdGeom.Sphere
            disable_gravity = False
            mass = 0.1
        elif self._robot_name == "suction-short":
            start_pt = [0, 0, 0.4]
            start_rot = [0, 90, 0]
            gripper_pt = [0, 0, -0.1001]
            load_asset = True
            mass = 0.2
            # asset_path = "http://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/2023.1.1/Isaac/Robots/UR10/Props/short_gripper.usd"
            # asset_path = assets_root_path + "/Isaac/Samples/Gripper/short_gripper.usd"
            asset_path = assets_root_path + "/Isaac/Robots/UR10/Props/short_gripper.usd"


        if make_rigid_shape:
            quat = deg_euler_to_quat(start_rot)
            self.gripper_start_pose = dc.Transform(start_pt, quat)
            self.gripperGeom = self.createRigidBody(
                form,
                self.gripperActorPath,
                0.100,
                [0.10, 0.10, 0.10],
                self.gripper_start_pose.p,
                self.gripper_start_pose.r,
                self.color_open,
            )
        if load_asset:
            quat = euler_angles_to_quat(start_rot)
            self.gripper_start_pose = dc.Transform(start_pt, quat)
            self.gripperGeom = add_reference_to_stage(asset_path, self.gripperActorPath)
            if self.gripperGeom is None:
                print("Failed to load gripper asset")
                return
            gprim : UsdGeom.Gprim = UsdGeom.Gprim(self.gripperGeom)
            (t,r,s) = GetXformOps(self.gripperGeom)
            t.Set(Gf.Vec3d(start_pt))
            r.Set(Gf.Vec3d(start_rot))
            mapi = UsdPhysics.MassAPI.Get(self._stage, self.gripperActorPath)
            if mapi is None:
                mapi = UsdPhysics.MassAPI.Apply(self.gripperGeom)
            mapi.CreateMassAttr(mass).Set(mass)
            rigi = UsdPhysics.RigidBodyAPI.Get(self._stage, self.gripperActorPath)
            if rigi is None:
                rigi = UsdPhysics.RigidBodyAPI.Apply(self.gripperGeom)
            physxRigidBody = PhysxSchema.PhysxRigidBodyAPI.Apply(self.gripperGeom)
            physxRigidBody.GetDisableGravityAttr().Set(disable_gravity)


        self.gripperComPath = f"{self.gripperActorPath}_COM"
        mk_pt = Gf.Vec3f(start_pt) + Gf.Vec3f(gripper_pt)
        self.createMarkerBody(
            UsdGeom.Sphere,
            self.gripperComPath ,
            [0.01, 0.01, 0.01],
            mk_pt,
            self.gripper_start_pose.r,
            Gf.Vec3f([1, 0.41, 0.7])
        )



        # Box to be picked
        self.box_start_pose = dc.Transform([0, 0, 0.10], [1, 0, 0, 0])
        self.boxGeom = self.createRigidBody(
            UsdGeom.Cube, "/Box", 0.10, [0.1, 0.1, 0.1], self.box_start_pose.p, self.box_start_pose.r, [0.2, 0.2, 1]
        )

        # Reordering the quaternion to follow DC convention for later use.
        self.gripper_start_pose = dc.Transform([0, 0, 0.301], [0, 0, 0, 1])
        self.box_start_pose = dc.Transform([0, 0, 0.10], [0, 0, 0, 1])

        # Gripper properties
        self.sgp = Surface_Gripper_Properties()
        self.sgp.d6JointPath = f"{self.gripperActorPath}/SurfaceGripper"
        self.sgp.parentPath = f"{self.gripperActorPath}"
        self.sgp.offset = dc.Transform()
        self.sgp.offset.p.x = 0
        # self.sgp.offset.p.z = 0.1001  # does not close -  no error
        #  self.sgp.offset.p.z = 0.1 # does not close - gripper is inside the parent rigid body please move it forwward 0.001000
        # self.sgp.offset.p.z = 0.05 # does not close - gripper is inside the parent rigid body please move it forwward 0.051000
        # self.sgp.offset.p.z = 0.0   # does not close -  gripper is inside the parent rigid body please move it forwward 0.101000
        # self.sgp.offset.p.z = -0.099 #  does not close - gripper is inside the parent rigid body please move it forwward 0.200000
        # self.sgp.offset.p.z = -0.100 #  does not close - gripper is inside the parent rigid body please move it forwward 0.201000
        # self.sgp.offset.p.z = -0.1001 # closes - no error
        self.sgp.offset.p.z = 0.1 # does not close - gripper is inside the parent rigid body please move it forwward 0.001000
        self.sgp.offset.p.x = gripper_pt[0]
        self.sgp.offset.p.y = gripper_pt[1]
        self.sgp.offset.p.z = gripper_pt[2]
        self.sgp.offset.r = [0.7071, 0, 0.7071, 0]  # Rotate to point gripper in Z direction
        self.sgp.gripThreshold = 0.02
        self.sgp.forceLimit = 1.0e2
        self.sgp.torqueLimit = 1.0e3
        self.sgp.bendAngle = np.pi / 4
        self.sgp.stiffness = 1.0e4
        self.sgp.damping = 1.0e3

        self.surface_gripper = Surface_Gripper(self._dc)
        self.surface_gripper.initialize(self.sgp)
        # Set camera to a nearby pose and looking directly at the Gripper cone
        set_camera_view(
            eye=[4.00, 4.00, 4.00], target=self.gripper_start_pose.p, camera_prim_path="/OmniverseKit_Persp"
        )

        # self._physx_subs = _physx.get_physx_interface().subscribe_physics_step_events(self._on_simulation_step)
        # self._timeline.play()




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

        self._object = self._cuboid
        print("load_scenario done - self._object", self._object)


    def createRigidBody(self, bodyType, boxActorPath, mass, scale, position, rotation, color):
        p = Gf.Vec3f(position[0], position[1], position[2])
        orientation = Gf.Quatf(rotation[0], rotation[1], rotation[2], rotation[3])
        scale = Gf.Vec3f(scale[0], scale[1], scale[2])

        stage = get_current_stage()
        bodyGeom = bodyType.Define(stage, boxActorPath)
        bodyPrim = stage.GetPrimAtPath(boxActorPath)
        bodyGeom.AddTranslateOp().Set(p)
        bodyGeom.AddOrientOp().Set(orientation)
        bodyGeom.AddScaleOp().Set(scale)
        bodyGeom.CreateDisplayColorAttr().Set([color])

        UsdPhysics.CollisionAPI.Apply(bodyPrim)
        if mass > 0:
            massAPI = UsdPhysics.MassAPI.Apply(bodyPrim)
            massAPI.CreateMassAttr(mass)
            # massAPI.CreateCenterOfMassAttr(Gf.Vec3f([0, 0, 0]))
            #  massAPI.CreateDensityAttr(mass)

        UsdPhysics.CollisionAPI(bodyPrim)
        UsdPhysics.RigidBodyAPI.Apply(bodyPrim)

        print(bodyPrim.GetPath().pathString)
        return bodyGeom

    def createMarkerBody(self, bodyType, boxActorPath, scale, position, rotation, color):
        p = Gf.Vec3f(position[0], position[1], position[2])
        orientation = Gf.Quatf(rotation[0], rotation[1], rotation[2], rotation[3])
        scale = Gf.Vec3f(scale[0], scale[1], scale[2])

        stage = get_current_stage()
        bodyGeom = bodyType.Define(stage, boxActorPath)
        bodyPrim = stage.GetPrimAtPath(boxActorPath)
        bodyGeom.AddTranslateOp().Set(p)
        bodyGeom.AddOrientOp().Set(orientation)
        bodyGeom.AddScaleOp().Set(scale)
        bodyGeom.CreateDisplayColorAttr().Set([color])

        print(bodyPrim.GetPath().pathString)
        return bodyGeom

    def move_com_marker(self):
        # Move the COM marker to the center of mass of the gripper
        mapi = UsdPhysics.MassAPI.Get(self._stage, self.gripperActorPath)
        com = mapi.GetCenterOfMassAttr().Get()
        mass = mapi.GetMassAttr().Get()
        isok = np.isfinite(com[0]) and np.isfinite(com[1]) and np.isfinite(com[2])
        if not isok:
            msg = f'move_com_marker:prim {self.gripperActorPath} bad com:{com} mass:{mass:.3f}'
            carb.log_warn(msg)
            return
        mkprim = self._stage.GetPrimAtPath(self.gripperComPath)
        if mkprim is None:
            msg = f'move_com_marker:prim {self.gripperComPath} not found'
            carb.log_warn(msg)
            return
        mkprim.GetAttribute("xformOp:translate").Set(com)
        pass


    def post_load_scenario(self):
        # self._articulation = articulation
        # self._object = object_prim
        print("setup_scenario - self._object", self._object)

    laststat = "None"
    def check_gripper_status(self):
        newstat = "Closed" if self.surface_gripper.is_closed() else "Open"
        if newstat != self.laststat:
            print(f"Gripper is now {newstat}")
            self.laststat = newstat
            if newstat == "Closed":
                # self.coneGeom.GetDisplayColorAttr().Set([self.color_closed])
                self.apply_material(self.mat_closed)

            else:
                # self.coneGeom.GetDisplayColorAttr().Set([self.color_open])
                self.apply_material(self.mat_open)

    nphysstep_calls = 0
    global_time = 0
    global_ang = 0
    def physics_step(self, step_size):
        if self.nphysstep_calls==0:
            pass

        # self.move_com_marker()

        self.check_gripper_status()

        self.nphysstep_calls += 1
        self.global_time += step_size

        return


    def teardown_scenario(self):
        pass


    def update_scenario(self, step: float):
        if not self._running_scenario:
            return
        self.physics_step(step)

    def apply_material(self, matname):
        prim = self._stage.GetPrimAtPath(self.gripperActorPath)
        if prim is None:
            msg = f'realize_eetarg_vis:prim {self.gripperActorPath} not found'
            carb.log_warn(msg)
            return
        gprim = UsdGeom.Gprim(prim)
        material = self._matman.GetMaterial(matname)
        UsdShade.MaterialBindingAPI(gprim).Bind(material)

    def scenario_action(self, actionname, mouse_button=0 ):
        print("Gripper scenario action:",actionname, "   mouse_button:",mouse_button)
        if actionname == "Close Gripper":
            self.surface_gripper.close()
        elif actionname == "Open Gripper":
            self.surface_gripper.open()
        else:
            print(f"Unknown actionname: {actionname}")

    def get_scenario_actions(self):
        rv = ["Close Gripper", "Open Gripper"]
        return rv