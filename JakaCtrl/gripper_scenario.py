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

from .senut import add_sphere_light_to_stage, deg_euler_to_quat, deg_euler_to_quatd, deg_euler_to_quatf
from .senut import find_prim_by_name, find_prims_by_name, GetXformOps
from .senut import apply_convex_decomposition_to_mesh_and_children, apply_material_to_prim_and_children

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
    def __init__(self, uibuilder=None):
        super().__init__()
        self._scenario_name = "gripper"
        self._scenario_desc = ScenarioBase.get_scenario_desc(self._scenario_name)
        self._object = None
        self._articulation = None
        self._nrobots = 1
        self.uibuilder = uibuilder


        self._running_scenario = False

        self._time = 0.0  # s

        self._object_radius = 0.5  # m
        self._object_height = 0.5  # m
        self._object_frequency = 0.25  # Hz

        self._joint_index = 0
        self._max_joint_speed = 4  # rad/sec

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
        super().load_scenario(robot_name, ground_opt)

        self._dc = dc.acquire_dynamic_control_interface()
        self._usd_context = omni.usd.get_context()
        self._stage = get_current_stage()


        add_sphere_light_to_stage()

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
        self.gripperRigidBodyPath = f"{self.gripperActorPath}"

        # Cone that will represent the gripper
        make_rigid_shape = False
        load_asset = False
        assets_root_path = get_assets_root_path()
        disable_gravity = True
        gripper_pt_size = 0.01
        needConvexDecomposition = False
        scale = 1
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
        elif self._robot_name == "cylinder":
            # start_pt = [0, 0, 0.301]
            start_pt = [0, 0, 0.4]
            start_rot = [0, 0, 0]
            gripper_pt = [0, 0, -0.101]
            make_rigid_shape = True
            form = UsdGeom.Cylinder
            disable_gravity = False
            mass = 0.1
        elif self._robot_name == "cube":
            # start_pt = [0, 0, 0.301]
            start_pt = [0, 0, 0.4]
            start_rot = [0, 0, 0]
            gripper_pt = [0, 0, -0.101]
            make_rigid_shape = True
            form = UsdGeom.Cube
            disable_gravity = False
            mass = 0.1
        elif self._robot_name == "cube-yrot":
            # start_pt = [0, 0, 0.301]
            start_pt = [0, 0, 0.4]
            start_rot = [0, 90, 0]
            # gripper_pt = [0, 0, -0.101]
            gripper_pt = [0.101, 0, 0]
            make_rigid_shape = True
            form = UsdGeom.Cube
            disable_gravity = False
            mass = 0.1
        elif self._robot_name == "suction-short":
            start_pt = [0, 0, 0.4]
            start_rot = [0, 90, 0]
            gripper_pt = [0.165, 0, 0]
            # gripper_pt = [0, 0, -0.165]
            load_asset = True
            mass = 0.1
            gripper_pt_size = 0.01
            disable_gravity = False
            asset_path = assets_root_path + "/Isaac/Robots/UR10/Props/short_gripper.usd"
        elif self._robot_name == "suction-dual":
            start_pt = [0, 0, 0.4]
            start_rot = [-90, 0, 0]
            gripper_pt = [0, 0.15, 0]
            # gripper_pt = [0, 0, -0.165]
            load_asset = True
            mass = 0.1
            gripper_pt_size = 0.01
            disable_gravity = True
            scale = 0.01
            # asset_path = "http://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/2023.1.1/Isaac/Robots/UR10/Props/short_gripper.usd"
            # asset_path = assets_root_path + "/Isaac/Samples/Gripper/short_gripper.usd"
            # asset_path = "D:/nv/ov/exts/omni.asimov.jaka/usd/dual_gripper_4small.usda"
            asset_path = "D:/nv/ov/exts/omni.asimov.jaka/usd/dual_gripper_3.usda"
            needConvexDecomposition = True
            # self.gripperRigidBodyPath = f"{self.gripperActorPath}/dual_gripper"
        elif self._robot_name == "suction-dual-0":
            start_pt = [0, 0, 0.4]
            start_rot = [0, 0, 0]
            gripper_pt = [0, 0.15, 0]
            # gripper_pt = [0, 0, -0.165]
            load_asset = True
            mass = 0.1
            gripper_pt_size = 0.01
            disable_gravity = True
            scale = 0.01
            # asset_path = "http://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/2023.1.1/Isaac/Robots/UR10/Props/short_gripper.usd"
            # asset_path = assets_root_path + "/Isaac/Samples/Gripper/short_gripper.usd"
            asset_path = "D:/nv/ov/exts/omni.asimov.jaka/usd/dual_gripper_3.usda"
            # self.gripperRigidBodyPath = f"{self.gripperActorPath}/dual_gripper"

        self.grip_point = Gf.Vec3f(gripper_pt)
        self.start_rot = start_rot
        self.scale = scale

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
            quat = deg_euler_to_quat(start_rot)
            self.gripper_start_pose = dc.Transform(start_pt, quat)
            print("load_asset", asset_path)
            self.gripperGeom = add_reference_to_stage(asset_path, self.gripperActorPath)
            if needConvexDecomposition:
                apply_convex_decomposition_to_mesh_and_children(self._stage, self.gripperActorPath)
            if self.gripperGeom is None:
                print("Failed to load gripper asset")
                return
            gprim : UsdGeom.Gprim = UsdGeom.Gprim(self.gripperGeom)
            (t,r,q,s) = GetXformOps(self.gripperGeom)
            if scale != 1:
                scvek = Gf.Vec3d([scale, scale, scale])
                s.Set(scvek)
            t.Set(Gf.Vec3d(start_pt))
            if r is None:
                quatf = deg_euler_to_quatf(start_rot)
                q.Set(quatf)
            else:
                r.Set(Gf.Vec3d(start_rot))
            mapi = UsdPhysics.MassAPI.Get(self._stage, self.gripperRigidBodyPath)
            if mapi is None:
                mapi = UsdPhysics.MassAPI.Apply(self.gripperGeom)
            mapi.CreateMassAttr(mass).Set(mass)
            rigi = UsdPhysics.RigidBodyAPI.Get(self._stage, self.gripperRigidBodyPath)
            if rigi is None:
                rigi = UsdPhysics.RigidBodyAPI.Apply(self.gripperGeom)
            prim = self._stage.GetPrimAtPath(self.gripperRigidBodyPath)
            physxRigidBody = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
            physxRigidBody.GetDisableGravityAttr().Set(disable_gravity)
            zero = Gf.Vec3f(0, 0, 0)
            if rigi is not None:
                gva = rigi.GetVelocityAttr()
                if gva is not None:
                    gva.Set(zero)
            self.gripperRigidBodyPrim = UsdGeom.Gprim(prim)


        self.gripperGripPointPath = f"{self.gripperActorPath}_GripPoint"
        quatd = deg_euler_to_quatd(start_rot)
        rot = Gf.Rotation(quatd)
        self.grip_scale = 1
        newgripot = rot.TransformDir(self.grip_point*self.grip_scale)
        mk_pt = Gf.Vec3f(start_pt) + newgripot
        self.lastmk_pt = mk_pt
        # self.gripperGripPointPath = f"{self.gripperActorPath}/GripPoint"
        # mk_pt = Gf.Vec3f(gripper_pt)
        orient = euler_angles_to_quat([0, 0, 0])
        sz = gripper_pt_size
        self.gripperGripPointPrim = self.createMarkerBody(
            UsdGeom.Sphere,
            self.gripperGripPointPath ,
            [sz, sz, sz],
            mk_pt,
            orient,
            Gf.Vec3f([1, 0.41, 0.7])
        )

        # Box to be picked
        self.box_start_pose = dc.Transform([0, 0, 0.10], [1, 0, 0, 0])
        self.boxGeom = self.createRigidBody(
            UsdGeom.Cube, "/Box", 0.10, [0.1, 0.1, 0.1], self.box_start_pose.p, self.box_start_pose.r, [0.2, 0.2, 1]
        )

        # Reordering the quaternion to follow DC convention for later use.

        self.box_start_pose = dc.Transform([0, 0, 0.10], [0, 0, 0, 1])

        # Gripper properties
        self.sgp = Surface_Gripper_Properties()
        self.sgp.d6JointPath = f"{self.gripperRigidBodyPath}/SurfaceGripper"
        self.sgp.parentPath = f"{self.gripperRigidBodyPath}"
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
        self.sgp.gripThreshold = 0.1 # orig 0.02
        self.sgp.forceLimit = 1.0e2
        self.sgp.torqueLimit = 1.0e3
        # self.sgp.bendAngle = np.pi / 4
        self.sgp.stiffness = 1.0e4
        self.sgp.damping = 1.0e3

        self.surface_gripper = Surface_Gripper(self._dc)
        self.surface_gripper.initialize(self.sgp)
        # Set camera to a nearby pose and looking directly at the Gripper cone
        self.cone_start_pose = dc.Transform([0, 0, 0.301], [0, 0, 0, 1])
        set_camera_view(
            eye=[20.00, 20.00, 20.00], target=self.cone_start_pose.p, camera_prim_path="/OmniverseKit_Persp"
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

    lastmk_pt = None

    def move_gripperpt_marker(self):
        # Move the gripper point marker right place
        (tranp, _, _, _) = GetXformOps(self.gripperRigidBodyPrim)
        (tranc, _, _, _) = GetXformOps(self.gripperGripPointPrim)
        quatd = deg_euler_to_quatd(self.start_rot)
        rot = Gf.Rotation(quatd)
        newgripot = rot.TransformDir(self.grip_point*self.grip_scale)
        grip_pt = Gf.Vec3f(tranp.Get())
        mk_pt = grip_pt + newgripot
        if Gf.Vec3f(mk_pt-self.lastmk_pt).GetLength() < 0.001:
            return
        print(f"move_gripperpt_marker - grip_pt:{grip_pt} mk_pt:{mk_pt}")
        tranc.Set(mk_pt)
        self.lastmk_pt = mk_pt


    def post_load_scenario(self):
        # self._articulation = articulation
        # self._object = object_prim
        set_camera_view(
            eye=[2.00, 2.00, 2.00], target=self.cone_start_pose.p, camera_prim_path="/OmniverseKit_Persp"
        )


        print("setup_scenario - self._object", self._object)

    laststat = "None"
    def check_gripper_status(self):
        newstat = "Closed" if self.surface_gripper.is_closed() else "Open"
        if newstat != self.laststat:
            print(f"Gripper is now {newstat}")
            self.laststat = newstat
            if newstat == "Closed":
                nhit = apply_material_to_prim_and_children(self._stage, self._matman, self.mat_closed, self.gripperActorPath)
            else:
                nhit = apply_material_to_prim_and_children(self._stage, self._matman, self.mat_open, self.gripperActorPath)

    nphysstep_calls = 0
    def physics_step(self, step_size):
        if self.nphysstep_calls==0:
            pass

        self.move_gripperpt_marker()
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

    def scenario_action(self, actionname, mouse_button=0 ):
        print("Gripper scenario action:",actionname, "   mouse_button:",mouse_button)
        if actionname == "Close Gripper":
            self.surface_gripper.close()
            isclosed = self.surface_gripper.is_closed()
            msg = f"Gripper Close executed - gripper isclosed:{isclosed}"
            carb.log_warn(msg)
        elif actionname == "Open Gripper":
            self.surface_gripper.open()
            isclosed = self.surface_gripper.is_closed()
            msg = f"Gripper Open executed - gripper isclosed:{isclosed}"
            carb.log_warn(msg)
        else:
            print(f"Unknown actionname: {actionname}")

    def get_scenario_actions(self):
        rv = ["Close Gripper", "Open Gripper"]
        return rv