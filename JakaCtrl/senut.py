import math
import os
import numpy as np
import lula
from omni.isaac.motion_generation.lula.interface_helper import LulaInterfaceHelper
from .matman import MatMan
from pxr import Usd, UsdGeom, UsdShade, Gf, UsdPhysics, UsdPhysics, PhysxSchema
from typing import Tuple, List

from omni.isaac.core.utils.rotations import euler_angles_to_quat

from pxr import Sdf, UsdLux, UsdPhysics, Usd

from omni.isaac.core.utils.stage import get_current_stage

from omni.isaac.core.prims import XFormPrim

from omni.isaac.core.utils.extensions import get_extension_path_from_name

import carb.settings
import omni.kit.app


_settings = None

def _init_settings():
    global _settings
    if _settings is None:
        _settings = carb.settings.get_settings()
    return _settings

SETTING_NAME = "/persistent/omni/jaka_control"

def get_setting(name, default, db=False):
    try:
        settings = _init_settings()
        key = f"{SETTING_NAME}/{name}"
        val = settings.get(key)
        if db:
            oval = val
            if oval is None:
                oval = "None"
        if val is None:
            val = default
        if db:
            print(f"get_setting {name} {oval} {val}")
    except Exception as e:
        val = default
        if db:
            print(f"Exception {e} in get_setting {name} {default} {val}")
    return val

def save_setting(name, value):
    settings = _init_settings()
    key = f"{SETTING_NAME}/{name}"
    settings.set(key, value)

# Misc Utilities

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

def find_prims_by_name(prim_name: str):
    stage = get_current_stage()
    found_prims = []
    for prim in stage.Traverse():
        try:
            if prim.GetName().startswith(prim_name):
                found_prims.append(prim)
        except:
            pass
    return found_prims

def find_prim_by_name( prim_name: str) -> Usd.Prim:
    stage = get_current_stage()
    for prim in stage.Traverse():
        try:
            if prim.GetName() == prim_name:
                return prim
        except:
            pass
    return None

def set_stiffness_for_joint(joint_name, stiffness):
    stage = get_current_stage()
    prim = find_prim_by_name(joint_name)
    joint = UsdPhysics.DriveAPI.Get(prim, "angular")
    val = joint.GetStiffnessAttr()
    val.Set(stiffness)

def set_damping_for_joint(joint_name, damping):
    stage = get_current_stage()
    prim = find_prim_by_name(joint_name)
    joint = UsdPhysics.DriveAPI.Get(prim, "angular")
    val = joint.GetDampingAttr()
    val.Set(damping)

def set_stiffness_for_joints(joint_names, stiffness):
    for jp in joint_names:
        set_stiffness_for_joint(jp, stiffness)

def set_damping_for_joints(joint_names, damping):
    for jp in joint_names:
        set_damping_for_joint(jp, damping)

def adjust_joint_values(joint_names, valname, fak):
    for jp in joint_names:
        adjust_joint_value(jp, valname, fak)

def adjust_joint_value(joint_name, valname, fak):
    stage = get_current_stage()
    prim = find_prim_by_name(joint_name)
    joint = UsdPhysics.DriveAPI.Get(prim, "angular")
    if valname=="stiffness":
        val = joint.GetStiffnessAttr()
        newval = val.Get() * fak
        val.Set(newval)
    elif valname=="damping":
        val = joint.GetDampingAttr()
        newval = val.Get() * fak
        val.Set(newval)

def cleanup_path(path: str) -> str:
    if path is not None:
        path = path.replace("\\", "/")
        path = path.replace("//", "/")
    return path

def GetXformOps(prim: Usd.Prim):
    tformop = None
    rformop = None
    qformop = None
    sformop = None
    gprim : UsdGeom.Gprim = UsdGeom.Gprim(prim)
    oops = gprim.GetOrderedXformOps()
    if oops is not None:
        for op in oops:
            match op.GetOpType():
                case UsdGeom.XformOp.TypeTranslate:
                    tformop = op
                case UsdGeom.XformOp.TypeRotateXYZ:
                    rformop = op
                case UsdGeom.XformOp.TypeOrient:
                    qformop = op
                case UsdGeom.XformOp.TypeScale:
                    sformop = op
    if tformop is None:
        tformop = UsdGeom.XformOp(gprim.AddTranslateOp())
    if rformop is None and qformop is None:
        qformop = UsdGeom.XformOp(gprim.AddOrientOp())
    if sformop is None:
        sformop = UsdGeom.XformOp(gprim.AddScaleOp())
    # might need to set the op order here
    return tformop, rformop, qformop, sformop

def deg_euler_to_quat(deg_euler):
    deg = np.array(deg_euler)*np.pi/180
    quat = euler_angles_to_quat(deg)
    return quat

def deg_euler_to_quatf(deg_euler):
    deg = np.array(deg_euler)*np.pi/180
    quat = euler_angles_to_quat(deg)
    quatf = Gf.Quatf(quat[0], quat[1], quat[2], quat[3])
    return quatf

def deg_euler_to_quatd(deg_euler):
    deg = np.array(deg_euler)*np.pi/180
    quat = euler_angles_to_quat(deg)
    quatd = Gf.Quatd(quat[0], quat[1], quat[2], quat[3])
    return quatd

def calc_robot_circle_pose(angle, cen=[0, 0, 0.85], rad=0.35, xang=0, yang=130):
    rads = np.pi*angle/180
    pos = cen + rad*np.array([np.cos(rads), np.sin(rads), 0])
    pos = Gf.Vec3d(list(pos))
    zang = angle-180
    rot = [xang, yang, zang]
    # print("pos:",pos," rot:",rot)
    return pos, rot

# def apply_material_to_prim_and_children_recur(stage, material, prim, level):
#     if level > 4:
#         return
#     gprim = UsdGeom.Gprim(prim)
#     UsdShade.MaterialBindingAPI(gprim).Bind(material)
#     children = prim.GetChildren()
#     for child in children:
#         apply_material_to_prim_and_children_recur(stage, material, child, level+1)


def apply_material_to_prim_and_children_recur(stage, material, prim, level):
    if level > 32:
        return 0
    nhit = 0
    gprim = UsdGeom.Gprim(prim)
    matapi = UsdShade.MaterialBindingAPI(gprim)
    if matapi is not None:
        matapi.Bind(material)
        nhit += 1
    children = prim.GetChildren()
    for child in children:
        nhit += apply_material_to_prim_and_children_recur(stage, material, child, level+1)
    return nhit

def apply_material_to_prim_and_children(stage, matman, matname, primname):
    material = matman.GetMaterial(matname)
    prim = stage.GetPrimAtPath(primname)
    nhit = apply_material_to_prim_and_children_recur(stage, material, prim, 0)
    return nhit


def apply_convex_decomposition_to_mesh_and_children_recur(stage, prim, level):
    if level > 12:
        return 0
    # https://forums.developer.nvidia.com/t/script-for-convex-decomposition-collisions/259649/2
    # collApi = UsdPhysics.CollisionAPI(prim)
    # if collApi is not None:
    #     collApi.GetPhysicsApproximationAttr().Set(UsdPhysics.Tokens.convexDecomposition)
    nhit = 0
    schemas = prim.GetAppliedSchemas()
    # print("prim:",prim.GetPath()," schemas:",schemas)
    if "PhysicsMeshCollisionAPI" in schemas:
        collApi = UsdPhysics.MeshCollisionAPI(prim)
        if collApi is not None:
            sans = collApi.GetSchemaAttributeNames()
            aproxatr = collApi.GetApproximationAttr()
            if aproxatr is not None:
                aproxatr.Set(UsdPhysics.Tokens.convexDecomposition)
                nhit += 1
            # aaa = collApi.GetAttribute("physics:approximation")
            # if aaa is not None:
            #     aaa.Set("convexDecomposition")
        # # aproxatr.Set(UsdPhysics.Tokens.convexDecomposition)
        # collApi.GetAttribute("physics:approximation").Set("convexDecomposition")
    children = prim.GetChildren()
    for child in children:
        nhit += apply_convex_decomposition_to_mesh_and_children_recur(stage, child, level+1)
    return nhit

def apply_convex_decomposition_to_mesh_and_children(stage, primname):
    prim = stage.GetPrimAtPath(primname)
    nhit = apply_convex_decomposition_to_mesh_and_children_recur(stage, prim, 0)
    print("apply_convex_decomposition_to_mesh_and_children:",primname," nit:",nhit)
    return nhit

def apply_diable_gravity_to_rigid_bodies_recur(stage, prim, level, disableGravity=True):
    if level > 12:
        return 0
    nhit = 0
    schemas = prim.GetAppliedSchemas()
    # print("prim:",prim.GetPath()," schemas:",schemas)
    if "PhysicsRigidBodyAPI" in schemas:
        rigi = UsdPhysics.RigidBodyAPI(prim)
        if rigi is not None:
            # prapi = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
            physxRigidBody = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
            physxRigidBody.GetDisableGravityAttr().Set(disableGravity)

            # gda = prapi.GetDisableGravityAttr()
            # gda.Set(True)
    children = prim.GetChildren()
    for child in children:
        nhit += apply_diable_gravity_to_rigid_bodies_recur(stage, child, level+1)
    return nhit

def apply_diable_gravity_to_rigid_bodies(stage, primname,  disableGravity=True):
    prim = stage.GetPrimAtPath(primname)
    nhit = apply_diable_gravity_to_rigid_bodies_recur(stage, prim, 0, disableGravity=disableGravity)
    print("apply_diable_gravity_to_rigid_bodies:",primname," nit:",nhit)
    return nhit


def interp(x, x1, x2, y1, y2):
    if (y1==y2):
        return y1
    return y1 + (x-x1)*(y2-y1)/(x2-x1)