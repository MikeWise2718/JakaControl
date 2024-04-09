import math
import numpy as np
from pxr import Usd, UsdGeom, UsdShade, Gf, UsdPhysics, UsdPhysics, PhysxSchema

from omni.isaac.core.utils.extensions import get_extension_path_from_name # why do we need this?
from omni.isaac.core.utils.rotations import euler_angles_to_quat

from pxr import Sdf, UsdLux, UsdPhysics, Usd

from omni.isaac.core.utils.stage import get_current_stage

from omni.isaac.core.prims import XFormPrim

import carb.settings
from omni.isaac.sensor import Camera

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

def add_sphere_light_to_stage():
    """
    A new stage does not have a light by default.  This function creates a spherical light
    """
    sphereLight = UsdLux.SphereLight.Define(get_current_stage(), Sdf.Path("/World/SphereLight"))
    sphereLight.CreateRadiusAttr(2)
    sphereLight.CreateIntensityAttr(100000)
    XFormPrim(str(sphereLight.GetPath())).set_world_pose([6.5, 0, 12])

def add_dome_light_to_stage():
    """
    A new stage does not have a light by default.  This function creates a dome light
    """
    domeLight = UsdLux.DomeLight.Define(get_current_stage(), Sdf.Path("/World/DomeLight"))
    # domeLight.CreateRadiusAttr(2)
    domeLight.CreateIntensityAttr(1000)

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

def set_stiffness_for_joints(active_joints, stiffness):
    for jp in active_joints:
        set_stiffness_for_joint(jp, stiffness)

def set_damping_for_joints(active_joints, damping):
    for jp in active_joints:
        set_damping_for_joint(jp, damping)

def adjust_joint_values(active_joints, valname, fak):
    for jp in active_joints:
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

def GetXformOpsFromPath(primpath:str):
    prim = get_current_stage().GetPrimAtPath(primpath)
    rv = GetXformOps(prim)
    return rv


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

def calc_robot_circle_pose(angle, cen=[0, 0, 0.85], rad=0.35, xang=0, yang=130):
    rads = np.pi*angle/180
    pos = cen + rad*np.array([np.cos(rads), np.sin(rads), 0])
    pos = Gf.Vec3d(list(pos))
    zang = angle-180
    rot = [xang, yang, zang]
    # print("pos:",pos," rot:",rot)
    return pos, rot


def build_material_dict_recur(stage, dict, prim, level):
    if level > 10:
        carb.log_warn("build_material_dict_recur - level too deep ({level})")
        return 0
    nhit = 0
    gprim = UsdGeom.Gprim(prim)
    matapi = UsdShade.MaterialBindingAPI(gprim)
    if matapi is not None:
        # matname = matapi.GetDirectBindingRel().GetTargets()[0].pathString
        gdbr = matapi.GetDirectBindingRel()
        targets = gdbr.GetTargets()
        if len(targets)>0:
            matname = targets[0].pathString
            # primpath = prim.GetPath().pathString
            primpath = prim.GetPath()
            ppstr = primpath.pathString
            dict[ppstr] = matname
            nhit += 1
    children = prim.GetChildren()
    for child_prim in children:
        nhit += build_material_dict_recur(stage, dict, child_prim, level+1)
    return nhit

def build_material_dict(stage,primname):
    prim = stage.GetPrimAtPath(primname)
    dict = {}
    build_material_dict_recur(stage, dict, prim, 0)
    return dict

def apply_material_to_prim_and_children_recur(stage, material, prim, level):
    if level > 32:
        carb.log_warn("apply_material_to_prim_and_children_recur - level too deep ({level})")
        return 0
    nhit = 0
    gprim = UsdGeom.Gprim(prim)
    matapi = UsdShade.MaterialBindingAPI(gprim)
    if matapi is not None:
        matapi.Bind(material)
        nhit += 1
    children = prim.GetChildren()
    for child_prim in children:
        nhit += apply_material_to_prim_and_children_recur(stage, material, child_prim, level+1)
    return nhit

def apply_material_to_prim_and_children(stage, matman, matname, primname):
    material = matman.GetMaterial(matname)
    prim = stage.GetPrimAtPath(primname)
    nhit = apply_material_to_prim_and_children_recur(stage, material, prim, 0)
    return nhit

def apply_matdict_to_prim_and_children_recur(stage, matdict, prim, level):
    if level > 32:
        carb.log_warn("apply_material_to_prim_and_children_recur - level too deep ({level})")
        return 0
    nhit = 0
    gprim = UsdGeom.Gprim(prim)
    path = prim.GetPath()
    mpath = matdict.get(path.pathString)
    if mpath is not None:
        material = UsdShade.Material(stage.GetPrimAtPath(mpath))
        matapi = UsdShade.MaterialBindingAPI(gprim)
        if matapi is not None:
            matapi.Bind(material)
            nhit += 1

    children = prim.GetChildren()
    for child_prim in children:
        nhit += apply_matdict_to_prim_and_children_recur(stage, matdict, child_prim, level+1)
    return nhit

def apply_matdict_to_prim_and_children(stage, matdict, primname):
    prim = stage.GetPrimAtPath(primname)
    nhit = apply_matdict_to_prim_and_children_recur(stage, matdict, prim, 0)
    return nhit

def apply_convex_decomposition_to_mesh_and_children_recur(stage, prim, level):
    if level > 12:
        carb.log_warn("apply_convex_decomposition_to_mesh_and_children_recur - level too deep ({level})")
        return 0
    # https://forums.developer.nvidia.com/t/script-for-convex-decomposition-collisions/259649/2
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
    children = prim.GetChildren()
    for child_prim in children:
        nhit += apply_convex_decomposition_to_mesh_and_children_recur(stage, child_prim, level+1)
    return nhit

def apply_convex_decomposition_to_mesh_and_children(stage, primname):
    prim = stage.GetPrimAtPath(primname)
    nhit = apply_convex_decomposition_to_mesh_and_children_recur(stage, prim, 0)
    print("apply_convex_decomposition_to_mesh_and_children:",primname," nit:",nhit)
    return nhit

def apply_diable_gravity_to_rigid_bodies_recur(stage, prim, level, disableGravity=True):
    if level > 12:
        carb.log_warn("apply_diable_gravity_to_rigid_bodies_recur - level too deep ({level})")
        return 0
    nhit = 0
    schemas = prim.GetAppliedSchemas()
    # print("prim:",prim.GetPath()," schemas:",schemas)
    if "PhysicsRigidBodyAPI" in schemas:
        rigi = UsdPhysics.RigidBodyAPI(prim)
        if rigi is not None:
            physxRigidBody = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
            physxRigidBody.GetDisableGravityAttr().Set(disableGravity)

    children = prim.GetChildren()
    for child in children:
        nhit += apply_diable_gravity_to_rigid_bodies_recur(stage, child, level+1)
    return nhit

def apply_diable_gravity_to_rigid_bodies(stage, primname,  disableGravity=True):
    prim = stage.GetPrimAtPath(primname)
    nhit = apply_diable_gravity_to_rigid_bodies_recur(stage, prim, 0, disableGravity=disableGravity)
    print("apply_diable_gravity_to_rigid_bodies:",primname," number disabled:",nhit)
    return nhit

def delete_articulations_recur(stage, prim, level):
    if level>12:
        carb.log_warn("delete_articulations_recur - level too deep ({level})")
        return 0
    nhit = 0
    if level>0:
        schemas = prim.GetAppliedSchemas()
        if "PhysicsArticulationRootAPI" in schemas:
            prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)
            nhit += 1
    children = prim.GetChildren()
    for child in children:
        nhit += delete_articulations_recur(stage, child, level+1)
    return nhit

def adjust_articulationAPI_location_if_needed(stage, primname):
    prim = stage.GetPrimAtPath(primname)
    schemas = prim.GetAppliedSchemas()
    nadd = 0
    if not "PhysicsArticulationRootAPI" in schemas:
        UsdPhysics.ArticulationRootAPI.Apply(prim)
        nadd += 1
    schemas = prim.GetAppliedSchemas()
    nhit = delete_articulations_recur(stage, prim, 0)
    if nhit>0 or nadd>0:
        msg = f"adjust_articulation - added {nadd} and removed {nhit} articulationAPIs from {primname}"
        carb.log_info(msg)
        print(msg)
    return

def interp(x, x1, x2, y1, y2):
    if (y1==y2):
        return y1
    return y1 + (x-x1)*(y2-y1)/(x2-x1)

def add_cam(robot_name, cam_root):
    #camera_ring_path = "/World/roborg/minicobo_v1_4/dummy_tcp/ring"
    stage = get_current_stage()
    camera_ring_path = f"{cam_root}/ring"
    camera_mount_path = f"{camera_ring_path}/mount"
    camera_point_path = f"{camera_mount_path}/point"
    camera_prim_path = f"{camera_point_path}/camera"
    if robot_name == "minicobo-dual-sucker":
        ring_rot = Gf.Vec3f([0,0,-45])
    else:
        ring_rot = Gf.Vec3f([0,0,0])
    ring_quat = deg_euler_to_quatf(ring_rot)
    mount_trans = Gf.Vec3f([0.011,0.147,-0.011])


    point_quat = Gf.Quatf(0.80383,Gf.Vec3f(-0.19581,-0.46288,-0.31822))

    ovcam = Camera(
        prim_path=camera_prim_path,
        resolution=[512,512]
    )
    ring = UsdGeom.Xform.Define(stage, camera_ring_path)
    [rtop,rrop,rqop,rsop] = GetXformOpsFromPath(camera_ring_path)
    rqop.Set(ring_quat)
    mount = UsdGeom.Xform.Define(stage, camera_mount_path)
    [mtop,mrop,mqop,msop] = GetXformOpsFromPath(camera_mount_path)
    mtop.Set(mount_trans)
    point = UsdGeom.Xform.Define(stage, camera_point_path)
    [ptop,prop,pqop,psop] = GetXformOpsFromPath(camera_point_path)
    pqop.Set(point_quat)
    # [ctop,crop,cqop,csop] = GetXformOpsFromPath(camera_camera_path)

    markername = f"{camera_mount_path}/marker"
    markerXform = UsdGeom.Xform.Define(stage, markername)
    [mktop,mkrop,mkqop,mksop] = GetXformOpsFromPath(markername)
    mksop.Set((0.005, 0.005, 0.005))
    spherePrim = UsdGeom.Sphere.Define(stage, markername + '/sphere')
    spherePrim.GetDisplayColorAttr().Set([(0, 0.6, 0.6)])


    # OpenCV camera matrix and width and height of the camera sensor, from the calibration file
    width, height = 1920, 1200
    camera_matrix = [[958.8, 0.0, 957.8], [0.0, 956.7, 589.5], [0.0, 0.0, 1.0]]

    # Pixel size in microns, aperture and focus distance from the camera sensor specification
    # Note: to disable the depth of field effect, set the f_stop to 0.0. This is useful for debugging.
    pixel_size = 3 * 1e-3   # in mm, 3 microns is a common pixel size for high resolution cameras
    f_stop = 1.8            # f-number, the ratio of the lens focal length to the diameter of the entrance pupil
    focus_distance = 0.6    # in meters, the distance from the camera to the object plane

    # Calculate the focal length and aperture size from the camera matrix
    ((fx,_,cx),(_,fy,cy),(_,_,_)) = camera_matrix
    horizontal_aperture =  pixel_size * width                   # The aperture size in mm
    vertical_aperture =  pixel_size * height
    focal_length_x  = fx * pixel_size
    focal_length_y  = fy * pixel_size
    focal_length = (focal_length_x + focal_length_y) / 2         # The focal length in mm

    # Set the camera parameters, note the unit conversion between Isaac Sim sensor and Kit
    ovcam.set_focal_length(focal_length / 10.0)                # Convert from mm to cm (or 1/10th of a world unit)
    ovcam.set_focus_distance(focus_distance)                   # The focus distance in meters
    ovcam.set_lens_aperture(f_stop * 100.0)                    # Convert the f-stop to Isaac Sim units
    # camera.set_horizontal_aperture(horizontal_aperture / 10.0)  # Convert from mm to cm (or 1/10th of a world unit)
    # camera.set_vertical_aperture(vertical_aperture / 10.0)

    ovcam.set_clipping_range(0.1, 1.0e5)

    return ovcam, camera_prim_path
