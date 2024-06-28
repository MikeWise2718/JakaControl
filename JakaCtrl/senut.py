import math
import numpy as np
from pxr import Usd, UsdGeom, UsdShade, Gf, UsdPhysics, UsdPhysics, PhysxSchema

from omni.isaac.core.utils.extensions import get_extension_path_from_name # why do we need this?
from omni.isaac.core.utils.rotations import euler_angles_to_quat

from pxr import Sdf, UsdLux, UsdPhysics, Usd
from omni.kit.widget.viewport import ViewportWidget

from omni.isaac.core.utils.stage import get_current_stage

from omni.isaac.core.prims import XFormPrim

import omni
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


def pvk(vek, fmt="{:0.3f}"):
    core = [fmt.format(x) for x in vek]
    # other possibl
    typ = type(vek)
    if typ is list:
        rv = "[" + ", ".join(core) + "]"
    elif typ is tuple:
        rv = "(" + ", ".join(core) + ")"
    elif typ is Gf.Vec3f:
        rv = "g(" + ", ".join(core) + ")"
    else:
        rv = "[" + ", ".join(core) + "]"
    return rv
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

def get_link_paths(dof_paths):
    # runs out in isaac sim each joint is a child of its parent
    # so given a joint path, this finds the link path

    link_paths = []
    for jpath in dof_paths:
        lastslash = jpath.rfind("/") # how nice python has rfind
        link_path = jpath[:lastslash]
        link_paths.append(link_path)
    return link_paths

def set_stiffness_for_joint(joint_name, stiffness):
    stage = get_current_stage()
    # prim = find_prim_by_name(joint_name)
    prim = stage.GetPrimAtPath(joint_name)
    joint = UsdPhysics.DriveAPI.Get(prim, "angular")
    val = joint.GetStiffnessAttr()
    val.Set(stiffness)

def set_damping_for_joint(joint_name, damping):
    stage = get_current_stage()
    # prim = find_prim_by_name(joint_name)
    prim = stage.GetPrimAtPath(joint_name)
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
    # prim = find_prim_by_name(joint_name)
    prim = stage.GetPrimAtPath(joint_name)
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

def checkskipcondition(pathname, skiplist):
    # this allows us to specify skip conditions for certain paths without knowing the full path
    for entry in skiplist:
        if pathname.endswith(entry):
            return True
    return False

def checkfiltcondition(pathname, filtlist):
    # this allows us to specify filter conditions for certain paths without knowing the full path
    for entry in filtlist:
        if pathname.endswith(entry):
            return True
    return False

def apply_collisionapis_to_mesh_and_children_recur(stage, prim, level, exclude=None, include=None, method=None, remove=False):
    if level > 12:
        carb.log_warn("apply_collisionapis_to_mesh_and_children_recur - level too deep ({level})")
        return 0
    # https://forums.developer.nvidia.com/t/script-for-convex-decomposition-collisions/259649/2
    nmesh = 0
    nphysapi = 0
    ncolapi = 0
    typename = prim.GetTypeName()
    pathname = prim.GetPath().pathString
    do_me = True
    if exclude is not None:
        do_me = not checkskipcondition(pathname, exclude)
    elif include is not None:
        do_me = checkfiltcondition(pathname, include)

    if typename == "Mesh" and do_me:
        nmesh += 1
        schemas = prim.GetAppliedSchemas()
        if remove:
            if "PhysicsCollisionAPI" in schemas:
                prim.RemoveAPI(UsdPhysics.CollisionAPI)
                ncolapi += 1
            if "PhysicsMeshCollisionAPI" in schemas:
                prim.RemoveAPI(UsdPhysics.MeshCollisionAPI)
                nphysapi += 1
        else:
            # print("prim:",prim.GetPath()," schemas:",schemas)
            if "PhysicsCollisionAPI" not in schemas:
                UsdPhysics.CollisionAPI.Apply(prim)
                ncolapi += 1
                pass
            if "PhysicsMeshCollisionAPI" not in schemas:
                phycollApi = UsdPhysics.MeshCollisionAPI.Apply(prim)
            else:
                phycollApi = UsdPhysics.MeshCollisionAPI(prim)
            if method is None:
                method = UsdPhysics.Tokens.convexDecomposition
            phycollApi.GetApproximationAttr().Set(method)
            nphysapi += 1
    children = prim.GetChildren()
    for child_prim in children:
        nmdt, coldt, npdt = apply_collisionapis_to_mesh_and_children_recur(stage, child_prim, level+1, exclude=exclude, include=include, method=method, remove=remove)
        nmesh += nmdt
        ncolapi += coldt
        nphysapi += npdt
    return nmesh,  ncolapi, nphysapi


def apply_collisionapis_to_mesh_and_children(stage, primname, exclude=None, include=None, method=None, remove=False):
    prim = stage.GetPrimAtPath(primname)
    nmesh, ncolapi, nphysapi  = apply_collisionapis_to_mesh_and_children_recur(stage, prim, 0, exclude=exclude, include=include, method=method, remove=remove)
    # print(f"apply_collisionapis_to_mesh_and_children:{primname} nmesh:{nmesh} ncolapi:{ncolapi} nphysapi:{nphysapi}")
    return nmesh, ncolapi, nphysapi


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

    #camera_ring_path = "/World/roborg/minicobo_v1_4/dummy_tcp/ring"
def add_rob_cam(cam_root, ring_rot, mount_trans, point_quat, camname="camera"):
    stage = get_current_stage()
    camera_ring_path = f"{cam_root}/ring"
    camera_mount_path = f"{camera_ring_path}/mount"
    camera_point_path = f"{camera_mount_path}/point"
    camera_prim_path = f"{camera_point_path}/{camname}"
    ring_quat = deg_euler_to_quatf(ring_rot)

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


def StrToColor(colorstr: str):
    nclr = colorstr
    if colorstr[0] == "#":
        nclr = colorstr[1:]
    if len(nclr) != 6:
        return (False, [(0.5, 0, 0)])
    r = int(nclr[0:2], 16) / 255.0
    g = int(nclr[2:4], 16) / 255.0
    b = int(nclr[4:6], 16) / 255.0
    color = [(r, g, b)]
    return (True, color)

def StrToGfColor(colorstr: str):
    nclr = colorstr
    if colorstr[0] == "#":
        nclr = colorstr[1:]
    if len(nclr) != 6:
        return (False, [(0.5, 0, 0)])
    r = int(nclr[0:2], 16) / 255.0
    g = int(nclr[2:4], 16) / 255.0
    b = int(nclr[4:6], 16) / 255.0
    color = Gf.Vec3f(r, g, b)
    return (True, color)


def ColorInterpolate(lamda: float, c1l: tuple, c2l: tuple):
    l1 = lamda
    l2 = 1 - lamda
    c1 = c1l[0]
    c2 = c2l[0]
    r = c1[0] * l1 + c2[0] * l2
    g = c1[1] * l1 + c2[1] * l2
    b = c1[2] * l1 + c2[2] * l2
    return [(r, g, b)]


def SetUsdPrimAttrString(graphPrim, attrName, attrValue: str):
    prim: Usd.Prim = graphPrim.GetPrim()
    attr = prim.CreateAttribute(attrName, Sdf.ValueTypeNames.String)
    attr.Set(attrValue)


def SetUsdPrimAttrFloat(graphPrim, attrName, attrValue: float):
    prim: Usd.Prim = graphPrim.GetPrim()
    attr = prim.CreateAttribute(attrName, Sdf.ValueTypeNames.Float)
    attr.Set(attrValue)


def SetUsdPrimAttrInt(graphPrim, attrName, attrValue: int):
    prim: Usd.Prim = graphPrim.GetPrim()
    attr = prim.CreateAttribute(attrName, Sdf.ValueTypeNames.Int)
    attr.Set(attrValue)

def SetUsdPrimAttrStringArray(graphPrim, attrName, attrValue: list):
    prim: Usd.Prim = graphPrim.GetPrim()
    attr = prim.CreateAttribute(attrName, Sdf.ValueTypeNames.StringArray)
    attr.Set(attrValue)

def SetUsdPrimAttrFloatArray(graphPrim, attrName, attrValue: list):
    prim: Usd.Prim = graphPrim.GetPrim()
    attr = prim.CreateAttribute(attrName, Sdf.ValueTypeNames.FloatArray)
    attr.Set(attrValue)

def DefinePrimFromString(stage, primname: str, formname: str):
    if formname == "cone":
        prim = UsdGeom.Cone.Define(stage, primname)
        primlen = 2
    elif formname == "cube":
        prim = UsdGeom.Cube.Define(stage, primname)
        primlen = 2
    elif formname == "sphere":
        prim = UsdGeom.Sphere.Define(stage, primname)
        primlen = 2
    else:
        if formname != "cyl":
            print(f"Unknown form: {formname}")
        prim = UsdGeom.Cylinder.Define(stage, primname)
        primlen = 2
    return (prim, primlen)