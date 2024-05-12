import os
import numpy as np
import copy


import carb
import carb.settings
from .matman import MatMan
from pxr import Usd, UsdGeom, UsdShade, Gf, UsdPhysics
from typing import List

from omni.isaac.core.utils.stage import get_current_stage
from omni.isaac.core.utils.rotations import euler_angles_to_quat

from omni.isaac.core.world import World
from omni.isaac.core.prims import XFormPrim

from omni.isaac.core.utils.stage import add_reference_to_stage

from .senut import apply_convex_decomposition_to_mesh_and_children
from .senut import apply_collisionapis_to_mesh_and_children
from .senut import apply_diable_gravity_to_rigid_bodies
from .senut import apply_material_to_prim_and_children
from .senut import StrToGfColor

from omni.isaac.core.utils.extensions import get_extension_path_from_name

class MotoMP50:
    def __init__(self, mm, idx, name, pos, rot, ska, clr="default_clr"):
        self._mm = mm
        self.idx = idx
        self.name = name
        self.inipos = np.array(pos)
        self.rot = np.array(rot)
        self.pos = np.array(pos)
        self.inirot = np.array(rot)
        self.ska = np.array(ska)
        usdpath = f"/World/moto50mp_{idx}"
        self.usdpath = usdpath
        self.clr = clr
        self.Construct()

    def Construct(self):
        mm = self._mm
        usdpath = self.usdpath
        filepath_to_moto_50mp_usd = f"{mm.current_extension_path}/usd/MOTO_50MP_v2fix.usda"
        add_reference_to_stage(filepath_to_moto_50mp_usd, usdpath)
        quat = euler_angles_to_quat(self.inirot)
        self._moto = XFormPrim(usdpath, scale=self.ska, position=self.inipos, orientation=quat )
        meth = UsdPhysics.Tokens.convexHull
        apply_collisionapis_to_mesh_and_children(mm._stage, usdpath, method=meth)

        prim = mm._stage.GetPrimAtPath(usdpath)
        UsdPhysics.RigidBodyAPI.Apply(prim)
        mapi = UsdPhysics.MassAPI.Apply(prim)
        mapi.CreateMassAttr(0.192) # g54 stats w=73.82 mm, h=161.56, d=8.89, pearl blue
        self.prim = prim
        if self.clr!="default_clr":
            self.ChangeColor(self.clr)

    def GetPose(self):
        tc = Usd.TimeCode.Default()
        xf = UsdGeom.Xformable(self.prim)
        world_transform: Gf.Matrix4d = xf.ComputeLocalToWorldTransform(tc)
        pos = world_transform.ExtractTranslation()
        rot = world_transform.ExtractRotation().GetQuaternion()
        return pos, rot

    def get_world_pose(self):
        rv = self.GetPose()
        return rv

    def ChangeColor(self, colorhexstr):
        mm = self._mm
        ok, rgb = StrToGfColor(colorhexstr)
        if not ok:
            carb.log_error(f"MotoMP50.ChangeColor: colorhexstr {colorhexstr} not valid")
            return
        # "/World/moto50mp_7/MOTO_50MP_v2/Looks/Powder_Coat___Rough__Light_Blue_"
        sdrsubpath = "MOTO_50MP_v2/Looks/Powder_Coat___Rough__Light_Blue_/Powder_Coat___Rough__Light_Blue_"
        sdrpath = f"{self.usdpath}/{sdrsubpath}"
        attname = "inputs:diffuse_color_constant"
        sdrprim = mm._stage.GetPrimAtPath(sdrpath)
        clrattr = sdrprim.GetAttribute(attname)
        clrattr.Set(rgb)

        # attrname = "Aldebo:Aldebo Color"
        # SetUsdPrimAttrFloatArray(clrprim, attrname, rgb)



class MotoTray:
    def __init__(self, mm, idx, name, fillstr, pos, rot, ska):
        self._mm = mm
        self.idx = idx
        self.name = name
        # fillstr is a string of 6 characters, each character is a color code, 0 means empty
        while len(fillstr)<6:
            fillstr += "0"
        fillstr = fillstr[:6]

        self.fillstr = fillstr
        self.inipos = np.array(pos)
        self.rot = np.array(rot)
        self.pos = np.array(pos)
        self.inirot = np.array(rot)
        self.ska = np.array(ska)
        self.w = 0.07382 # in meters
        self.h = 0.16156
        usdpath = f"/World/moto_tray_{idx}"
        self.usdpath = usdpath
        self.construct()

    def construct(self):
        mm: MotoMan = self._mm
        usdpath = self.usdpath

        filepath_to_moto_tray_usd = f"{mm.current_extension_path}/usd/MOTO_TRAY_v2fix.usda"
        add_reference_to_stage(filepath_to_moto_tray_usd, usdpath)
        quat = euler_angles_to_quat(self.inirot)
        self._moto = XFormPrim(usdpath, scale=self.ska, position=self.inipos, orientation=quat )
        # Don't do body1 for now, all the options are too big to let the phone slip through
        #     it needs to be custom vertical and horizontal strips
        # meth = UsdPhysics.Tokens.convexHull
        # options are: boundingCube, convexHull, convexDecomposition and probably a few more
        # apply_collisionapis_to_mesh_and_children(mm._stage, usdpath,
        #                                          filt_end_path=["Body1"],method=meth )
        meth = UsdPhysics.Tokens.convexHull
        apply_collisionapis_to_mesh_and_children(mm._stage, usdpath, include=["Body2"],method=meth )
        prim = mm._stage.GetPrimAtPath(usdpath)
        UsdPhysics.RigidBodyAPI.Apply(prim)
        mapi = UsdPhysics.MassAPI.Apply(prim)
        mapi.CreateMassAttr(0.2)

        # Now add the phones according to fillstr - everywhere it is a 1 we add a phone
        a90 = np.pi/2
        zrot = self.rot[2]

        iw = 0 # 0,1,2  - corresponds to width of mp50 which is 0.07382 meters
        ih = 0 # 0,1    - corresponds to height of mp50 which is 0.16156 meters

        for c in self.fillstr:
            (xp,yp,zp) = self.get_trayslot_pos(iw,ih)
            clr = "default_clr"
            match c:
                case "1": clr = "default_clr"
                case "b": clr = "#0000ff"
                case "r": clr = "#ff0000"
                case "g": clr = "#00ff00"
                case "y": clr = "#ffff00"
                case "o": clr = "#ff8000"
                case "p": clr = "#ff00ff"
                case "w": clr = "#ffffff"
                case "c": clr = "#00ffff"
                case "m": clr = "#800080"
                case "k": clr = "#000000"
                case "0": clr = "skip"


            if c!="skip":
                mm.AddMoto50mp(f"{self.name}_moto_t{self.idx}",pos=[xp,yp,zp],rot=[-a90,0,a90+zrot],ska=[1,1,1],clr=clr)
            iw += 1
            if iw>2:
                iw  = 0
                ih += 1

    def get_trayslot_pos_delt(self, iw,ih):
        # iw range is 0,1,2 and ih range is  0,1
        # this returns the center of the tray slot position
        if iw<0 or 2<iw or ih<0 or 1<ih:
            carb.log_error(f"getpos: iw {iw} or ih {ih} out of range")
            return None
        # yp = (iw-2.5)*self.w + self.pos[0] + iw*0.01
        # xp = (ih+0.0)*self.h + self.pos[1] + ih*0.01 + 0.015
        # zp = 0.02 + self.pos[2]
        iiw = iw - 1
        iih = ih - 0.5
        yp = iiw*self.w + iiw*0.01
        xp = iih*self.h + iih*0.01
        zp = 0.02 + self.pos[2]
        rv = np.array([xp,yp,zp])
        return rv

    def get_trayslot_pos(self, iw,ih):
        delt = self.get_trayslot_pos_delt(iw,ih)
        s = np.sin(self.rot[2])
        c = np.cos(self.rot[2])
        deltrot = np.array([c*delt[0]-s*delt[1], s*delt[0]+c*delt[1], delt[2]])
        rv = self.pos + deltrot
        return rv


class MotoMan:
    def __init__(self, stage, matman: MatMan):
        self._stage = stage
        self.current_extension_path = get_extension_path_from_name("JakaControl")
        self._moto50mp_list = []
        self._moto_tray_list = []
        self._matman = matman

    def AddMoto50mp(self, name, pos=[0,0,0],rot=[0,0,0],ska=[1,1,1],clr="default_clr"):
            idx = len(self._moto50mp_list)
            moto = MotoMP50(self, idx, name, pos, rot, ska, clr)
            self._moto50mp_list.append(moto)
            return moto

    def GetMoto50mpByIdx(self, idx):
        if idx>=len(self._moto50mp_list):
            carb.log_error(f"GetMoto50mpByIdx: idx {idx} out of range")
            return None
        return self._moto50mp_list[idx]

    def GetMoto50mpByName(self, name):
        for moto in self._moto50mp_list:
            if moto["name"] == name:
                return moto
        carb.log_error(f"GetMoto50mpByName: name {name} not found")
        return None

    def AddMotoTray(self, name, fillstr="000000", pos=[0,0,0],rot=[0,0,0],ska=[1.01,1.01,1.01]):
        idx = len(self._moto_tray_list)
        mototray = MotoTray(self, idx, name, fillstr, pos, rot, ska)
        self._moto_tray_list.append(mototray)
        return mototray

    def AddMotoTrayOld(self, name, fillstr="000000", pos=[0,0,0],rot=[0,0,0],ska=[1.01,1.01,1.01]):
        idx = len(self._moto_tray_list)
        usdpath = f"/World/moto_tray_{idx}"
        filepath_to_moto_tray_usd = f"{self.current_extension_path}/usd/MOTO_TRAY_v2fix.usda"
        add_reference_to_stage(filepath_to_moto_tray_usd, usdpath)
        quat = euler_angles_to_quat(rot)
        self._moto = XFormPrim(usdpath, scale=ska, position=pos, orientation=quat )
        # Don't do body1 for now, all the options are too big to let the phone slip through
        #     it needs to be custom vertical and horizontal strips
        # meth = UsdPhysics.Tokens.boundingCube
        # meth = UsdPhysics.Tokens.convexHull
        # apply_collisionapis_to_mesh_and_children(self._stage, usdpath,
        #                                          filt_end_path=["Body1"],method=meth )
        # options are: boundingCube, convexHull, convexDecomposition and probably a few more
        meth = UsdPhysics.Tokens.convexDecomposition
        apply_collisionapis_to_mesh_and_children(self._stage, usdpath,
                                                include=["Body2"],method=meth )

        prim = self._stage.GetPrimAtPath(usdpath)
        UsdPhysics.RigidBodyAPI.Apply(prim)
        mapi = UsdPhysics.MassAPI.Apply(prim)
        mapi.CreateMassAttr(0.2)
        # apply_diable_gravity_to_rigid_bodies(self._stage, usdpath)

        mototray = {"usdpath":usdpath, "prim":prim, "idx":idx, "name":name}
        self._moto_tray_list.append(mototray)

        while len(fillstr)<6:
            fillstr += "0"

        a90 = np.pi/2

        w = 0.07382
        h = 0.16156
        iw = 0 # 0,1,2  - corresponds to width of mp50 which is 0.07382 meters
        ih = 0 # 0,1    - corresponds to height of mp50 which is 0.16156 meters
        for c in fillstr:
            yp = (iw-2.5)*w + pos[0] + iw*0.01
            xp = (ih+0.0)*h + pos[1] + ih*0.01 + 0.015
            zp = 0.02 + pos[2]
            if c=="1":
                self.AddMoto50mp(f"{name}_moto{idx}",pos=[xp,yp,zp],rot=[-a90,0,a90],ska=[1,1,1])
            iw += 1
            if iw>2:
                iw  = 0
                ih += 1

    def GetMotoTrayByIdx(self, idx):
        if idx>=len(self._moto_tray_list):
            carb.log_error(f"GetMotoTrayByIdx: idx {idx} out of range")
            return None
        return self._moto_tray_list[idx]

    def GetMotoTrayByName(self, name):
        for moto in self._moto_tray_list:
            if moto["name"] == name:
                return moto
        carb.log_error(f"GetMotoTrayByName: name {name} not found")
        return None

    def AddCage(self):
        usdpath = "/World/cage_v1"
        # cagevariant = "cage_with_static_colliders"
        cagevariant = "cage_v1"
        if cagevariant == "cage_v1":
            filepath_to_cage_usd = f"{self.current_extension_path}/usd/cage_v1.usda"
            self._cage = XFormPrim(usdpath, scale=[1,1,1], position=[0,0,0])
        else:
            filepath_to_cage_usd = f"{self.current_extension_path}/usd/cage_with_static_colliders.usda"
            sz = 0.0254
            quat = euler_angles_to_quat([np.pi/2,0,0])
            self._cage = XFormPrim(usdpath, scale=[sz,sz,sz], position=[0,0,0], orientation=quat)

        add_reference_to_stage(filepath_to_cage_usd, usdpath)

        # adjust collision shapes
        if cagevariant == "cage_v1":
            meth = UsdPhysics.Tokens.convexHull
            apply_collisionapis_to_mesh_and_children(self._stage, usdpath, method=meth )
        else:
            ppath1 = "ACRYLIC___FIXTURE_V1_v8_1/ACRYLIC___FIXTURE_V1_v8/Body1/Body1"
            ppath2 = "ACRYLIC___FIXTURE_V1_v8_2/ACRYLIC___FIXTURE_V1_v8/Body1/Body1"
            meth = UsdPhysics.Tokens.convexHull
            apply_collisionapis_to_mesh_and_children(self._stage, usdpath, include=[ppath1,ppath2],method=meth )

        apply_material_to_prim_and_children(self._stage, self._matman, "Steel_Blued", usdpath)
        self.cagepath = usdpath