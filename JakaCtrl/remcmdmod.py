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

class RemCmd:

    def __init__(self):
        self.cmdatts = {}
        pass

    def __getitem__(self, idx):
        if idx in self.cmdatts:
            return self.cmdatts[idx]
        else:
            return None

    def __setitem__(self, idx, item):
        self.cmdatts[idx] = item


class RemCmdList:

    def __init__(self, narms=2):
        self.listatts = {}
        self.nextcmdid = 0
        self.narms = narms
        self["arms"] = []
        armlist = []
        for i in range(self.narms):
            armid = f"arm_{i}"
            armlist.append(armid)
            self[armid] = []
        self["arms"] = armlist
        self.nextcmdid = 1
        self["lookup"] = {}

    def __getitem__(self, idx):
        if idx in self.listatts:
            return self.listatts[idx]
        else:
            return None

    def __setitem__(self, idx, item):
        self.listatts[idx] = item

    def add_preset_remote_command(self, arm, sourcetray, sourceslot, targettray, targetslot, precmdid="---") -> str:
        cmdid = f"cmd-{self.nextcmdid}"
        cmd = RemCmd()
        cmd["arm"] = arm
        cmd["sourcetray"] = sourcetray
        cmd["sourceslot"] = sourceslot
        cmd["targettray"] = targettray
        cmd["targetslot"] = targetslot
        cmd["id"] = cmdid
        cmd["precmdid"] = precmdid
        cmd["status"] = "pending"
        self[arm].append(cmd)
        self["lookup"][cmdid] = cmd
        self.nextcmdid += 1
        for c in self[arm]:
            print(f"   {c['id']} {c['status']} {c['sourcetray']} {c['sourceslot']} {c['targettray']} {c['targetslot']} {c['precmdid']}")
        return cmdid

    def set_status_command(self, cmdid, status):
        if status not in ["pending", "done", "failed", "executing", "invalid"]:
            carb.log_error(f"Invalid status {status}")
            return
        cmd = self["lookup"][cmdid]
        if cmd is None:
            carb.log_error(f"set_status_command: Command for cmdid {cmdid} not found")
            return
        cmd["status"] = status


    def add_preset_remote_commands(self):
        cid1 = self.add_preset_remote_command("arm_0", "tray1", 0, "tray4", 0)
        cid2 = self.add_preset_remote_command("arm_0", "tray1", 1, "tray4", 1, precmdid=[cid1])
        cid3 = self.add_preset_remote_command("arm_0", "tray1", 2, "tray4", 2, precmdid=[cid2])

        cid4 = self.add_preset_remote_command("arm_1", "tray3", 0, "tray2", 0)
        cid5 = self.add_preset_remote_command("arm_1", "tray3", 1, "tray2", 1, precmdid=[cid4])
        cid6 = self.add_preset_remote_command("arm_1", "tray3", 2, "tray2", 2, precmdid=[cid5,cid3])

    def cmd_prequisites_met(self, cmd):
        prereq = cmd["precmdid"]
        if prereq is None or prereq == "---":
            return True
        # make sure we have a list of prereqs
        if type(prereq) == list:
            prereqlist = prereq
        elif type(prereq) == str:
            prereqlist = [prereq]
        for pr in prereqlist:
            precmd = self["lookup"][pr]
            if precmd is None:
                line = f"Prereq {pr} not found"
                carb.log_error(line)
                return False
            else:
                if precmd["status"] != "done":
                    return False
        return True

    def get_next_command(self, arm):
        armid = f"arm_{arm}"
        cmds = self[armid]
        if cmds is None:
            return None
        for cmd in cmds:
            stat = cmd["status"]
            if stat == "pending":
                if self.cmd_prequisites_met(cmd):
                    return cmd
        return None

    def get_cmd_stats(self):
        ncmd = len(self["lookup"])
        nexe = 0
        npend = 0
        ndone = 0
        ninv = 0
        for cid in self["lookup"]:
            cmd = self["lookup"].get(cid,None)
            if cmd is None:
                carb.log_error(f"Command for cmdid {cid} not found in get_cmd_stats")
            stat = cmd["status"]
            if stat == "executing":
                nexe += 1
            elif stat == "pending":
                npend += 1
            elif stat == "done":
                ndone += 1
            elif stat == "invalid":
                ninv += 1
        return ncmd, nexe, npend, ndone, ninv

    def dump_commands(self):
        for arm in self["arms"]:
            cmds = self[arm]
            line = f"Robot Arm:{arm}"
            print(line)
            for c in cmds:
                line = f"{c['id']} {c['status']} {c['sourcetray']} {c['sourceslot']} {c['targettray']} {c['targetslot']} {c['precmdid']}"
                print(f"   {line}")
