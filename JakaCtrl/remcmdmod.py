import os
import numpy as np
import copy
import time
import glob
import json
import carb
import carb.settings
from omni.isaac.core.utils.extensions import get_extension_path_from_name # why do we need this?
from .senut import cleanup_path


class RemCmd:

    def __init__(self):
        self.cmdatts = {}
        pass

    def __getitem__(self, idx):
        if idx in self.cmdatts:
            return self.cmdatts[idx]
        else:
            return None

    def __contains__(self, item):
        rv = item in self.cmdatts
        return rv

    def __setitem__(self, idx, item):
        self.cmdatts[idx] = item


class RemCmdList:

    reverse_count = 0

    def __init__(self, narms=2):
        self.initialize(narms)

    def initialize(self, narms=2):
        self.listatts = {}
        self.nextcmdid = 0
        self.narms = narms
        self["arms"] = []
        armlist = []
        for i in range(self.narms):
            armid = f"arm{i}"
            armlist.append(armid)
            self[armid] = []
        self["arms"] = armlist
        self.nextcmdid = 1
        self.lookup = {}
        self.startime = time.time()

    def __getitem__(self, idx):
        if idx in self.listatts:
            return self.listatts[idx]
        else:
            return None

    def __setitem__(self, idx, item):
        self.listatts[idx] = item

    def __contains__(self, item):
        rv = item in self.listatts
        return rv


    def add_remote_command(self, arm, sourcetray, sourceslot, targettray, targetslot, precmdid=None, cmdid=None) -> str:
        if cmdid is None:
            cmdid = f"cmd-{self.nextcmdid}"
        cmd = RemCmd()
        cmd["arm"] = arm
        cmd["sourcetray"] = sourcetray
        cmd["sourceslot"] = sourceslot
        cmd["targettray"] = targettray
        cmd["targetslot"] = targetslot
        cmd["id"] = cmdid
        if precmdid is None:
            cmd["precmdid"] = []
        else:
            cmd["precmdid"] = precmdid
        cmd["status"] = "pending"
        cmd["tbeg"] = 0.0
        cmd["tfin"] = 0.0
        self[arm].append(cmd)
        self.lookup[cmdid] = cmd
        self.nextcmdid += 1
        # for c in self[arm]:
        #     print(f"   {c['id']} {c['status']} {c['sourcetray']} {c['sourceslot']} {c['targettray']} {c['targetslot']} {c['precmdid']}")
        return cmdid

    def set_status_command(self, cmdid, status) -> None:
        if status not in ["pending", "done", "failed", "executing", "invalid"]:
            carb.log_error(f"Invalid status {status}")
            return
        cmd = self.lookup[cmdid]
        if cmd is None:
            carb.log_error(f"set_status_command: Command for cmdid {cmdid} not found")
            return
        if status == "executing":
            cmd["tbeg"] = time.time()-self.startime
        if status == "done":
            cmd["tfin"] = time.time()-self.startime
        cmd["status"] = status

    def start_execution(self):
        for arm in self["arms"]:
            cmds = self[arm]
            for cmd in cmds:
                cmd["status"] = "pending"
        self.startime = time.time()

    @staticmethod
    def get_gpt_remote_command_options() -> list[str]:
        current_extension_dir = cleanup_path(get_extension_path_from_name("JakaControl"))
        # get a directory listing of the gpt files (in json format)
        mask = f"{current_extension_dir}/gptcommands/*.json"
        gpt_files = glob.glob(mask)
        for i,fname in enumerate(gpt_files):
            newfname = cleanup_path(fname)
            if newfname != fname:
                gpt_files[i] = newfname
        gpt_files.sort()
        return gpt_files

    def crack_tray_slot(self, tray_slot_str) -> tuple[str,int]:
        ok = False
        tray_slot_str = tray_slot_str.strip().lower()
        if not tray_slot_str.startswith("tray"):
            carb.log_error(f"Invalid tray_slot {tray_slot_str}")
            return ok, 0, 0
        traystr, slotstr = tray_slot_str.split("_")
        traynum = int(traystr[-1])
        slotnum = int(slotstr[-1])
        ok = True
        return ok, traynum, slotnum

    def process_move(self, carr) -> None:
        if len(carr) < 6:
            carb.log_error(f"Invalid move command {carr}")
            return
        id = carr[1]
        arm = carr[2].strip().lower()
        phone = carr[3].strip().lower()
        ok, sourcetray, sourceslot = self.crack_tray_slot( carr[4] )
        if not ok:
            carb.log_error(f"Invalid source tray_slot {carr[1]}")
            return
        ok, targettray, targetslot = self.crack_tray_slot( carr[5] )
        if not ok:
            carb.log_error(f"Invalid target tray_slot {carr[2]}")
            return
        self.add_remote_command(arm, sourcetray, sourceslot, targettray, targetslot, cmdid=id)


    def read_commands_from_file(self, fullpathname) -> None:
        with open(fullpathname) as fd:
            json_data = json.load(fd)
        rcmdseq = json_data["sequence"]
        for rcmd in rcmdseq:
            rcmd = rcmd.replace("(",",")
            rcmd= rcmd.replace(")",",")
            carr = rcmd.split(",")
            if len(carr) == 0:
                carb.log_error(f"Invalid command {rcmd}")
                continue
            key = carr[0].lower().strip()
            match key:
                case "move":
                    self.process_move(carr)
                case "_":
                    carb.log_error(f"Invalid or unimplemented command {rcmd}")

    def add_preset_remote_commands(self) -> None:
        cid1 = self.add_remote_command("arm0", "tray1", 0, "tray4", 0)
        cid2 = self.add_remote_command("arm0", "tray1", 1, "tray4", 1, precmdid=[cid1])
        cid3 = self.add_remote_command("arm0", "tray1", 2, "tray4", 2, precmdid=[cid2])

        cid4 = self.add_remote_command("arm1", "tray3", 0, "tray2", 0)
        cid5 = self.add_remote_command("arm1", "tray3", 1, "tray2", 1, precmdid=[cid4])
        cid6 = self.add_remote_command("arm1", "tray3", 2, "tray2", 2, precmdid=[cid5,cid3])

    def cmd_prequisites_met(self, cmd) -> bool:
        prereq = cmd["precmdid"]
        if prereq is None or prereq == []:
            return True
        # make sure we have a list of prereqs
        if type(prereq) == list:
            prereqlist = prereq
        elif type(prereq) == str:
            prereqlist = [prereq]
        for pr in prereqlist:
            precmd = self.lookup[pr]
            if precmd is None:
                line = f"Prereq {pr} not found"
                carb.log_error(line)
                return False
            else:
                if precmd["status"] != "done":
                    return False
        return True

    def get_next_command(self, arm):
        armid = f"arm{arm}"
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
        ncmd = len(self.lookup)
        nexe = 0
        npend = 0
        ndone = 0
        ninv = 0
        for cid in self.lookup:
            cmd = self.lookup[cid]
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
                tbeg = c["tbeg"]
                tfin = c["tfin"]
                line = f"{c['id']} {tbeg:6.1f} {tfin:6.1f} {c['status']} {c['sourcetray']} {c['sourceslot']} {c['targettray']} {c['targetslot']} {c['precmdid']}"
                print(f"   {line}")

    def get_latest_executed_command(self):
        maxtime = -1
        maxcmd = None
        keys = copy.deepcopy(list(self.lookup.keys()))
        revkeys = reversed(keys)
        for cmdid in revkeys:
            cmd = self.lookup[cmdid]
            if cmd["status"] in ["pending","done"]:
                if cmd["tfin"] > maxtime:
                    maxtime = cmd["tfin"]
                    maxcmd = cmd
        if maxcmd is not None:
            maxcmd["status"] = "retrieved"
        return maxcmd

    def reverse_commands(self):

        old_to_new_map = {}

        # make a list of dependencies - we will need to add these back in
        old_dependencies = []
        for cmdid in self.lookup:
            cmd = self.lookup[cmdid]
            for d_id in cmd["precmdid"]:
                if d_id in self.lookup:
                    # remember that cmd["id"] depends on key being executed first
                    old_dependencies.append([cmd["id"],d_id])
                else:
                    carb.log_error(f"reverse_commands - building old_dependencies could not find key {d_id} - this should not happen")
            cmd["revstatus"] = "pending"

        maxiter = len(self.lookup)+1
        iter = 0
        cmdstack = []
        while True:
            cmd = self.get_latest_executed_command()
            if cmd is None:
                break
            if iter > maxiter:
                break
            cmdstack.append(cmd)
            iter += 1

        # now we have a stack of commands in the reverse order they were executed

        self.initialize(self.narms)
        for cmd in cmdstack:
            arm = cmd["arm"]
            sourcetray = cmd["targettray"]
            sourceslot = cmd["targetslot"]
            targettray = cmd["sourcetray"]
            targetslot = cmd["sourceslot"]
            newcmdid = self.add_remote_command(arm, sourcetray, sourceslot, targettray, targetslot)
            old_to_new_map[cmd["id"]] = newcmdid

        # print("old_to_new_map")
        # print(old_to_new_map)

        # print("old_dependencies")
        # print(old_dependencies)

        # now add the dependencies back, translating the old ids to the new ids
        for dep in old_dependencies:
            o_id0 = dep[0]
            o_id1 = dep[1]
            if o_id0 not in old_to_new_map:
                carb.log_error(f"reverse_commands - could not find old key {o_id0} - this should not happen")
                continue
            if o_id1 not in old_to_new_map:
                carb.log_error(f"reverse_commands - could not find old key {o_id1} - this should not happen")
                continue
            n_id0 = old_to_new_map[o_id0]
            n_id1 = old_to_new_map[o_id1]
            if n_id0 not in self.lookup:
                carb.log_error(f"reverse_commands - could not find new key {n_id0} - this should not happen")
                continue
            if n_id1 not in self.lookup:
                carb.log_error(f"reverse_commands - could not find new key {n_id1} - this should not happen")
                continue
            newcmd = self.lookup[n_id1]
            # print(f"Adding prereq {n_id0} to {n_id1}  - original {o_id0} {o_id1}")
            newcmd["precmdid"].append(n_id0)

        self.reverse_count += 1