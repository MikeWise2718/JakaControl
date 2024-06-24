# This software contains source code provided by NVIDIA Corporation.
# Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
import carb
import numpy as np

import omni.timeline
import omni.kit

import omni.ui as ui
from omni.isaac.core.utils.stage import create_new_stage
from omni.ui import Button
from omni.ui import color as uiclr
from omni.isaac.ui.element_wrappers import CollapsableFrame, StateButton
from omni.isaac.ui.element_wrappers.core_connectors import LoadButton, ResetButton
from omni.isaac.ui.ui_utils import get_style
from omni.usd import StageEventType
from .senut import set_stiffness_for_joints, set_damping_for_joints


from .scenario_base import ScenarioBase
from .invkin_scenario import InvkinScenario
from .rmp_scenario import RMPflowScenario
from .rmp_new_scenario import RMPflowNewScenario
from .pickplace_scenario import PickAndPlaceScenario
from .pickplace_new_scenario import PickAndPlaceNewScenario
from .franka_pickplace_scenario import FrankaPickAndPlaceScenario
from .sinusoid_scenario import SinusoidJointScenario
from .object_inspection_scenario import ObjectInspectionScenario
from .cage_rmpflow_scenario import CageRmpflowScenario
from .gripper_scenario import GripperScenario

from .senut import get_setting, save_setting

class UIBuilder:
    btwhite = uiclr("#fffff")
    btblack = uiclr("#000000")
    btgray = uiclr("#808080")
    btgreen = uiclr("#00ff00")
    btblue = uiclr("#6060ff")
    btred = uiclr("#ff0000")
    btyellow = uiclr("#ffff00")
    btpurple = uiclr("#ff00ff")
    btcyan = uiclr("#00ffff")
    dkgreen = uiclr("#004000")
    dkblue = uiclr("#000040")
    dkred = uiclr("#400000")
    dkyellow = uiclr("#404000")
    dkpurple = uiclr("#400040")
    dkcyan = uiclr("#004040")
    _scenario_names = ScenarioBase.get_scenario_names()
    _scenario_name = ScenarioBase.get_default_scenario()
    _robot_names = ["ur3e"]
    _robot_name = "ur3e"
    _ground_opts = ["none", "default", "groundplane", "groundplane-blue"]
    _ground_opt = "default"
    _joint_alarms = True
    _modes = ["CollisionSpheres","none"]
    _mode = "none"
    _choices = ["choice 1","choice 2"]
    _choice = "choice 1"
    _scenario_action_list = ["--none--"]
    _action = "--none--"
    _colprims = None
    _colvis_opts = ["Invisible", "Red", "RedGlass", "Glass"]
    _collider_vis = "Invisible"
    _eevis_opts = ["Invisible", "Blue", "BlueGlass", "Glass"]
    _eetarg_vis = "Invisible"
    _rmptarg_vis_opts = ["Invisible", "Yellow"]
    _rmptarg_vis = "Invisible"
    _rotate_opts = ["None", "RotateForward", "none", "RotateBackward"]
    _rotate_opt = "None"
    _robskin_opts = ["Default",
                     "Clear Glass","Tinted Glass 85","Tinted Glass 75","Tinted Glass",
                     "Red Glass","Green Glass","Blue Glass", "Red/Green Glass", "Red/Blue Glass", "Green/Blue Glass"]
    _robskin_opt = "Default"
    _last_created_robot_name = ""

    def __init__(self):
        # Frames are sub-windows that can contain multiple UI elements
        self.frames = []
        # UI elements created using a UIElementWrapper instance
        self.wrapped_ui_elements = []

        # Get access to the timeline to control stop/pause/play programmatically
        self._timeline = omni.timeline.get_timeline_interface()

        self._robot_names =ScenarioBase.get_scenario_robots("all")
        self._robot_name = self.find_valid_robot_name(self._scenario_name, self._robot_name, 1)


        # Run initialization for the provided example
        self._on_init()

    def SaveSettings(self):
        # print("SaveSettings")
        try:
            save_setting("p_robot_name", self._robot_name)
            save_setting("p_ground_opt", self._ground_opt)
            save_setting("p_robskin_opt", self._robskin_opt)
            save_setting("p_scenario_name", self._scenario_name)
            save_setting("p_mode", self._mode)
            save_setting("p_choice", self._choice)
            save_setting("p_action", self._action)
            save_setting("p_joint_alarms", self._joint_alarms)
            # print(f"SaveSettings p_joint_alarms:{self._joint_alarms}")

        except Exception as e:
            carb.log_error(f"Exception in SaveSettings: {e}")

    def LoadSettings(self):
        # print("LoadSettings")
        self._robot_name = get_setting("p_robot_name", self._robot_name)
        self._ground_opt = get_setting("p_ground_opt", self._ground_opt)
        self._robskin_opt = get_setting("p_robskin_opt", self._robskin_opt)
        self._joint_alarms = get_setting("p_joint_alarms", self._joint_alarms)
        # print(f"LoadSettings p_joint_alarms:{self._joint_alarms}")

        self._scenario_name = get_setting("p_scenario_name", self._scenario_name)
        self._base_scenario_action_list = ScenarioBase.get_scenario_actions(self._scenario_name)
        self._scenario_action_list = ScenarioBase.get_scenario_actions(self._scenario_name)
        self._base_robot_action_list = ScenarioBase.get_robot_actions(self._scenario_name)
        self._robot_action_list = ScenarioBase.get_robot_actions(self._scenario_name)

        # set the default action in the UI
        action = get_setting("p_action", self._action)
        self._action = action if action in self._scenario_action_list else self._scenario_action_list[0]

        self._mode = get_setting("p_mode", self._mode)
        self._choice = get_setting("p_choice", self._choice)
        # print("Done LoadSettings")

    ###################################################################################
    #           The Functions Below Are Called Automatically By extension.py
    ###################################################################################

    def on_menu_callback(self):
        """Callback for when the UI is opened from the toolbar.
        This is called directly after build_ui().
        """
        pass

    def on_timeline_event(self, event):
        """Callback for Timeline events (Play, Pause, Stop)

        Args:
            event (omni.timeline.TimelineEventType): Event Type
        """
        if event.type == int(omni.timeline.TimelineEventType.STOP):
            # When the user hits the stop button through the UI, they will inevitably discover edge cases where things break
            # For complete robustness, the user should resolve those edge cases here
            # In general, for extensions based off this template, there is no value to having the user click the play/stop
            # button instead of using the Load/Reset/Run buttons provided.
            self._scenario_state_btn.reset()
            self._scenario_state_btn.enabled = False

    def on_physics_step(self, step: float):
        """Callback for Physics Step.
        Physics steps only occur when the timeline is playing

        Args:
            step (float): Size of physics step
        """
        pass

    def on_stage_event(self, event):
        """Callback for Stage Events

        Args:
            event (omni.usd.StageEventType): Event Type
        """
        if event.type == int(StageEventType.OPENED):
            # If the user opens a new stage, the extension should completely reset
            self._reset_extension()

    def cleanup(self):
        """
        Called when the stage is closed or the extension is hot reloaded.
        Perform any necessary cleanup such as removing active callback functions
        Buttons imported from omni.isaac.ui.element_wrappers implement a cleanup function that should be called
        """
        for ui_elem in self.wrapped_ui_elements:
            ui_elem.cleanup()

        self.SaveSettings()

    def build_ui(self):
        """
        Build a custom UI tool to run your extension.
        This function will be called any time the UI window is closed and reopened.
        """
        # print("build_ui")
        world_config_frame = CollapsableFrame("World Config", collapsed=False)
        self.world_config_frame = world_config_frame

        with world_config_frame:
            with ui.VStack(style=get_style(), spacing=5, height=0):
                with ui.HStack(style=get_style(), spacing=5, height=0):
                    ui.Label("Scenario Name:",
                            style={'color': self.btyellow},
                            width=50)

                    self._scenario_name_combobox = ui.ComboBox(
                        style={'background_color': self.dkblue, "font_size": 22},
                        name="Scenario Name"
                    )
                    for s_name in self._scenario_names:
                        self._scenario_name_combobox.model.append_child_item(None, ui.SimpleStringModel(s_name))
                    
                    self._scenario_name_combobox.model.get_item_value_model().set_value(self._scenario_name)
                    self._scenario_name_combobox.model.add_item_changed_fn(self._combobox_change_scenario_name)

                with ui.HStack(style=get_style(), spacing=5, height=0):
                    ui.Label("Scenario Desc:",
                            style={'color': self.btyellow},
                            width=50)
                    self._scenario_desc_lab = ui.Label(
                        ScenarioBase.get_scenario_desc(self._scenario_name),
                        style={'color': self.btwhite},
                        word_wrap=True
                    )
                with ui.HStack(style=get_style(), spacing=5, height=0):
                    ui.Label("Robot Name:",
                            style={'color': self.btyellow},
                            width=50)
                    self._robot_btn = Button(
                        self._robot_name, mouse_pressed_fn=self._change_robot_name,
                        style={'background_color': self.dkred}
                    )
                    ui.Label("Ground:",
                            style={'color': self.btyellow},
                            width=50)
                    self._ground_btn = Button(
                        self._ground_opt, clicked_fn=self._change_ground_opt,
                        style={'background_color': self.dkred}
                    )
                with ui.HStack(style=get_style(), spacing=5, height=0):
                    ui.Label("Robot Desc:",
                            style={'color': self.btyellow},
                            width=50)
                    self._robot_desc_lab = ui.Label(
                        ScenarioBase.get_robot_desc(self._robot_name),
                        style={'color': self.btwhite},
                        word_wrap=True
                    )
                with ui.HStack(style=get_style(), spacing=5, height=0):
                    ui.Label("Mode:",
                            style={'color': self.btyellow},
                            width=50)
                    self._mode_btn = Button(
                        self._mode, clicked_fn=self._change_mode,
                        style={'background_color': self.dkred}
                    )
                    ui.Label("Choice:",
                            style={'color': self.btyellow},
                            width=50)
                    self._choice_btn = Button(
                        self._choice, clicked_fn=self._change_choice,
                        style={'background_color': self.dkred}
                    )
                with ui.HStack(style=get_style(), spacing=5, height=0):
                    ui.Label("RMP status:",
                            style={'color': self.btyellow},
                            width=50)
                    self._rmpactive_btn = Button(
                        "stopped", mouse_pressed_fn=self._change_rmp_active,
                        style={'background_color': self.dkred}
                    )
                    ui.Label("Rotate:",
                            style={'color': self.btyellow},
                            width=50)
                    self._rotate_opt_btn = Button(
                        self._rotate_opt, mouse_pressed_fn=self._change_rotate,
                        style={'background_color': self.dkred}
                    )
                with ui.HStack(style=get_style(), spacing=5, height=0):
                    ui.Label("Action Selection:",
                            style={'color': self.btyellow},
                            width=50)
                    self._actionsel_btn = Button(
                        self._action, mouse_pressed_fn=self._change_action,
                        style={'background_color': self.dkgreen}
                    )
                    self._executeaction_btn = Button(
                        "Execute", mouse_pressed_fn=self._exec_action,
                        style={'background_color': self.dkred}
                    )
                # self.wrapped_ui_elements.append(self._robot_btn)

        scenario_actions_frame = CollapsableFrame("Scenario Actions", collapsed=False)

        with scenario_actions_frame:
            self._scenario_action_vstack = ui.VStack(style=get_style(), spacing=5, height=0)
            self.load_scenario_action_vstack()

        robot_actions_frame = CollapsableFrame("Robot Actions", collapsed=False)

        with robot_actions_frame:
            self._robot_action_vstack = ui.VStack(style=get_style(), spacing=5, height=0)
            self.load_robot_action_vstack()

        rob_appearance_frame = CollapsableFrame("Robot Appearance", collapsed=False)

        with rob_appearance_frame:
            with ui.VStack(style=get_style(), spacing=5, height=0):
                with ui.HStack(style=get_style(), spacing=5, height=0):
                    ui.Label("Collider Vis:",
                            style={'color': self.btyellow},
                            width=50)
                    self._collider_vis_btn = Button(
                        self._collider_vis, mouse_pressed_fn=self._change_collider_vis,
                        style={'background_color': self.dkred}
                    )
                    ui.Label("EEtarg Vis:",
                            style={'color': self.btyellow},
                            width=50)
                    self._eetarg_vis_btn = Button(
                        self._eetarg_vis, mouse_pressed_fn=self._change_eetarg_vis,
                        style={'background_color': self.dkred}
                    )
                with ui.HStack(style=get_style(), spacing=5, height=0):
                    ui.Label("RMP Target Vis:",
                            style={'color': self.btyellow},
                            width=50)
                    self._rmptarg_vis_btn = Button(
                        self._rmptarg_vis, mouse_pressed_fn=self._change_rmptarg_vis,
                        style={'background_color': self.dkred}
                    )
                    ui.Label("Robot Skin:",
                            style={'color': self.btyellow},
                            width=50)
                    self._robskin_opt_btn = Button(
                        self._robskin_opt, mouse_pressed_fn=self._change_robskin_opt,
                        style={'background_color': self.dkred}
                    )

        world_controls_frame = CollapsableFrame("World Controls", collapsed=False)

        with world_controls_frame:
            with ui.VStack(style=get_style(), spacing=5, height=0):
                self._load_btn = LoadButton(
                    "Load/Create Scene", "Create", setup_scene_fn=self._setup_scene, setup_post_load_fn=self._setup_post_load
                )
                self._load_btn.set_world_settings(physics_dt=1 / 60.0, rendering_dt=1 / 60.0)
                self.wrapped_ui_elements.append(self._load_btn)


                self._reset_btn = ResetButton(
                    "Reset Button", "Reset", pre_reset_fn=None, post_reset_fn=self._on_post_reset_btn
                )
                self._reset_btn.enabled = False
                self.wrapped_ui_elements.append(self._reset_btn)

        run_scenario_frame = CollapsableFrame("Run Scenario")

        with run_scenario_frame:
            with ui.VStack(style=get_style(), spacing=5, height=0):
                self._scenario_state_btn = StateButton(
                    "Run Scenario",
                    "Run",
                    "Stop",
                    on_a_click_fn=self._on_run_scenario_a_text,
                    on_b_click_fn=self._on_run_scenario_b_text,
                    physics_callback_fn=self._update_scenario,
                )
                self._scenario_state_btn.enabled = False
                self.wrapped_ui_elements.append(self._scenario_state_btn)

        robot_config_frame = CollapsableFrame("Robot Config")

        with robot_config_frame:
            self.rob_config_vstack = ui.VStack(style=get_style(), spacing=5, height=0)
            with self.rob_config_vstack:
                self._show_robot_config_btn = Button(
                        "Show Robot Config", clicked_fn=self._show_robot_config,
                        style={'background_color': self.dkblue}
                )
                self._show_robot_config_btn.enabled = True
                # self.wrapped_ui_elements.append(self._show_robot_config_btn)

        self.build_ui_scenario_dependent()

    robot_joints_frame = None
    robot_joints_frame1 = None
    rob_joints_vstack = None
    rob_joints_vstack1 = None

    def build_ui_scenario_dependent(self):

        if self.robot_joints_frame is not None:
            if self.rob_joints_vstack is not None:
                self.rob_joints_vstack.clear()
            self.clear_ui_dict_of_robot(0)
        else:
            self.robot_joints_frame = CollapsableFrame("Robot DOF Joints 0")

        if self.robot_joints_frame1 is not None:
            if self.rob_joints_vstack1 is not None:
                self.rob_joints_vstack1.clear()
            self.clear_ui_dict_of_robot(1)
        else:
            self.robot_joints_frame1 = CollapsableFrame("Robot DOF Joints 1")

        if self._cur_scenario is None:
            return

        if self._cur_scenario._nrobots > 0:
            with self.robot_joints_frame:
                self.rob_joints_vstack = ui.VStack(style=get_style(), spacing=5, height=0)
                with self.rob_joints_vstack:
                    sjv_fn = lambda: self._show_joint_values_for_robot(0,self.rob_joints_vstack)
                    self._show_robot_joint_btn = Button(
                            "Show Robot DOF Joints 0", clicked_fn=sjv_fn,
                            style={'background_color': self.dkblue}
                    )
                    self._show_robot_joint_btn.enabled = True


        if self._cur_scenario._nrobots > 1:
            with self.robot_joints_frame1:
                self.rob_joints_vstack1 = ui.VStack(style=get_style(), spacing=5, height=0)
                with self.rob_joints_vstack1:
                    sjv_fn = lambda: self._show_joint_values_for_robot(1,self.rob_joints_vstack1)
                    self._show_robot_joint_btn1 = Button(
                            "Show Robot DOF Joints 1", clicked_fn=sjv_fn,
                            style={'background_color': self.dkblue}
                    )
                    self._show_robot_joint_btn1.enabled = True


    ######################################################################################
    # Functions Below This Point Support The Provided Example And Can Be Deleted/Replaced
    ######################################################################################

    custbuttdict = {}
    def load_scenario_action_vstack(self):
        self._scenario_action_vstack.clear()
        with self._scenario_action_vstack:
            self._scenario_action_list = self._cur_scenario.get_scenario_actions()
            for action in self._scenario_action_list:
                btnclr = self.dkred
                if action in self._base_scenario_action_list:
                    btnclr = self.dkblue
                def do_action(action):
                    return lambda x,y,b,m: self._do_scenario_action(action, x,y,b,m)
                button_text = self._cur_scenario.get_scenario_action_button_text( action, None )
                tool_tip = self._cur_scenario.get_scenario_action_button_tooltip( action, None )
                # print(f"load_action_vstack {action} {button_text}")
                butt = Button(
                    button_text, mouse_pressed_fn=do_action(action),
                    tooltip = tool_tip,
                    style={'background_color': btnclr}
                )
                self.custbuttdict[action] = butt

    def load_robot_action_vstack(self):
        self._robot_action_vstack.clear()
        with self._robot_action_vstack:
            self._robot_action_list = self._cur_scenario.get_robot_actions()
            for action in self._robot_action_list:
                btnclr = self.dkpurple
                if action in self._base_robot_action_list:
                    btnclr = self.dkblue
                def do_action(action):
                    return lambda x,y,b,m: self._do_robot_action(action, x,y,b,m)
                button_text = self._cur_scenario.get_scenario_action_button_text( action, None )
                tool_tip = self._cur_scenario.get_scenario_action_button_tooltip( action, None )
                # print(f"load_robot_action_vstack {action} {button_text}")
                butt = Button(
                    button_text, mouse_pressed_fn=do_action(action),
                    tooltip = tool_tip,
                    style={'background_color': btnclr}
                )
                self.custbuttdict[action] = butt

    cfg_lab_dict = {}
    config_line_list = []
    def _load_one_param(self, rcfg,  param_name, clr):
        pname = param_name
        if hasattr(rcfg, pname):
            val = getattr(rcfg, pname)
        else:
            val = f"{param_name} not found in robcfg"

        l1txt = f"{param_name}"
        l2txt = f"{val}"
        self.config_line_list.append(f"{l1txt}: {l2txt}")
        if param_name in self.cfg_lab_dict:
            (l1, l2) = self.cfg_lab_dict[param_name]
            l1.text = l1txt
            l2.text = l2txt
        else:
            hstack = ui.HStack(style=get_style(), spacing=5, height=0)
            with hstack:
                l1 = ui.Label(l1txt, style={'color': self.btwhite}, width=120)
                l2 = ui.Label(l2txt, style={'color': clr})
            self.cfg_lab_dict[param_name] = (l1, l2)
            self.rob_config_vstack.add_child(hstack)

    def _add_title(self, title, clr):
        self.config_line_list.append(f"{title}")
        hstack = ui.HStack(style=get_style(), spacing=5, height=0)
        with hstack:
            ui.Label(title, style={'color': clr}, width=120)
        self.rob_config_vstack.add_child(hstack)

    def _copy_to_clipboard(self):
        str = "\n".join(self.config_line_list)
        omni.kit.clipboard.copy(str)

    def get_robot_config(self, index=0):
        return self._cur_scenario.get_robot_config(index)

    def _show_robot_config(self, index=0):
        # print(f"_show_robot_config {index}")
        rc = self.get_robot_config(index)
        # print(f"  rc {rc.robot_name} {rc.robot_id} {rc.robmatskin}")
        self.rob_config_vstack.clear()
        self.cfg_lab_dict = {}
        self.config_line_list = []
        nrobots = self._cur_scenario._nrobots
        with self.rob_config_vstack:
            with ui.HStack(style=get_style(), spacing=5, height=0):
                for i in range(nrobots):
                    def show_config(i):
                        return lambda: self._show_robot_config(i)
                    butt = Button(
                            f"Show Robot Config {i}", clicked_fn=show_config(i),
                            style={'background_color': self.dkblue}
                    )
                    butt.enabled = True

        bl = self.btblue
        gn = self.btgreen
        yt = self.btyellow
        cy = self.btcyan
        self._add_title("Identity", bl)
        self._load_one_param(rc, "robot_name", cy)
        self._load_one_param(rc, "robot_id", cy)
        self._load_one_param(rc, "manufacturer", cy)
        self._load_one_param(rc, "model", cy)
        self._load_one_param(rc, "grippername", cy)
        self._load_one_param(rc, "desc", cy)
        self._load_one_param(rc, "robot_prim_path", cy)
        self._load_one_param(rc, "ground_opt", cy)
        self._load_one_param(rc, "eeframe_name", cy)

        self._add_title("Parameters", bl)
        self._load_one_param(rc, "max_step_size", cy)
        self._load_one_param(rc, "stiffness", cy)
        self._load_one_param(rc, "damping", cy)

        self._add_title("Directories", bl)
        self._load_one_param(rc, "mg_extension_dir", gn)
        self._load_one_param(rc, "rmp_config_dir", gn)
        self._load_one_param(rc, "jc_extension_dir", gn)

        self._add_title("Config Files", bl)
        self._load_one_param(rc, "urdf_path", yt)
        self._load_one_param(rc, "rdf_path", yt)
        self._load_one_param(rc, "rmp_config_path", yt)
        self._load_one_param(rc, "robot_usd_file_path", yt)
        # print("done _show_robot_config")

    def _rot_robot_joint(self, robot_idx, joint_idx, jname, inc):
        rc = self.get_robot_config(robot_idx)
        art = rc._articulation
        jidx = int(joint_idx)
        jidxlist = [jidx]
        pos = art.get_joint_positions()
        rinc = self.joint_inc_step if inc>0 else -self.joint_inc_step
        newjpos = pos[jidx] + rinc*np.pi/180
        art.set_joint_positions(joint_indices=jidxlist, positions=[newjpos])
        self.refresh_robot_joint_values(robot_idx, joint_idx)

        # print(f"_rot_robot_joint robot_idx:{robot_idx} joint_idx {joint_idx} ({jname}) by {rinc} degrees")

    def add_spheres_to_joints(self, x, y, b, m, ridx=0):
        self._cur_scenario.add_spheres_to_joints(ridx)

    def show_joint_limit_warnings(self, x, y, b, m, ridx=0):
        rc = self.get_robot_config(ridx)
        stat = self._cur_scenario.toggle_show_joints_close_to_limits(ridx)
        onoff = "On" if stat else "Off"
        txt = f"Joint Limit Warnings for {ridx} - {onoff}"
        butts =  self.joint_ui_dict.get((ridx, -1, "joint-option-buttons"))
        if butts is not None:
            butt1 = butts[0]
            butt1.text = txt
        else:
            carb.log.error(f"show_joint_limit_warnings - no button found for robot {ridx}")
        # self._show_joint_limit_warnings_btn.text = txt
        # print(f"{txt} - {rc.robot_id} - {rc.show_joints_close_to_limits}")

    def _change_joint_inc(self, x, y, b, m):
        if b == 0:
            self.joint_inc_step *= 2
        else:
            self.joint_inc_step *= 0.5
        self._joint_inc_btn.text = f"Joint inc: {self.joint_inc_step}"

    def _change_joint_stiffness(self, x, y, b, m, ridx=-1):
        rc = self.get_robot_config(ridx)
        if b == 0:
            rc.stiffness *= 1.125
        else:
            rc.stiffness /= 1.125
        set_stiffness_for_joints(rc.dof_paths, rc.stiffness)
        self._adjust_stiffness_btn.text = f"Stiffness: {rc.stiffness:.2f}"

    def _change_joint_damping(self, x, y, b, m, ridx=-1):
        rc = self.get_robot_config(ridx)
        if b == 0:
            rc.damping *= 1.125
        else:
            rc.damping /= 1.125
        set_damping_for_joints(rc.dof_paths, rc.damping)
        self._adjust_damping_btn.text = f"Damping: {rc.damping:.2f}"

    joint_ui_dict = {}
    def clear_ui_dict_of_robot(self, robot_idx):
        # print(f"clear_ui_dict_of_robot {robot_idx} nkeys {len(self.joint_ui_dict)} ")
        kez = list(self.joint_ui_dict.keys())
        for k in kez:
            (k_ridx, _, _) = k
            if k_ridx == robot_idx:
                del self.joint_ui_dict[k]
        # print(f"clear_ui_dict_of_robot - done -  {robot_idx} nkeys {len(self.joint_ui_dict)} ")

    def _show_joint_values_for_robot(self, robot_idx=0, robvstack=None):
        if robvstack is None:
            robvstack = self.rob_joints_vstack
        robvstack.clear()
        # print(f"_show_joint_values_for_robot {robot_idx}")
        rc = self.get_robot_config(robot_idx)
        if rc is None:
            msg = f"Robot {robot_idx} not found - probably not initialized with \"Create\" yet"
            # print(msg)
            carb.log_warn(msg)
            robvstack.add_child(ui.Label(msg, style={'color': self.btred}))
            return
        # print(f"  rc {rc.robot_name} {rc.robot_id} {rc.robmatskin}")
        self.rob_config_stack = ui.VStack(style=get_style(), spacing=5, height=0)
        self.clear_ui_dict_of_robot(robot_idx)
        self.joint_inc_step = 5
        nrobots = self._cur_scenario._nrobots
        with robvstack:
            with ui.HStack(style=get_style(), spacing=5, height=0):
                for ridx in range(nrobots):
                    def show_joints_for_robot(i):
                        return lambda: self._show_joint_values_for_robot(i,robvstack)
                    butt = Button(
                            f"Show Joints of Robot {ridx}", clicked_fn=show_joints_for_robot(ridx),
                            style={'background_color': self.dkblue}
                    )
                    butt.enabled = True
            with ui.HStack(style=get_style(), spacing=5, height=0):
                sjw_fn = lambda x,y,b,m: self.show_joint_limit_warnings(x,y,b,m, ridx=robot_idx)
                onoff = "On" if rc.show_joints_close_to_limits else "Off"
                butt1 = self._show_joint_limit_warnings_btn = Button(
                        f"Joint Limit Warnings for {robot_idx} - {onoff}", mouse_pressed_fn=sjw_fn,
                        style={'background_color': self.dkpurple}
                )
                self._show_joint_limit_warnings_btn.enabled = True
                butt2 = self._add_spheres_to_joints_btn = Button(
                        f"Add Spheres to Joints of {robot_idx}", mouse_pressed_fn=self.add_spheres_to_joints,
                        style={'background_color': self.dkpurple}
                )
                self._add_spheres_to_joints_btn.enabled = True
                self._joint_inc_btn = Button(
                        f"Joint inc:{self.joint_inc_step}", mouse_pressed_fn=self._change_joint_inc,
                        style={'background_color': self.dkgreen}
                )
                self._joint_inc_btn.enabled = True
                ajs_fn = lambda x,y,b,m: self._change_joint_stiffness(x,y,b,m, ridx=robot_idx)
                butt3 = self._adjust_stiffness_btn = Button(
#                         f"Stiffness:{rc.stiffness}", mouse_pressed_fn=self._change_joint_stiffness,
                        f"Stiffness:{rc.stiffness:.2f}", mouse_pressed_fn=ajs_fn,
                        style={'background_color': self.dkcyan}
                )
                self._adjust_stiffness_btn.enabled = True
                ajd_fn = lambda x,y,b,m: self._change_joint_damping(x,y,b,m, ridx=robot_idx)
                butt4 = self._adjust_damping_btn = Button(
                        f"Damping:{rc.damping:.2f}", mouse_pressed_fn=ajd_fn,
                        style={'background_color': self.dkcyan}
                )
                self._adjust_damping_btn.enabled = True
                self.joint_ui_dict[(robot_idx,-1,"joint-option-buttons")] = (butt1,butt2,butt3,butt4)


        hstack = ui.HStack(style=get_style(), spacing=5, height=0)
        with hstack:
            ui.Label(f"idx:{robot_idx} - {rc.robot_name} - {rc.robot_id} - {rc.robmatskin}", style={'color': self.btwhite}, width=120)
            labtime = ui.Label("", style={'color': self.btwhite}, width=120)
        robvstack.add_child(hstack)
        # if not hasattr(rc, "_articulation"):
        #     carb.log_warn(f"Robot {robot_idx} has no articulation - probably not initialized yet")
        #     return
        for j,jn in enumerate(rc.dof_names):
            self.config_line_list.append(f"{jn}")
            hstack = ui.HStack(style=get_style(), spacing=5, height=0)
            def rot_joint(j,jn,inc):
                return lambda: self._rot_robot_joint(robot_idx,j,jn,inc)
            with hstack:
                labstyle = {'color': self.btwhite}
                btnstyle = {'background_color': self.dkgreen}
                lab1 = ui.Label("", style=labstyle, width=120)
                lab2 = ui.Label("", style=labstyle, width=120)
                lab3 = ui.Label("", style=labstyle, width=120)
                but1 = ui.Button("", clicked_fn=rot_joint(j,jn,+1), style=btnstyle)
                but2 = ui.Button("", clicked_fn=rot_joint(j,jn,-1), style=btnstyle)
                lab4 = ui.Label("", style=labstyle, width=120)
            self.joint_ui_dict[(robot_idx,j,jn)] = (labtime, lab1, lab2, lab3, but1, but2, lab4)
            robvstack.add_child(hstack)
            self.refresh_robot_joint_values(robot_idx, j)
        # print("done _show_joint_values_for_robot")

    def refresh_robot_joint_values(self, robot_idx, joint_idx):
        rc = self.get_robot_config(robot_idx)
        j = joint_idx
        jn = rc.dof_names[j]
        uientry = self.joint_ui_dict.get((robot_idx,j,jn))
        if uientry is None:
            print(f"refresh_robot_joint_values - no entry for robot {robot_idx} joint {j} {jn}")
            return
        self._cur_scenario.check_alarm_status(rc)
        txttime = f"Secs: {self._cur_scenario.global_time:.3f}"
        art = rc._articulation
        pos = art.get_joint_positions()
        props = art.dof_properties
        stiff = props["stiffness"][j]
        damp = props["damping"][j]
        degs = 180/np.pi
        jpos = degs*pos[j]
        llim = degs*rc.lower_dof_lim[j]
        ulim = degs*rc.upper_dof_lim[j]
        lmb = 100*rc.dof_lamda[j]
        jtyp = rc.dof_types[j]
        txt1 = f"{j}: {jn}"
        txt2 = f"{llim:.1f} to {ulim:.1f}"
        txt3 = f" cur: {jpos:.1f}  ({lmb:.1f}%)"
        txt4 = f"{jtyp}  --  stiff-damp: {stiff:8.1f} {damp:8.1f}"

        clr = self.btred if rc.dof_alarm[j] else self.btwhite
        alarmstyle={'color': clr}
        (labtime, lab1, lab2, lab3, but1, but2, lab4) = uientry
        labtime.text = txttime
        lab1.text = txt1
        lab2.text = txt2
        lab3.text = txt3
        lab4.text = txt4
        but1.text = "+"
        but2.text = "-"
        lab3.set_style(alarmstyle)

    def refresh_open_robot_joint_values(self):
        if hasattr(self, "joint_ui_dict"):
            for (robot_idx,j,_) in self.joint_ui_dict:
                if j>=0:
                    self.refresh_robot_joint_values(robot_idx, j)
        self._cur_scenario.realize_joint_alarms_for_all()

    def pick_scenario(self, scenario_name):
        if scenario_name == "sinusoid-joint":
            self._cur_scenario = SinusoidJointScenario(self)
        elif scenario_name == "pick-and-place":
            self._cur_scenario = PickAndPlaceScenario(self)
        elif scenario_name == "pick-and-place-new":
            self._cur_scenario = PickAndPlaceNewScenario(self)
        elif scenario_name == "franka-pick-and-place":
            self._cur_scenario = FrankaPickAndPlaceScenario(self)
        elif scenario_name == "rmpflow":
            self._cur_scenario = RMPflowScenario(self)
        elif scenario_name == "rmpflow-new":
            self._cur_scenario = RMPflowNewScenario(self)
        elif scenario_name == "object-inspection":
            self._cur_scenario = ObjectInspectionScenario()
        elif scenario_name == "cage-rmpflow":
            self._cur_scenario = CageRmpflowScenario(self)
        elif scenario_name == "inverse-kinematics":
            self._cur_scenario = InvkinScenario(self)
        elif scenario_name == "gripper":
            self._cur_scenario = GripperScenario(self)
        else:
            self._cur_scenario = SinusoidJointScenario(self)
        self._cur_scenario.show_joint_limits_for_all_robots(showthem=self._joint_alarms)

        # self._cur_scenario.show_joint_limits_for_all_robots = self._joint_alarms

    def _on_init(self):
        # self._articulation = None
        # self._cuboid = None
        self.LoadSettings()
        self.pick_scenario(self._scenario_name)
        # print("Done _on_init")

    def _setup_scene(self):
        # print("ui_builder._setup_scene")
        self.pick_scenario(self._scenario_name)

        create_new_stage()

        self._cur_scenario.load_scenario(self._robot_name, self._ground_opt)

        self._cur_scenario.realize_robot_skin(self._robskin_opt)

        self._scenario_action_list = self._cur_scenario.get_scenario_actions()
        if len(self._scenario_action_list) > 0:
            self._action = self._scenario_action_list[0]
        else:
            self._action = ""
        self._actionsel_btn.text = self._action
        self._last_created_robot_name = self._robot_name
        self.load_scenario_action_vstack()
        self.load_robot_action_vstack()

    def get_next_val_safe(self, lst, val, inc=1):
        try:
            idx = lst.index(val)
            idx = (idx + inc) % len(lst)
            rv = lst[idx]
        except:
            idx = 0
            rv = lst[idx]
        return rv

    binc = [-1, 1]

    def _do_scenario_action(self, action, x,y,b,km):
        argdict = {"k":km, "b":b, "x":x, "y":y}
        self._cur_scenario.scenario_action(action, argdict)
        butt = self.custbuttdict.get(action)
        if butt is not None:
            butt.text = self._cur_scenario.get_scenario_action_button_text( action, argdict )

    def _do_robot_action(self, action, x,y,b,m):
        argdict = {"k":m, "b":b, "x":x, "y":y}
        self._cur_scenario.robot_action(action, argdict)
        self._refresh_robot_action_button_texts()
        # butt = self.custbuttdict.get(action)
        # if butt is not None:
        #     butt.text = self._cur_scenario.get_robot_action_button_text( action, argdict )

    def _refresh_robot_action_button_texts(self):
        for action in self._robot_action_list:
            butt = self.custbuttdict.get(action)
            if butt is not None:
                butt.text = self._cur_scenario.get_robot_action_button_text( action, None )

    def _change_action(self, x, y, b, m):
        self._action = self.get_next_val_safe(self._scenario_action_list, self._action, self.binc[b])
        self._actionsel_btn.text = self._action

    def _change_choice(self):
        self._choice = self.get_next_val_safe(self._choices, self._choice)
        self._choice_btn.text = self._choice

    def _change_mode(self):
        self._mode = self.get_next_val_safe(self._modes, self._mode)
        self._mode_btn.text = self._mode


    def find_valid_robot_name(self, scenario_name, robot_name, binc=1):
        cur_robot = robot_name
        iter = 0
        maxiters = len(self._robot_names)
        # iterate until we find a robot that the current scenario can handle
        while True:
            robot_name = self.get_next_val_safe(self._robot_names, robot_name, binc)
            if ScenarioBase.can_handle_robot(scenario_name, robot_name):
                # print(f"Found valid robot name {robot_name} for scenario {scenario_name}")
                # if ScenarioBase.can_handle_robot(scenario_name, self._last_created_robot_name):
                    # print(f"Overrode robot name {robot_name} with {self._last_created_robot_name}")
                #     robot_name = self._last_created_robot_name
                break
            iter += 1
            if iter > maxiters:
                return ""
        return robot_name

    def _change_robot_name(self, x, y, b, m):
        nx_robot_name = self.find_valid_robot_name(self._scenario_name, self._robot_name, self.binc[b])
        if nx_robot_name != "":
            self._robot_name = nx_robot_name
            self._robot_btn.text = self._robot_name
            self._robot_desc_lab.text = ScenarioBase.get_robot_desc(self._robot_name)

    def _change_scenario_name(self, x, y, b, m):
        self._scenario_name = self.get_next_val_safe(self._scenario_names, self._scenario_name, self.binc[b])
        # self._scenario_name_btn.text = self._scenario_name
        self._scenario_desc_lab.text = ScenarioBase.get_scenario_desc(self._scenario_name)
        if not ScenarioBase.can_handle_robot(self._scenario_name, self._robot_name):
            self._robot_name = self.find_valid_robot_name(self._scenario_name, self._robot_name, 1)
            self._robot_btn.text = self._robot_name
            self._robot_desc_lab.text = ScenarioBase.get_robot_desc(self._robot_name)

    def _combobox_change_scenario_name(self, item_model, item):
        item_index: int = item_model.get_item_value_model().get_value_as_int()
        selected_scenario: str = self._scenario_names[item_index]

        print(f"_combobox_change_scenario_name {selected_scenario}")
        self._scenario_name = selected_scenario
        # self._scenario_name_btn.text = selected_scenario
        self._scenario_desc_lab.text = ScenarioBase.get_scenario_desc(selected_scenario)
        if not ScenarioBase.can_handle_robot(self._scenario_name, self._robot_name):
            self._robot_name = self.find_valid_robot_name(self._scenario_name, self._robot_name, 1)
            self._robot_btn.text = self._robot_name
            self._robot_desc_lab.text = ScenarioBase.get_robot_desc(self._robot_name)

    def _change_collider_vis(self, x, y, b, m):
        self._collider_vis = self.get_next_val_safe(self._colvis_opts, self._collider_vis, self.binc[b])
        self._collider_vis_btn.text = self._collider_vis
        self._cur_scenario.realize_collider_vis_opt(self._collider_vis)

    def _change_eetarg_vis(self, x, y, b, m):
        self._eetarg_vis = self.get_next_val_safe(self._eevis_opts, self._eetarg_vis, self.binc[b])
        self._eetarg_vis_btn.text = self._eetarg_vis
        self._cur_scenario.realize_eetarg_vis(self._eetarg_vis)

    def _change_rmptarg_vis(self, x, y, b, m):
        self._rmptarg_vis = self.get_next_val_safe(self._rmptarg_vis_opts, self._rmptarg_vis, self.binc[b])
        self._rmptarg_vis_btn.text = self._rmptarg_vis
        self._cur_scenario.realize_rmptarg_vis(self._rmptarg_vis)

    def _change_robskin_opt(self, x, y, b, m):
        print(f"_change_robskin_opt x:{x} y:{y} b:{b} m:{m}")
        if m==0:
            self._robskin_opt = self.get_next_val_safe(self._robskin_opts, self._robskin_opt, self.binc[b])
            self._robskin_opt_btn.text = self._robskin_opt
        else: # shft, or ctrl or alt are pressed
            self._cur_scenario.realize_robot_skin(self._robskin_opt)

    def _change_rmp_active(self, x, y, b, m):
        self._cur_scenario.rmpactive = not self._cur_scenario.rmpactive
        self._rmpactive_btn.text = "active" if self._cur_scenario.rmpactive  else "stopped"

    def _change_rotate(self, x, y, b, m):
        self._rotate_opt = self.get_next_val_safe(self._rotate_opts, self._rotate_opt, self.binc[b])
        self._rotate_opt_btn.text = self._rotate_opt
        self._cur_scenario.realize_rotate_opt(self._rotate_opt)

    def _change_ground_opt(self):
        self._ground_opt = self.get_next_val_safe(self._ground_opts, self._ground_opt)
        self._ground_btn.text = self._ground_opt

    def _exec_action(self, x, y, b, m):
        self._cur_scenario.scenario_action(self._action, b)

    def _setup_post_load(self):
        """
        This function is attached to the Load Button as the setup_post_load_fn callback.
        The user may assume that their assets have been loaded by their setup_scene_fn callback, that
        their objects are properly initialized, and that the timeline is paused on timestep 0.

        In this example, a scenario is initialized which will move each robot joint one at a time in a loop while moving the
        provided prim in a circle around the robot.
        """
        # print("ui_builder._setup_post_load")
        # self._reset_scenario() # we can't reset before post_load .... not sure what the intent was
        self._colprims = None


        self._cur_scenario.setup_scenario()
        self._cur_scenario.post_load_scenario()

        self._cur_scenario.reset_scenario() # should always be able to do a reset after post_load

        self._cur_scenario.realize_collider_vis_opt(self._collider_vis)
        self._cur_scenario.realize_eetarg_vis(self._eetarg_vis)
        self._cur_scenario.realize_rmptarg_vis(self._rmptarg_vis)

        # UI management
        self._scenario_state_btn.reset()
        self._scenario_state_btn.enabled = True
        self._reset_btn.enabled = True
        # print("ui_builder._setup_post_load almost done")
        self.build_ui_scenario_dependent()
        # print("ui_builder._setup_post_load done")


    def _reset_scenario(self):
        # print("ui_builder._reset_scenario")
        # self._ppc.reset()
        self._colprims = None
        self._cur_scenario.teardown_scenario()
        self._cur_scenario.setup_scenario()

        self._cur_scenario.reset_scenario()
        # print("ui_builder._reset_scenario done")


    def _on_post_reset_btn(self):
        """
        This function is attached to the Reset Button as the post_reset_fn callback.
        The user may assume that their objects are properly initialized, and that the timeline is paused on timestep 0.

        They may also assume that objects that were added to the World.Scene have been moved to their default positions.
        I.e. the cube prim will move back to the position it was in when it was created in self._setup_scene().
        """
        self._reset_scenario()

        # UI management
        self._scenario_state_btn.reset()
        self._scenario_state_btn.enabled = True



    def _update_scenario(self, step: float):
        """This function is attached to the Run Scenario StateButton.
        This function was passed in as the physics_callback_fn argument.
        This means that when the a_text "RUN" is pressed, a subscription is made to call this function on every physics step.
        When the b_text "STOP" is pressed, the physics callback is removed.

        Args:
            step (float): The dt of the current physics step
        """
        self._cur_scenario.update_scenario(step)
        self.refresh_open_robot_joint_values()

    def _on_run_scenario_a_text(self):
        """
        This function is attached to the Run Scenario StateButton.
        This function was passed in as the on_a_click_fn argument.
        It is called when the StateButton is clicked while saying a_text "RUN".

        This function simply plays the timeline, which means that physics steps will start happening.  After the world is loaded or reset,
        the timeline is paused, which means that no physics steps will occur until the user makes it play either programmatically or
        through the left-hand UI toolbar.
        """
        self._cur_scenario._running_scenario = True
        self._timeline.play()

    def _on_run_scenario_b_text(self):
        """
        This function is attached to the Run Scenario StateButton.
        This function was passed in as the on_b_click_fn argument.
        It is called when the StateButton is clicked while saying a_text "STOP"

        Pausing the timeline on b_text is not strictly necessary for this example to run.
        Clicking "STOP" will cancel the physics subscription that updates the scenario, which means that
        the robot will stop getting new commands and the cube will stop updating without needing to
        pause at all.  The reason that the timeline is paused here is to prevent the robot being carried
        forward by momentum for a few frames after the physics subscription is canceled.  Pausing here makes
        this example prettier, but if curious, the user should observe what happens when this line is removed.
        """
        self._cur_scenario._running_scenario = False
        self._timeline.pause()

    def _reset_extension(self):
        """This is called when the user opens a new stage from self.on_stage_event().
        All state should be reset.
        """
        self._on_init()
        self._reset_ui()

    def _reset_ui(self):
        self._scenario_state_btn.reset()
        self._scenario_state_btn.enabled = False
        self._reset_btn.enabled = False
