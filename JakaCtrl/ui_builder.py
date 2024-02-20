# This software contains source code provided by NVIDIA Corporation.
# Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

import omni.timeline
import omni.ui as ui
from omni.isaac.core.utils.stage import create_new_stage
from omni.ui import Button
from omni.ui import color as uiclr
from omni.isaac.ui.element_wrappers import CollapsableFrame, StateButton
from omni.isaac.ui.element_wrappers.core_connectors import LoadButton, ResetButton
from omni.isaac.ui.ui_utils import get_style
from omni.usd import StageEventType

from .RMPscenario import RMPflowScenario
from .PickAndPlaceScenario import PickAndPlaceScenario
from .SinusoidScenario import SinusoidJointScenario

from .senut import get_setting, save_setting


class UIBuilder:
    btgreen = uiclr("#00ff00")
    btblue = uiclr("#0000ff")
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
    _scenario_names = ["sinusoid-joint", "pick-and-place", "rmpflow"]
    _scenario_name = "sinusoid-joint"
    _robot_names = ["ur3e", "ur5e", "ur10e", "jaka-minicobo", "rs007n", "franka", "fancy_franka", "jetbot"]
    _robot_name = "jaka-minicobo"
    _ground_opts = ["none", "default", "groundplane", "groundplane-blue"]
    _ground_opt = "default"
    _modes = ["mode1","mode2"]
    _mode = "mode1"
    _choices = ["choice 1","choice 2"]
    _choice = "choice 1"

    def __init__(self):
        # Frames are sub-windows that can contain multiple UI elements
        self.frames = []
        # UI elements created using a UIElementWrapper instance
        self.wrapped_ui_elements = []

        # Get access to the timeline to control stop/pause/play programmatically
        self._timeline = omni.timeline.get_timeline_interface()

        # Run initialization for the provided example
        self._on_init()

    def SaveSettings(self):
        print("SaveSettings")
        try:
            save_setting("p_robot_name", self._robot_name)
            save_setting("p_ground_opt", self._ground_opt)
            save_setting("p_scenario_name", self._scenario_name)

        except Exception as e:
            carb.log_error(f"Exception in SaveSettings: {e}")

    def LoadSettings(self):
        print("LoadSettings")
        self._robot_name = get_setting("p_robot_name", self._robot_name)
        self._ground_opt = get_setting("p_ground_opt", self._ground_opt)
        self._scenario_name = get_setting("p_scenario_name", self._scenario_name)
        print("Done LoadSettings")

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
        print("build_ui")
        world_config_frame = CollapsableFrame("World Config", collapsed=False)

        with world_config_frame:
            with ui.VStack(style=get_style(), spacing=5, height=0):
                with ui.HStack(style=get_style(), spacing=5, height=0):
                    ui.Label("Scenario Name:",
                            style={'color': self.btyellow},
                            width=50)
                    self._scenario_name_btn = Button(
                        self._scenario_name, clicked_fn=self._change_scenario_name,
                        style={'background_color': self.dkblue}
                    )
                with ui.HStack(style=get_style(), spacing=5, height=0):
                    ui.Label("Robot:",
                            style={'color': self.btyellow},
                            width=50)
                    self._robot_btn = Button(
                        self._robot_name, clicked_fn=self._change_robot_name,
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
                # self.wrapped_ui_elements.append(self._robot_btn)

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

    ######################################################################################
    # Functions Below This Point Support The Provided Example And Can Be Deleted/Replaced
    ######################################################################################

    def pick_scenario(self, scenario_name):
        if scenario_name == "sinusoid-joint":
            self._cur_scenario = SinusoidJointScenario()
        elif scenario_name == "pick-and-place":
            self._cur_scenario = PickAndPlaceScenario()
        elif scenario_name == "rmpflow":
            self._cur_scenario = RMPflowScenario()
        else:
            raise ValueError(f"Unknown scenario name {scenario_name}")

    def _on_init(self):
        # self._articulation = None
        # self._cuboid = None
        self.LoadSettings()
        self.pick_scenario(self._scenario_name)
        print("Done _on_init")



    def _setup_scene(self):
        print("ui_builder._setup_scene")
        self.pick_scenario(self._scenario_name)

        create_new_stage()

        self._cur_scenario.load_scenario(self._robot_name, self._ground_opt)

    def get_next_val_safe(self, lst, val):
        try:
            idx = lst.index(val)
            idx = (idx + 1) % len(lst)
            rv = lst[idx]
        except:
            idx = 0
            rv = lst[idx]
        return rv

    def _change_choice(self):
        self._choice = self.get_next_val_safe(self._choices, self._choice)
        self._choice_btn.text = self._choice

    def _change_mode(self):
        self._mode = self.get_next_val_safe(self._modes, self._mode)
        self._mode_btn.text = self._mode

    def _change_robot_name(self):
        self._robot_name = self.get_next_val_safe(self._robot_names, self._robot_name)
        self._robot_btn.text = self._robot_name

    def _change_scenario_name(self):
        self._scenario_name = self.get_next_val_safe(self._scenario_names, self._scenario_name)
        self._scenario_name_btn.text = self._scenario_name

    def _change_ground_opt(self):
        self._ground_opt = self.get_next_val_safe(self._ground_opts, self._ground_opt)
        self._ground_btn.text = self._ground_opt

    def _setup_post_load(self):
        """
        This function is attached to the Load Button as the setup_post_load_fn callback.
        The user may assume that their assets have been loaded by their setup_scene_fn callback, that
        their objects are properly initialized, and that the timeline is paused on timestep 0.

        In this example, a scenario is initialized which will move each robot joint one at a time in a loop while moving the
        provided prim in a circle around the robot.
        """
        print("ui_builder._setup_post_load")
        self._reset_scenario()

        self._cur_scenario.post_load_scenario()

        # UI management
        self._scenario_state_btn.reset()
        self._scenario_state_btn.enabled = True
        self._reset_btn.enabled = True

    def _reset_scenario(self):
        print("ui_builder._reset_scenario")
        self._cur_scenario.teardown_scenario()
        self._cur_scenario.setup_scenario()

        self._cur_scenario.reset_scenario()


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

    def _on_run_scenario_a_text(self):
        """
        This function is attached to the Run Scenario StateButton.
        This function was passed in as the on_a_click_fn argument.
        It is called when the StateButton is clicked while saying a_text "RUN".

        This function simply plays the timeline, which means that physics steps will start happening.  After the world is loaded or reset,
        the timeline is paused, which means that no physics steps will occur until the user makes it play either programmatically or
        through the left-hand UI toolbar.
        """
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
