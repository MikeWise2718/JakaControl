# This software contains source code provided by NVIDIA Corporation.
# Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

import numpy as np
import math
import omni.timeline
import omni.ui as ui
from omni.isaac.core.articulations import Articulation
from omni.isaac.core.objects.cuboid import FixedCuboid
from omni.isaac.core.prims import XFormPrim
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.prims import is_prim_path_valid
from omni.isaac.core.utils.stage import add_reference_to_stage, create_new_stage, get_current_stage
from omni.isaac.core.world import World
from omni.ui import Button
from omni.ui import color as uiclr
from omni.isaac.ui.element_wrappers import CollapsableFrame, StateButton
from omni.isaac.ui.element_wrappers.core_connectors import LoadButton, ResetButton
from omni.isaac.ui.ui_utils import get_style
from omni.usd import StageEventType
from pxr import Sdf, UsdLux, UsdPhysics, Usd

from .scenario import ExampleScenario

import carb.settings


_settings = None


def _init_settings():
    global _settings
    if _settings is None:
        _settings = carb.settings.get_settings()
    return _settings


def get_setting(name, default, db=False):
    settings = _init_settings()
    key = f"/persistent/omni/sphereflake/{name}"
    val = settings.get(key)
    if db:
        oval = val
        if oval is None:
            oval = "None"
    if val is None:
        val = default
    if db:
        print(f"get_setting {name} {oval} {val}")
    return val


def save_setting(name, value):
    settings = _init_settings()
    key = f"/persistent/omni/sphereflake/{name}"
    settings.set(key, value)


def truncf(number, digits) -> float:
    # Improve accuracy with floating point operations, to avoid truncate(16.4, 2) = 16.39 or truncate(-1.13, 2) = -1.12
    nbDecimals = len(str(number).split('.')[1])
    if nbDecimals <= digits:
        return number
    stepper = 10.0 ** digits
    return math.trunc(stepper * number) / stepper



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
    _robot_names = ["ur10e", "jaka", "franka"]
    _robot_name = "jaka"

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

        except Exception as e:
            carb.log_error(f"Exception in SaveSettings: {e}")

    def LoadSettings(self):
        print("LoadSettings")
        self._robot_name = get_setting("p_robot_name", self._robot_name)

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
        world_config_frame = CollapsableFrame("World Config", collapsed=False)

        with world_config_frame:
            with ui.VStack(style=get_style(), spacing=5, height=0):
                with ui.HStack(style=get_style(), spacing=5, height=0):
                    ui.Label("Robot: ",
                            style={'color': self.btyellow},
                            width=50)
                    self._robot_btn = Button(
                        self._robot_name, clicked_fn=self._change_robot_name,
                        style={'background_color': self.dkgreen},
                    )
                # self.wrapped_ui_elements.append(self._robot_btn)



        world_controls_frame = CollapsableFrame("World Controls", collapsed=False)

        with world_controls_frame:
            with ui.VStack(style=get_style(), spacing=5, height=0):
                self._load_btn = LoadButton(
                    "Load/Create Scene", "Create", setup_scene_fn=self._setup_scene_gen, setup_post_load_fn=self._setup_scenario
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

    def _on_init(self):
        self._articulation = None
        self._cuboid = None
        self._scenario = ExampleScenario()
        self.LoadSettings()

    def _add_light_to_stage(self):
        """
        A new stage does not have a light by default.  This function creates a spherical light
        """
        sphereLight = UsdLux.SphereLight.Define(get_current_stage(), Sdf.Path("/World/SphereLight"))
        sphereLight.CreateRadiusAttr(2)
        sphereLight.CreateIntensityAttr(100000)
        XFormPrim(str(sphereLight.GetPath())).set_world_pose([6.5, 0, 12])

    # def _setup_scene(self):
    #     """
    #     This function is attached to the Load Button as the setup_scene_fn callback.
    #     On pressing the Load Button, a new instance of World() is created and then this function is called.
    #     The user should now load their assets onto the stage and add them to the World Scene.

    #     In this example, a new stage is loaded explicitly, and all assets are reloaded.
    #     If the user is relying on hot-reloading and does not want to reload assets every time,
    #     they may perform a check here to see if their desired assets are already on the stage,
    #     and avoid loading anything if they are.  In this case, the user would still need to add
    #     their assets to the World (which has low overhead).  See commented code section in this function.
    #     """
    #     # Load the UR10e
    #     robot_prim_path = "/ur10e"
    #     path_to_robot_usd = get_assets_root_path() + "/Isaac/Robots/UniversalRobots/ur10e/ur10e.usd"
    #     print("Loading ur10e")

    #     create_new_stage()
    #     self._add_light_to_stage()
    #     add_reference_to_stage(path_to_robot_usd, robot_prim_path)

    #     # Create a cuboid
    #     self._cuboid = FixedCuboid(
    #         "/Scenario/cuboid", position=np.array([0.3, 0.3, 0.5]), size=0.05, color=np.array([255, 0, 0])
    #     )

    #     self._articulation = Articulation(robot_prim_path)

    #     # Add user-loaded objects to the World
    #     world = World.instance()
    #     world.scene.add(self._articulation)
    #     world.scene.add(self._cuboid)

    # def _setup_scene_jaka(self):
    #     """
    #     This function is attached to the Load Button as the setup_scene_fn callback.
    #     On pressing the Load Button, a new instance of World() is created and then this function is called.
    #     The user should now load their assets onto the stage and add them to the World Scene.

    #     In this example, a new stage is loaded explicitly, and all assets are reloaded.
    #     If the user is relying on hot-reloading and does not want to reload assets every time,
    #     they may perform a check here to see if their desired assets are already on the stage,
    #     and avoid loading anything if they are.  In this case, the user would still need to add
    #     their assets to the World (which has low overhead).  See commented code section in this function.
    #     """

    #     robot_prim_path = "/World/minicobo_v1_4/world"
    #     path_to_robot_usd = "d:/nv/ov/exts/JakaControl/usd/jaka1_blue.usda"
    #     print("Loading Jaka ")


    #     create_new_stage()
    #     self._add_light_to_stage()
    #     add_reference_to_stage(path_to_robot_usd, robot_prim_path)

    #     # Create a cuboid
    #     self._cuboid = FixedCuboid(
    #         "/Scenario/cuboid", position=np.array([0.3, 0.3, 0.5]), size=0.05, color=np.array([255, 0, 0])
    #     )
    #     prim = get_current_stage().GetPrimAtPath(robot_prim_path)
    #     UsdPhysics.ArticulationRootAPI.Apply(prim)

    #     self._articulation = Articulation(robot_prim_path)
    #     print("Jaka Articulation: ", self._articulation)

    #     # Add user-loaded objects to the World
    #     world = World.instance()
    #     world.scene.add(self._articulation)
    #     print("Jaka added articulation to world")
    #     world.scene.add(self._cuboid)

    def _setup_scene_gen(self):

        match self._robot_name:
            case "ur10e":
                robot_prim_path = "/ur10e"
                path_to_robot_usd = get_assets_root_path() + "/Isaac/Robots/UniversalRobots/ur10e/ur10e.usd"
            case "jaka":
                robot_prim_path = "/World/minicobo_v1_4/world"
                path_to_robot_usd = "d:/nv/ov/exts/JakaControl/usd/jaka1_blue.usda"
            case "franka":
                robot_prim_path = "/ur10e"
                path_to_robot_usd = get_assets_root_path() + "/Isaac/Robots/UniversalRobots/ur10e/ur10e.usd"
            case _:
                self.error(f"Unknown robot name {self._robot_name}")


        print(f"Loading {self._robot_name}")


        create_new_stage()
        self._add_light_to_stage()
        add_reference_to_stage(path_to_robot_usd, robot_prim_path)

        # Create a cuboid
        self._cuboid = FixedCuboid(
            "/Scenario/cuboid", position=np.array([0.3, 0.3, 0.5]), size=0.05, color=np.array([255, 0, 0])
        )
        if self._robot_name == "jaka":
            prim = get_current_stage().GetPrimAtPath(robot_prim_path)
            UsdPhysics.ArticulationRootAPI.Apply(prim)

        self._articulation = Articulation(robot_prim_path)

        # Add user-loaded objects to the World
        world = World.instance()
        world.scene.add(self._articulation)
        world.scene.add(self._cuboid)


    def _change_robot_name(self):
        idx = self._robot_names.index(self._robot_name)
        idx = (idx + 1) % len(self._robot_names)
        self._robot_name = self._robot_names[idx]
        self._robot_btn.text = self._robot_name

    def _setup_scenario(self):
        """
        This function is attached to the Load Button as the setup_post_load_fn callback.
        The user may assume that their assets have been loaded by their setup_scene_fn callback, that
        their objects are properly initialized, and that the timeline is paused on timestep 0.

        In this example, a scenario is initialized which will move each robot joint one at a time in a loop while moving the
        provided prim in a circle around the robot.
        """
        self._reset_scenario()

        # UI management
        self._scenario_state_btn.reset()
        self._scenario_state_btn.enabled = True
        self._reset_btn.enabled = True

    def _reset_scenario(self):
        self._scenario.teardown_scenario()
        self._scenario.setup_scenario(self._articulation, self._cuboid)

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
        self._scenario.update_scenario(step)

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
