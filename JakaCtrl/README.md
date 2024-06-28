# Jaka Control
Mike Wise - 21 Feb 2024

While the motivating goal is to create a control program for the JAKA robot using Isaac Sim,
this repo is actually more of a platform for investigating various Isaac Sim approaches.
Isaac Sim is a rich platform with many options, so a robot can be controled in a variety of
fashions and it is not clear to us which one is best for our task at hand.

This extension is derived from an Issac Sim template extension (remember to ask Drew which one).
However it was refactored to a degree since the original approach distributed logic across both
ui_builder.py and scenario.py, preventing its easy use as a multi-scenario platform.
It original form is documented in "README-orig.md".

Now the ui_builder only has "ui" code in it, and it can be used to launch and control multiple-scenarios,
at the moment

# Notes
THe RMP motion policy configuration can be found here in the following directory.
```
c:\users\mike\appdata\local\ov\pkg\isaac_sim-2023.1.1\exts\omni.isaac.motion_generation\motion_policy_configs
```

# Loading Extension
To enable this extension, run Isaac Sim with the flags --ext-folder {path_to_ext_folder} --enable {ext_directory_name}
The user will see the extension appear on the toolbar on startup with the title they specified in the Extension Generator


# Extension Usage
This template extension creates a Load, Reset, and Run button in a simple UI.
The Load and Reset buttons interact with the omni.isaac.core World() in order
to simplify user interaction with the simulator and provide certain gurantees to the user
at the times their callback functions are called.


# Code Overview
The template is well documented and is meant to be self-explanatory to the user should they
start reading the provided python files.  A short overview is also provided here - if someone adds new files
they should add them to this listing.

extension.py:
    A class containing the standard boilerplate necessary to have the user extension show up on the Toolbar.  This
    class is meant to fulfill most ues-cases without modification.
    In extension.py, useful standard callback functions are created that the user may complete in ui_builder.py.

ui_builder.py:
    This file is the user's main entrypoint into the template.  Here, the user can see useful callback functions that have been
    set up for them, and they may also create UI buttons that are hooked up to more user-defined callback functions. Isaac Sim logic should
    not be contained here (as it was in the original).

global_variables.py:
    A script that stores in global variables that the user specified when creating this extension such as the Title and Description.
    There is like nothing in here...

senut.py:
    Place where we chuck utilities that don't use class varibles and can stand alone.

sinusoid_scenario:
    This scenario is the original example, modified a bit. It now starts from the robot zero position (instead of the min joint),
    and uses a constant time interval to switch between joint movements.

pickplace_scenario:
   This scenario is the pick and place scenario from the Issac Sim tutorial examples. It should work on any robot with a gripper.

rmp_scenario:
    This scenario is for testing rmp policy. It works with pretty much any robot that has the appropriate definitions:
       - URDF
       - Robot Definition File (for Lula)
       - motion_control_policy file (for RMP)
