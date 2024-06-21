# Jaka Control

Mike Wise - 21 Feb 2024

While the motivating goal is to create a control program for the JAKA robot using Isaac Sim,
this repo is actually more of a platform for investigating various Isaac Sim approaches.
Isaac Sim is a rich platform with many options, so a robot can be controled in a variety of
fashions and it is not clear to us which one is best for our task at hand.

This extension is derived from an Issac Sim template extension (remember to ask Drew which one).

# Structure

This is the root directory of an Omniverse extension. There are several subdirectories:

- `config` - subdir contains the `extension.toml` that has various extension configuration settings
- `data` - subdir containing the extension icon and a more descriptive screen
- `JakaCtrl` - subdir containing the actual python code
- `usd` - subdir conttaining some usd files that the extension needs (maybe these should be in JakaCtrl?)

# Gett Started

## Clone Extension Repos

Clone all the repos to the same directory

- https://github.com/MikeWise2718/omni.asimov.manipulator
- https://github.com/MikeWise2718/omni.asimov.jaka
- https://github.com/MikeWise2718/JakaControl

## Open Isaac Sim 4.x

## Open Extensions Window

- Select "Window" > Click "Extensions"

## Add the directory where you cloned the repos to NVidia Extensions Search Paths

- Near the top, Select the Hamburger Menu (3 lines) -> Click "Settings"
- In "Extension Search Paths", add an entry using directory where you cloned the repos above
    - E.g. `/home/mattm/repos`

Note: The above steps are shown [visually here](https://docs.omniverse.nvidia.com/workflows/latest/extensions/ui_window_tutorial.html#step-2-1-navigate-to-the-extensions-list)

## View "Third-Party" extension

![img](https://github.com/MikeWise2718/JakaControl/assets/2856501/24d7f8f7-e3fd-42b4-acb8-669d631b28cf)

- Verify all 3 extensions are listed
  - Asimov Jaka Minicobo Robot
  - Asimov Manipulators
  - Jaka Control
- Enable all 3 extensions
  - Set the toggle switch to right of extension to "enabled"

![img](https://github.com/MikeWise2718/JakaControl/assets/2856501/e590cffb-412a-4bae-8766-580eda086adb)

## Close Extensions Windows

## Open Jaka Control Extension

When you enabled the JakaControl extension a new menu should have appeared called "Jaka"

Select "Jaka" -> Click "Jaka"