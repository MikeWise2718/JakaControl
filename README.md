# Jaka Control
Mike Wise - 21 Feb 2024

While the modivating goal is to create a control program for the JAKA robot using Isaac Sim,
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
