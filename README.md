# MineRL NeurIPS 2019

NeurIPS 2019 : MineRL Competition
The MineRL Competition for Sample-Efficient Reinforcement Learning
https://www.aicrowd.com/challenges/neurips-2019-minerl-competition#abstract
Competition repository: https://github.com/minerllabs/minerl

## Installation

```bash
pip install git+https://github.com/Xpitfire/minerl-neurips-2019
```

## Development

```bash
git clone https://github.com/Xpitfire/minerl-neurips-2019
cd minerl-neurips-2019

# pip install with source-map
pip install -e .
```

## Important functions and classes
Please provide an overview over the important functions/classes here,
such that others can get a quick idea about what already exists and where to find what. 

### General data handling
- [get_frames()](minerl_neurips_2019/utils/data_handling.py): 
Extract subsequence of frames from a mp4 video as numpy array.
- [show_frames()](minerl_neurips_2019/utils/data_handling.py): 
Show frames in numpy array as video
- [SpacesMapper()](minerl_neurips_2019/utils/data_handling.py): 
Provides mapping functions for translating inventory-, equipment-, and action space to other spaces or item keys. 
See http://minerl.io/docs/environments/index.html#minerlobtaindiamond-v0 for more information on spaces.
