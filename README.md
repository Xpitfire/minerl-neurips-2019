# MineRL NeurIPS 2019

NeurIPS 2019 : MineRL Competition
The MineRL Competition for Sample-Efficient Reinforcement Learning
https://www.aicrowd.com/challenges/neurips-2019-minerl-competition#abstract

## How to contribute
Open tasks are listed as open issues in the [todo section](https://git.bioinf.jku.at/minerl/minerl-neurips-2019/boards)
of our gitlab.
The ```module:``` label of the issue will tell you the supervisor for an issue.

The project is divided into the following subprojects:
- Subtask identification
- Subtask scheduler
- Behavioral cloning
- Reward redistribution
- RL methods

There are branches for each of the subprojects
[subtask_identification](https://git.bioinf.jku.at/minerl/minerl-neurips-2019/tree/subtask_identification), 
[subtask_scheduler](https://git.bioinf.jku.at/minerl/minerl-neurips-2019/tree/subtask_scheduler), 
[behavioral_cloning](https://git.bioinf.jku.at/minerl/minerl-neurips-2019/tree/behavioral_cloning), 
[reward_redistribution](https://git.bioinf.jku.at/minerl/minerl-neurips-2019/tree/reward_redistribution), 
[rl_methods](https://git.bioinf.jku.at/minerl/minerl-neurips-2019/tree/rl_methods),
as well as a [master](https://git.bioinf.jku.at/minerl/minerl-neurips-2019/tree/master) branch.

If you want to work on an issue, please contact the supervisor to be assigned to the issue.
If you are assigned, please create a new branch from the respective subproject branch with name 
```last_name/issue_number``` and implement your solution in this branch.
Once you are done and your code is tested and tidy, please create a 
[merge request](https://git.bioinf.jku.at/minerl/minerl-neurips-2019/merge_requests/new) to the subproject branch.
Title of the merge request should be ```last_name/issue_number``` and target branch should be the branch associated
with the ```module:``` label of the issue.

#### Example
In [todo section](https://git.bioinf.jku.at/minerl/minerl-neurips-2019/boards) you saw that
[issue #5](https://git.bioinf.jku.at/minerl/minerl-neurips-2019/issues/5) is in the todo list and you want to work on 
it.
You would first contact the supervisor that is stated in the ```module: master``` label and would be assigned to the 
issue.
You would then create a branch ```last_name/5``` from branch 
[master](https://git.bioinf.jku.at/minerl/minerl-neurips-2019/tree/master) since it is labeled ```module: master```.
You would then implement your code in your branch and create a 
[merge request](https://git.bioinf.jku.at/minerl/minerl-neurips-2019/merge_requests/new) once you are done and the code 
is tested and tidy.
Title of the merge request would be ```last_name/5``` and target branch would be ```master``` since the issue label
stated ```module: master```.
The supervisor will then accept your pull request or suggest revisions.
Done! :)


## Installation

```bash
pip install git+https://git.bioinf.jku.at/minerl/minerl-neurips-2019
```

## Development

```bash
git clone https://git.bioinf.jku.at/minerl/minerl-neurips-2019
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

### Subtask identification
IMPORTANT: [The current dataset is wrong](https://github.com/minerllabs/minerl/issues/41), 
so e.g. the item data doesn't make sense at all. Please take care.
- [make_lookup_dict_from_raw()](minerl_neurips_2019/subtask_identifier/subtask_identifier.py):
Identify subtasks and create lookup dictionary in file "subtask_lookup.json"
- [load_subtask_lookup()](minerl_neurips_2019/subtask_identifier/subtask_identifier.py):
Loads "subtask_lookup.json"


### Subtask scheduler

### Behavioral cloning

### Reward redistribution

### RL methods
