# icegame

## intro
Designed as an environments interacting with spin ice system.

## required
Only support python2.7 now.

* boost-python

## Move to next
Plan to be more flexible environment and get rid of redundancies.  

Quick version?
* Provide a configuration (1D)
* Actions = change of configuration
* Target: Global updates

## Compile & Install 

### compile

```
sh compile.sh
```

will generate icegame.so in build/src. please move icegame.so to where you executing experiment

```
mv build/src/icegame.so /where/you/run/icegame
```
or add lib path in your python code
```
import sys
sys.path.append("path to libicegame.so")
```

### gym-icegame

Install python interface 

```
python setup.py install
```

which depends on openai gym.

## Quick Start
One can run
```
python random_bot.py
```
to see how this icegame works.


## Game Scenario
Draw a proposed loop, then summit.

### actions
* Directional action: (up, down, left, right, next up, next down) 6 operations in total

Two options: 
* Metropolis button: 
* Loop Auto-detection (Do we need this?)

function called by env.step(action_index)
* step()

### rewards

* Step-wise reward
```
    r = scale * (icemove_w * rets[0] + energy_w * rets[1] + defect_w * rets[2] + baseline)
```
* Accepted reward
```
    r = +1.0 * (loop_length / 4.0)
```

```
rets[0]: +1/-1 accpet/reject
rets[1]: difference of totoal mean energy 
rets[2]: difference of defect density
rets[3]: ratio of configuration difference
```

### observations
Stacked scene
* spin configuration
* trajectory (canvas)
* energy map
* defect map
in format of `NHWC`.  

Values in map:
* Configuration: +1 for up; -1 for spin down (`get_state_t_map`)
* Configuration: A sublatt (+1/-1 for up/down); B sublatt (+0.5/-0.5 for up/down) (`get_state_t_map_color`)
* Canvas: +1 for occupied A sublattice, -1 for B sublatt. 0 is empty site.
* Energy: Nearest neighbor local Hamiltonian. (s_i * s_j / # of neighbors)

## Callable interface from libicegame
List in sqice.hpp

## TODO Lists
* Consider `terminate` condition. Does game exit after execute metropolis? Or terminate by timeout? (Keep playing if update successfully)
* Better visulization tools.
* Support Ising model
