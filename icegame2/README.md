# icegame2

## intro
New design of the icegame!

### What's New?

* Support long loop algorithm.
* Can do double visiting.

### TODOs

* states and observations.

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

## Game Scenario
Draw a proposed loop, then summit.

### actions

### rewards

### observations

## Callable interface from libicegame
List in sqice.hpp

## TODO Lists
