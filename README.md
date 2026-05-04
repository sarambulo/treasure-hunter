# Tiny Treasure-Hunter

## Project goal

Teach the VEX AIM robot how to find objects, even when hidden behind walls

## Setup

1. Install base dependencies

```bash
pip install -r requirements.txt
```

2. Install Vex AIM Tools (follow this [instructions](https://www.cs.cmu.edu/afs/cs.cmu.edu/academic/class/15494-s26/install.html))

3. Create file `.env` and store WandB API key in it

```
# .env
WANDB_API_KEY=<your api key>
```

## Main commands

```bash
# train agent
python train.py --config configs/train.yaml --device cuda

# display train CLI arguments
python train.py -h

# run agent on VEX AIM Robot
python <path to simple_cli>/simple_cli Explore
# once the program starts, send a TM (empty text message) to
# the robot to start looking for barrels
```

## FAQ

- `simple_cli` cannot find a dependency that you are sure you have already installed: It is important that you use
`python <path to simple_cli>/simple_cli Explore` and not `simple_cli Explore`. The former uses the python binaries for 
the current virtual environment, while the later uses the global python binaries.

- When running Explore, the world map, particle filter and robot camera vieweres do not show up: Hit Enter on the console
running `simple_cli`.

- `Explore` is running but the robot is not moving: Send a TM message to make the robot start looking for barrels

- The robot is looking for barrels but the 2D grid showing the explored world-map does not show up: Hit Enter on the console
running `simple_cli` once the robot has started moving