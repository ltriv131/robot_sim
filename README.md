# robot_sim

A 2D sandbox for motion-planning algorithms from *Principles of Robot Motion* (Choset et al.). It currently implements **Bug1** and **Tangent Bug**, and I add other algorithms from the book as I find them interesting enough to implement.

The simulation models a point robot moving among polygonal obstacles. It has a range of sensors, from a simple contact sensor to a 360° LIDAR-style ring of distance sensors.

## Getting started

Requires Python 3.10+.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1      # macOS/Linux: source .venv/bin/activate
pip install -e .
```

Run a scenario from the project root:

```powershell
python -m robot_sim            # default scenario: spiral
python -m robot_sim cup
```

When the window opens, **click anywhere on the map to set the robot's start position**. The robot then plans its way to the goal, which is marked with a gold star.

### Scenarios

| Name      | What it tests |
|-----------|---------------|
| `glide`   | Small convex shapes on the start–goal line. The robot should slip around them without switching to boundary following. |
| `trap`    | Concave and wide shapes that make the heuristic rise, forcing boundary following. |
| `cup`     | The goal sits inside a C that opens away from the robot, behind a comb and a zigzag wall. |
| `spiral`  | The goal is at the centre of a spiral, so the robot has to wind its way in. |
| `clutter` | A dense field of mixed shapes, including a narrow gap and several concave traps. |

Scenarios are defined in `SCENARIOS` in [`src/robot_sim/__main__.py`](src/robot_sim/__main__.py).

## Algorithms

### Bug1

When the robot hits an obstacle that blocks the goal, it circumnavigates the entire obstacle and remembers the boundary point closest to the goal. It then keeps following the boundary back to that point and heads for the goal from there. Bug1 only needs a contact sensor.

### Tangent Bug

Tangent Bug assumes a 360° array of distance sensors with a limited range. It uses that range data to take shortcuts that Bug1 can't.

**Detecting obstacles.** The sensor ring gives a range reading at each angle. Obstacle endpoints, called *O<sub>i</sub>*, are the angles where that range-versus-angle reading jumps: either where an obstacle edge drops out of sensor range, or where one surface gives way to another surface further back. The span between two endpoints is one visible obstacle.

**Motion to goal.**
- If nothing lies between the robot and the goal, the robot drives straight at it. An obstacle only counts as blocking if the ray toward the goal hits it *before* reaching the goal. A wall behind the goal does not block.
- If an obstacle blocks the way, the robot heads toward whichever endpoint minimises the heuristic distance d(x, O<sub>i</sub>) + d(O<sub>i</sub>, goal). The aim point is pushed sideways off the endpoint by the follow distance, so the robot clears the corner instead of grazing it.
- This continues for as long as the heuristic keeps decreasing.

**Boundary following.**
- When the heuristic stops decreasing, the robot is in a local minimum and switches to following the blocking obstacle's boundary. It keeps going in the same direction it was already heading.
- At the switch it records *d<sub>followed</sub>*, the closest distance to the goal among the sensed points on that obstacle. This point is drawn in purple as the "m point".
- While following, it holds a fixed distance from the closest sensed point on the wall and moves along the wall's tangent.

**Leaving the boundary.** The robot returns to motion to goal once some visible point is closer to the goal than *d<sub>followed</sub>*, i.e. *d<sub>reach</sub>* < *d<sub>followed</sub>*. This exit is drawn in olive as the "leave point".

## Visualization

| Marker | Meaning |
|---|---|
| Blue dot | Robot |
| Grey dots | Robot's trail |
| Gold star | Goal |
| Cyan / red lines | Range rays that hit an obstacle / rays that found nothing in range |
| Green / orange dots | Tangent Bug aim points: the endpoint being followed, and the other endpoint |
| Purple dot | m point (d<sub>followed</sub>) |
| Olive dot | Leave point |

## Configuration

The robot, sensors and planner are set up in `main()` in [`src/robot_sim/__main__.py`](src/robot_sim/__main__.py).
- **Choosing the planner:** `Simulation` uses `planner_tangent`. Pass `planner` instead to run Bug1.
- **Planner tuning:** the main Tangent Bug parameters are set in its constructor in [`tangent_bug.py`](src/robot_sim/planner/tangent_bug.py):

| Parameter | Default | Purpose |
|---|---|---|
| `goal_tolerance` | 0.2 | Distance at which the goal counts as reached |
| `max_range` | 4.0 | Sensor range used for endpoint detection (should match the sensor) |
| `jump_threshold` | 0.5 | Range jump that counts as an endpoint |
| `follow_distance` | 0.7 | Distance kept from walls while following, and the endpoint offset |

## Project layout

```
src/robot_sim/
├── __main__.py        Entry point and scenario definitions
├── simulation.py      Main loop: sense → plan → move → draw
├── robot.py           Point robot kinematics
├── world.py           World bounds and obstacles
├── observation.py     Sensor measurement types
├── visualizer.py      Matplotlib rendering
├── obstacles/         Polygon obstacles (ray intersection, tangents)
├── sensors/           Contact, position, and radial distance sensors
└── planner/
    ├── planner.py     Planner interface and Bug1
    └── tangent_bug.py Tangent Bug
```
