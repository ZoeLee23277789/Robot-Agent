# RoboMaster LLM Agent

An LLM as the high-level controller of a DJI RoboMaster EP, in CoppeliaSim and on the real robot. The agent sees through a front camera, an overhead camera and telemetry, and acts through the chassis, the arm and the gripper, plus one-call skills for navigation and grasping.

The observe → think → act loop (`RobotAgent.py` → `robot_agent/service.py`) is written for this project and does not depend on an agent framework. `robot_eval/` is a milestone-based evaluation suite with 100 tasks.

## Architecture

The RoboMaster SDK only runs on Python 3.6–3.8 and the LLM layer needs 3.11+, so the system is two processes talking over HTTP. The same split works when the robot is on another machine.

```
Agent side (.venv, Python 3.11+)                Robot side (~/rmenv, Python 3.6–3.8)
RobotAgent.py          entry point              Test_Robot/robot_server.py   HTTP server:
robot_agent/                                      limits, action lock, odometry checks,
  service.py           main loop, planner,        collision guards, /live dashboard
                       summary, self-check      Test_Robot/nav_map.py        occupancy grid + A*
  actions.py           action registry          Test_Robot/nexarm.py         NexArm driver
  skills.py            go_to, approach, pick,   Test_Robot/rm_connection.py  SDK connection
                       place, drive_through
  skills_nexarm.py     real-robot pick/place          │
  perception.py        visual pointing                ├─ RoboMaster SDK ─▶ CoppeliaSim or real EP
  llm/                 OpenAI / Gemini wrappers       └─ USB / Wi-Fi ────▶ NexArm
  client.py  ───── HTTP (JSON + JPEG) ─────▶
```

Each step the agent reads telemetry and both camera images, asks the LLM for a structured answer (reasoning, evaluation of the last step, memory, next goal, 1–3 actions), runs the actions, and records the result. The LLM chooses what to do; distances, paths and arm poses inside skills are computed from the map and the sensors, and every skill verifies its own result (odometry, distance sensor, gripper force or servo feedback) instead of trusting the picture.

## Setup

1. Agent side: `pip install -r requirements.txt`, then copy `.env.example` to `.env` and fill in the API key.
2. Robot side: a Python 3.6–3.8 environment at `~/rmenv` with `robomaster` and `opencv-python`.
3. CoppeliaSim with the RoboMaster plugin, for simulation.

## Quick start (simulation)

```bash
./start_all.sh                 # start the simulation and robot_server.py (port 8765)
./start_all.sh --restart       # after editing robot_server.py
./run_agent.sh --task "Find the red box and stop 30 cm in front of it"
./run_agent.sh --loop          # interactive: keep giving tasks
```

Open `http://127.0.0.1:8765/live` to watch the map, both cameras, telemetry and the agent's reasoning live.

No simulator? `~/rmenv/bin/python -u Test_Robot/robot_server.py --mode mock` runs a fake 2D robot for testing the loop.

Providers: `--provider openai | google | robotics-er | mlx` (default set by `ROBOT_LLM_PROVIDER`).

## Modes

| Mode | Eyes and map | Start with |
| --- | --- | --- |
| Simulation | CoppeliaSim | `start_all.sh`, `run_agent.sh` |
| Mirror | Simulation holds the map; every action is forwarded to the real robot (port 8767), which supplies the camera and distance sensor | `start_office.sh`, `run_mirror.sh` |
| Explore | The real robot builds its own map from its distance sensor, odometry and camera | `run_explore.sh` |
| Arm only | A NexArm on the desk, no chassis | `start_arm.sh` |

On the real robot the agent asks for confirmation before every physical step unless you pass `--no-confirm`.

## Actions

25 actions, registered in `robot_agent/actions.py`:

- **Primitives**: `move_chassis`, `move_arm`, `arm_to`, `recenter_arm`, `lift_arm`, `gripper`, `wait`, `stop`
- **Perception**: `locate`, `face`, `locate_overhead`, `scan_overhead`, `align_to_tunnel`, `survey_scene`
- **Skills**: `go_to`, `approach`, `pick`, `place`, `pick_from_top`, `place_on_top`, `drive_through`, `return_to_origin`
- **Other**: `remember`, `ask_human`, `done`

To add one: add an entry to `ACTION_SPECS`, implement the `_h_<name>` handler in `service.py`, and describe it in `system_prompt.md`.

You can also drive the robot by hand:

```bash
curl http://127.0.0.1:8765/state
curl -X POST http://127.0.0.1:8765/action -H 'Content-Type: application/json' \
  -d '{"name":"move_chassis","params":{"forward_m":0.2,"right_m":0,"turn_left_deg":0}}'
curl -X POST http://127.0.0.1:8765/stop      # emergency stop, bypasses the action lock
```

## Safety

- All values are clamped on the server: at most 1 m, 180° and 100 mm of arm travel per action.
- One action at a time; `/stop` is never blocked.
- A forward move stops 0.15 m short of anything the front distance sensor sees.
- A move the SDK reports as done is rejected if odometry shows the robot barely moved.
- The server listens on 127.0.0.1 only. For a remote robot use an SSH tunnel, or `--token`.

## Evaluation

```bash
./run_all.sh                                             # all 100 tasks, headless, in the background
./run_all.sh --difficulty hard                           # or --task r01, --resume <batch_dir>
python -m robot_eval.run_evaluation --task e01 --repeat 5
```

`robot_eval/dataset.json` has 100 tasks (34 easy, 33 medium, 33 hard) across navigation, grounding, reasoning, manipulation and tidying. Each task is a list of milestones, judged by telemetry conditions or simulator ground truth where possible, and by a judge LLM on sampled frames otherwise. The report also counts tasks where the agent claimed success but failed. `robot_eval/office_tasks.json` holds 10 real-robot tasks scored by a human (`run_office_tasks.sh`).

## Tokens and cost

Every run prints its LLM calls and token count and stores them in `runs/<time>/history.json`. Set a price per million tokens in `.env` (`ROBOT_PRICE_*`, see `.env.example`) to get a cost as well.

```bash
.venv/bin/python tools/token_report.py          # tokens and cost per task, from all recorded runs
```

## Repository layout

```
RobotAgent.py        entry point
robot_agent/         agent loop, actions, skills, perception, LLM wrappers
robot_eval/          datasets, judging, metrics, batch runner
Test_Robot/          robot-side server, navigation map, arm drivers
Env/                 CoppeliaSim scene and sensor build scripts
tools/               calibration, recording, scan-to-scene and report utilities
scans/               LiDAR scan and floor plan of the office
*.sh                 launch scripts
```

## Credits

`robot_agent/llm/` is adapted from the LLM wrappers of the browser-use project (MIT). Everything else is original to this project. The agent contacts only the LLM API you configure and the local robot server.
