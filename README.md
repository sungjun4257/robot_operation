# zmq_hyundai_client_bimanual_cartesian.py

A ZMQ client that performs a single pick-and-place sequence on a Hyundai bimanual robot using **Cartesian (XYZ + RPY) control**.

The script combines an object pose from FoundationPose (`cam2target`) with a hand-eye calibration result (`cam2base`) to compute `base2target`, then runs: gripper-orientation correction → via/approach/grasp/end pose sequence → ZMQ command stream to the robot server.

---

## 1. Prerequisites

### Robot server
- A ZMQ REP server must be running on the robot PC at `tcp://161.122.114.39:5555`. (Same form as the `vision` command in the [calibration_hand2eye setup](memory:project_hand2eye_robot_server).)
- This script opens a REQ socket and connects to it ([zmq_hyundai_client_bimanual_cartesian.py:303-305](zmq_hyundai_client_bimanual_cartesian.py#L303-L305)).

### Input files

| Path | Contents |
|------|----------|
| `/home/vision/packages/FoundationPose/result/` | Read by `data_load_grasp()`. Requires `transformation` (4×4 cam2target) and `arm` (LEFT=0 / RIGHT=1) keys |
| `/home/vision/packages/FoundationPose/result/config.json` | `object_name` key. Used to choose the end (drop) height |
| `/home/vision/packages/calibration/data/data_1030_left/no_depth/hand_eye_v1_c2b_e2c_nominal_mean.yaml` | LEFT arm hand-eye (`c2b`) |
| `/home/vision/packages/calibration/data/data_1030_right/no_depth/hand_eye_v1_c2b_e2c_nominal_mean.yaml` | RIGHT arm hand-eye |

### Output
- `/home/vision/packages/FoundationPose/result/video.mp4` — RealSense color stream recording (640×480 @ 30 fps)

### Camera
- One Intel RealSense device. Color + depth streams are enabled and aligned to color ([zmq_hyundai_client_bimanual_cartesian.py:282-291](zmq_hyundai_client_bimanual_cartesian.py#L282-L291)).

---

## 2. Running

```bash
cd /home/vision/packages/robot_operation
python zmq_hyundai_client_bimanual_cartesian.py
```

No CLI arguments (the `argparse` block is commented out). `object_name` is read automatically from `config.json`.

---

## 3. Motion sequence

After `arm_flag` selects which arm to drive, the script sends 7 motion commands in this order:

```
current → via → before → target → [GRIPPER_CLOSE]
                                 ↓
                          before → via → end → [GRIPPER_OPEN] → current (home)
```

| Step | Pose | Meaning |
|------|------|---------|
| 1 | `via_matrix_xyzrpy` | 0.1 m above target, ±0.25 m lateral offset. Safe entry waypoint |
| 2 | `base2target_before_xyzrpy` | Pre-grasp approach, 8 cm back along gripper z |
| 3 | `base2target_xyzrpy` | Actual grasp pose |
| 4 | — | `GRIPPER_CLOSE` |
| 5 | `base2target_before_xyzrpy` | Retract |
| 6 | `via_matrix_xyzrpy` | Lift back to via |
| 7 | `end_matrix_xyzrpy` | Drop pose (height depends on `object_name`) |
| 8 | — | `GRIPPER_OPEN` |
| 9 | `current_xyzrpw` | Return to starting (home) pose |

### Drop height per object (`end_position.z`)
- `the red box.` → 0.20 m
- `the red can.`, `the white can.` → 0.14 m
- Anything else → 0.15 m

(XY is `[0.3, 0.0]` for both LEFT and RIGHT.)

---

## 4. Coordinate transform pipeline

```
cam2target  (FoundationPose) ─┐
                              ├─ base2target = cam2base · cam2target
cam2base    (hand-eye yaml)  ─┘
                              │
                              ├─ +0.1034 m along gripper z (TCP offset)
                              ├─ approach point: -0.08 m retreat
                              ├─ gripper orientation correction (down / forward facing)
                              └─ -90° rotation around z (gripper/base frame alignment)
```

### Automatic gripper-orientation correction ([zmq_hyundai_client_bimanual_cartesian.py:407-450](zmq_hyundai_client_bimanual_cartesian.py#L407-L450))
- If the gripper z-axis points down (`dot_down > 0.7`), rotate 180° about z so gripper y faces outward (base x).
- If the gripper z-axis points forward (`dot_forward > 0.7`), rotate 180° about z so gripper y faces up (base z).
- Always apply a final −90° rotation about z (gripper ↔ base frame alignment).

### `offset_y()`
- LEFT arm: y **+0.3 m**
- RIGHT arm: y **−0.3 m**

(Compensates because the two arms do not share a single base origin.)

---

## 5. ZMQ protocol

### Arm flag (1 byte)
| Value | Meaning |
|-------|---------|
| 0 | LEFT |
| 1 | RIGHT |

### NetProto (1 byte)
| Value | Name | Meaning |
|-------|------|---------|
| 0 | `MOVE_ARM_BY_XYZRPY` | Move in XYZ+RPY. Payload: 6 × float64 + motion_time (float64) |
| 1 | `MOVE_ARM_BY_Q` | Move in joint space (unused in this script) |
| 2 | `COMPLETE_MOVE_ARM` | Motion-complete reply |
| 3 | `REQ_B2EE` | Request base→ee matrix |
| 4 | `REQ_XYZROLLPITCHYAW` | Request current XYZRPY |
| 5 | `REQ_Q` | Request current joint angles |
| 6 | `REP_Q` | Joint-angles reply |
| 7 | `GRIPPER_OPEN` | Open gripper |
| 8 | `GRIPPER_CLOSE` | Close gripper |
| 9 | `COMPLETE_GRIPER` | Gripper-action complete reply |

### Wire format
```
[ARM_flag(1B)] [NetProto(1B)] [payload...]
```
- `MOVE_ARM_BY_XYZRPY`: payload = `6 × float64 (XYZRPY)` + `1 × float64 (motion_time)`
- `GRIPPER_OPEN/CLOSE`: no payload
- After sending, the script blocks on `socket_control.recv()` for the completion reply. While waiting, the non-blocking loop keeps grabbing RealSense frames and writing them to the video file.

---

## 6. Motion-time calculation

`compute_motion_time()` ([zmq_hyundai_client_bimanual_cartesian.py:243-269](zmq_hyundai_client_bimanual_cartesian.py#L243-L269))

```python
time_pos = ||Δxyz|| / max_linear_speed   # default 0.1 m/s
time_rot = ||Δrpy|| / max_angular_speed  # default 0.5 rad/s
motion_time = max(time_pos, time_rot)
```

Lower the two speed caps for more conservative motion.

---

## 7. Main dependencies

- `numpy`, `scipy`, `opencv-python`, `pyrealsense2`, `open3d`, `torch`, `pillow`, `pyzmq`
- `spatialmath-python`, `roboticstoolbox-python` (Panda DH IK)
- Local modules: [robot_simulation.py](robot_simulation.py) (`data_load_grasp`, `data_load_calib`), [dh_utils/DH_Panda.py](dh_utils/DH_Panda.py), [utils/DH_Panda.py](utils/DH_Panda.py)

---

## 8. Common issues

| Symptom | Cause / Fix |
|---------|-------------|
| `zmq.error.ZMQError: Connection refused` | ZMQ server (`vision`) not running on the robot PC. Start the server first |
| `Couldn't solve the IK: ...` | Target is outside the reachable workspace. Check `cam2target` or the calibration |
| RealSense stalls on the first frame | Check USB3 port, `pyrealsense2` install, or other processes holding the camera |
| Gripper approaches from the wrong direction | Inspect `dot_down` / `dot_forward` branch logs — if correction is insufficient, tweak the thresholds at [zmq_hyundai_client_bimanual_cartesian.py:407-450](zmq_hyundai_client_bimanual_cartesian.py#L407-L450) |
| Left/right poses are off | Verify the ±0.3 m in `offset_y()` matches the actual distance between the two arm bases |

---
