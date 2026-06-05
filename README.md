# EgoNav

A lightweight hybrid Vision-Language-Action system for real-time indoor robot navigation with stateful ego-context memory. EgoNav combines a frozen Vision-Language Model with deterministic control subsystems to achieve zero-shot, open-vocabulary object-goal navigation on consumer-grade hardware — no task-specific training required.

---

## Overview

EgoNav is a closed-loop navigation system that enables a mobile robot to navigate toward arbitrary objects described in natural language. Instead of relying on expensive task-specific training or pre-built maps, EgoNav leverages a frozen Vision-Language Model (VLM) as its deliberative core, treating navigation as a continuous visual question-answering task.

**Key design principles:**

- **Zero-shot navigation** — goals are specified in plain language; no fine-tuning or reward shaping is needed.
- **No task-specific training** — the VLM is used as-is, with structured prompting and ego-context memory providing task grounding.
- **Three-camera panoramic perception** — three wide-angle cameras at −30°, 0°, and +30° are stitched into a single 180° FOV composite, giving the robot wide spatial awareness.
- **Qwen2.5-VL:7b for deliberative reasoning** — a 7-billion-parameter VLM served locally via Ollama reasons about goal location, path clearance, and action selection.
- **YOLOv8n for reactive obstacle detection** — a lightweight object detector runs in parallel to guard against collisions, operating at frame rate independently of VLM latency.
- **Deterministic momentum controller** — during VLM inference (4–7 s), the last validated action is replayed to maintain continuous forward progress.
- **Stateful Ego-Context Memory** — an episodic memory buffer accumulates observations, actions, and reasoning across VLM calls, providing temporal context that a single-frame prompt cannot.
- **Runs on consumer hardware** — a Raspberry Pi 5 drives the robot; a consumer laptop with a GPU handles VLM inference. The two communicate over WiFi.

---

## System Architecture

EgoNav is organized into three layers that separate perception, reasoning, and actuation.

### Robot Layer

The robot layer runs on a **Raspberry Pi 5** and handles sensing and motor execution.

- Three USB cameras (OV2710 sensors) mounted at −30°, 0°, and +30° azimuth
- Differential-drive motor controller for discrete movement pulses
- **Flask-based MJPEG streaming server** — streams camera frames to the host over WiFi
- **FastAPI motor control server** — accepts movement commands from the host via HTTP

### Host Compute Layer

The host layer runs on a **consumer laptop** and handles all perception processing and decision-making.

- **Frame stitching** — captures from three cameras are composited into a single labeled panoramic image (960 × 213 px)
- **Asynchronous VLM inference** — Qwen2.5-VL:7b is queried with the panoramic image, the current goal, and the ego-context memory; inference runs asynchronously so the control loop is never blocked
- **YOLOv8n obstacle guard** — a position-based detection pipeline identifies obstacles in the robot's forward path and vetoes unsafe actions
- **Momentum controller** — replays the last validated action during VLM inference latency, ensuring the robot continues moving smoothly
- **Action validation pipeline** — cross-checks VLM-proposed actions against YOLO detections and robot state before execution

### Navigation Layer

The navigation layer bridges high-level decisions and low-level motor commands.

- **Trajectory controller** — translates discrete action tokens (forward, left, right, stop) into timed motor pulses
- **Robot state update** — tracks heading, action history, and goal proximity
- **Observation feedback** — new camera frames after each action close the perception–action loop

### Control Flow

```
Goal → Cameras → Frame Stitch → VLM Reasoning → Action Validation → Motor Execution → Updated Observation → ...
```

The loop runs at **12.5 Hz**. VLM calls are asynchronous; between calls, the momentum controller and YOLO guard maintain safe, continuous motion.

```
EgoNav (this repository)
        ↓
Validated Navigation Commands
        ↓
VLA-Bot-Controller (companion repository)
        ↓
Motor Execution and Physical Motion
```

The complete EgoNav platform consists of both the high-level navigation stack presented in this repository and the robot-side execution framework available in the companion [VLA-Bot-Controller](https://github.com/SparshSunilNaik/VLA-Bot-Controller.git) repository.

---

## Features

- **Natural-language goal specification** — e.g., *"find the chair"*, *"go to the door"*
- **Open-vocabulary navigation** — navigate to any object the VLM can recognize; no predefined object list
- **Three-camera 180° panoramic perception** — wide field of view with labeled spatial regions
- **Stateful Ego-Context Memory** — accumulates observations and reasoning across VLM calls for temporally grounded decisions
- **Position-based YOLO obstacle avoidance** — reactive collision prevention independent of VLM latency
- **Two-phase goal approach** — coarse navigation followed by fine-grained final-stop precision
- **Person safety detection** — detects and avoids people in the robot's path
- **Real-time operation** — 12.5 Hz control loop with 45 ms mean action latency
- **Voice and CLI goal input** — set goals by typing or speaking
- **Manual override** — WASD keyboard control for direct teleoperation
- **Multiple goal modes** — *approach*, *avoid*, and *around* modes for flexible task specification
- **Trajectory visualization overlay** — real-time HUD showing action history and goal status

---

## Repository Structure

```
GIT/
|--Bot Cam/
        |--- camcontrol.py
├── navigate.py
├── ego_state.py
└── README.md
```

> **Note:** Robot-side control code (motor server, camera streaming server), experimental logs, and configuration files are maintained in the companion [VLA-Bot-Controller](https://github.com/SparshSunilNaik/VLA-Bot-Controller.git) repository. See the [Companion Repository](#companion-repository) section below.

---

## How It Works

EgoNav operates as a closed-loop cycle:

1. **Goal input** — the user provides a natural-language goal (e.g., *"find the red backpack"*).
2. **Observation capture** — three cameras capture overlapping views of the environment.
3. **Frame stitching** — the captures are composited into a single labeled panoramic image with spatial region annotations (far-left, left, center, right, far-right).
4. **VLM reasoning** — the panoramic image, goal description, and accumulated ego-context memory are sent to Qwen2.5-VL:7b, which reasons about where the goal object is and which action to take.
5. **YOLO obstacle guard** — YOLOv8n validates the proposed action by checking for obstacles in the corresponding forward region.
6. **Motor execution** — the validated action is sent to the Raspberry Pi as a timed motor pulse.
7. **State update** — the robot's heading, action history, and ego-context memory are updated.
8. **Loop** — new camera frames are captured and the cycle repeats at 12.5 Hz.

During VLM inference (typically 4–7 s per call), the **momentum controller** replays the last validated action to keep the robot moving. The YOLO guard continues running at full frame rate to ensure safety throughout.

---

## Requirements

### Software

| Package | Purpose |
|---------|---------|
| Python 3.10+ | Runtime |
| `opencv-python` | Frame capture, stitching, and visualization |
| `numpy` | Array operations |
| `requests` | HTTP communication with robot servers |
| `ultralytics` | YOLOv8n obstacle detection |
| [Ollama](https://ollama.ai) | Local VLM serving (Qwen2.5-VL:7b) |

### Optional

| Package | Purpose |
|---------|---------|
| `speech_recognition` | Voice-based goal input |
| `pynput` | Global keyboard shortcuts for manual override |

### Hardware

- **Host:** Consumer laptop with a GPU (for VLM inference via Ollama)
- **Robot:** Raspberry Pi 5 with three USB cameras and a differential-drive motor controller
- **Network:** WiFi connection between host and robot

---

## Setup

### 1. Install Dependencies

```bash
pip install opencv-python numpy requests ultralytics
```

### 2. Install and Start Ollama

```bash
# Install Ollama from https://ollama.ai
ollama pull qwen2.5vl:7b
```

### 3. Configure Robot IP

Edit the `PI_IP` variable in `navigate.py` to match your Raspberry Pi's IP address:

```python
PI_IP = "192.168.x.x"  # Replace with your Pi's address
```

### 4. Start Robot Services

On the Raspberry Pi, start the camera streaming server and motor control server:

```bash
# Start the MJPEG camera server
python camera_server.py

# Start the FastAPI motor control server
python motor_server.py
```

### 5. Launch Navigation

```bash
python navigate.py
```

---

## Controls

| Key | Action |
|-----|--------|
| `1` | CLI goal input |
| `2` | Voice goal input |
| `3` | Switch to FAST mode (YOLO only) |
| `4` | Switch to SLOW mode (VLM deliberation) |
| `0` | Clear current goal |
| `W` / `A` / `S` / `D` | Manual override (forward / left / backward / right) |
| `7` / `8` / `9` | Toggle camera view (Left / Center / Right) |
| `Q` | Quit |

---

## Limitations

- Designed for **indoor environments** with relatively stable lighting conditions.
- **VLM inference latency** (4–7 s per call) limits the system's reactive capability; the momentum controller mitigates but does not eliminate this.
- **No global localization or mapping** — the robot has no persistent spatial map and relies entirely on egocentric visual context.
- **RGB-only perception** — no depth sensor is used; obstacle distance is estimated from bounding-box position in the image.
- **YOLOv8n detection limits** — the detector may miss objects in non-canonical orientations or unusual lighting.
- **Fixed forward safety cap** — the collision avoidance threshold does not adapt dynamically to obstacle distance or robot speed.

---

## Companion Repository

EgoNav is part of a two-repository system. This repository contains the high-level navigation stack — Vision-Language-Action reasoning, panoramic perception, Qwen2.5-VL inference, Stateful Ego-Context Memory, and action validation. The robot-side execution framework is maintained separately:

**[VLA-Bot-Controller](https://github.com/SparshSunilNaik/VLA-Bot-Controller.git)**

The companion repository provides:

- Robot motor control and actuation
- Command execution via FastAPI
- Camera streaming server
- Embedded Raspberry Pi operation

Together, the two repositories form the complete EgoNav platform. Reviewers and researchers can access both to reproduce or extend the full system.
