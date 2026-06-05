"""
navigate.py — EgoNav: Hybrid VLA Navigation System
==================================================
A hybrid Vision-Language-Action navigation system combining:
  - Frozen VLM (Qwen2.5-VL) for deliberative reasoning
  - YOLOv8n for reactive obstacle detection
  - Deterministic momentum controller for continuous motion
  - Stateful Ego-Context Memory for episodic context

Usage:
    python3 navigate.py
"""

import cv2
import time
import threading
import requests
import json
import os
import math
import base64
import numpy as np
from collections import deque
from datetime import datetime
from ego_state import EgoState, StatefulVLM

# ── Voice ─────────────────────────────────────────────────────────
try:
    import speech_recognition as sr
    VOICE_AVAILABLE = True
except ImportError:
    VOICE_AVAILABLE = False
    print("⚠ SpeechRecognition not available.")

# ── Keyboard ──────────────────────────────────────────────────────
# pynput loads the macOS Quartz framework which can take 5-10s on Python 3.14.
# Load it in a background thread so startup is instant.
pynput_keyboard = None
PYNPUT_AVAILABLE = False
_pynput_lock = threading.Lock()

def _load_pynput():
    global pynput_keyboard, PYNPUT_AVAILABLE
    try:
        from pynput import keyboard as _kb
        with _pynput_lock:
            pynput_keyboard = _kb
            PYNPUT_AVAILABLE = True
        print("  ✓ pynput loaded (global keyboard active)")
    except Exception as e:
        print(f"  ⚠ pynput unavailable: {e}. Keys only work in OpenCV window.")

threading.Thread(target=_load_pynput, daemon=True, name="pynput-loader").start()

# ── YOLO ─────────────────────────────────────────────────────────
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("⚠ ultralytics not installed. YOLO fast mode disabled.")
    print("  Install with: pip install ultralytics")


# ══════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════

#PI_IP = "192.168.29.149" #home
PI_IP = "10.253.9.119" #hotspot

CAM_LEFT   = f"http://{PI_IP}:9000/left"
CAM_CENTER = f"http://{PI_IP}:9000/center"
CAM_RIGHT  = f"http://{PI_IP}:9000/right"

ROBOT_URL = f"http://{PI_IP}:9999"

VLM_MODEL = "qwen2.5vl:7b"
#VLM_MODEL="gemma4:e2b"

VLM_API_URL = "http://localhost:11434/api/chat"
VLM_TIMEOUT = 30
VLM_API_FORMAT = "ollama"     # "ollama" or "llamacpp"
VLM_IMG_WIDTH = 320           # wider panoramic = better spatial resolution (~960px panoramic)
VLM_NUM_CTX = 2048            # larger context for better reasoning
VLM_NUM_PREDICT = 80          # longer CoT → fewer hallucinations (~1s extra latency)

STEP_DURATION = 0.10
STOP_DELAY = 0.02

# Blind spot: if object vanishes, continue last action for this many seconds
BLIND_SPOT_TIMEOUT = 1.5
EGO_HISTORY_SIZE = 1   # 1 pair = 2 msgs, minimal history for speed

LOG_DIR = "logs"
CYCLE_DELAY = 0.08
DIRECT_STEP_DURATION = 1.0   # seconds per step in a sequence

# YOLO config
YOLO_MODEL_PATH = "yolov8n.pt"
YOLO_CONFIDENCE = 0.30   # lower threshold catches objects at distance/angle
YOLO_MIN_AREA = 500       # minimum bbox area in pixels to count as valid
YOLO_EVERY_N = 3          # only run YOLO every N cycles, cache between
YOLO_TRACKED_OBJECTS = {
    "bottle", "person", "chair", "backpack", "cup",
    "laptop", "book", "cell phone", "keyboard", "dog", "cat",
    "handbag", "suitcase", "umbrella", "sports ball",
}

# Default inference mode: "fast" (YOLO) or "slow" (VLM)
DEFAULT_INFERENCE_MODE = "slow"

# ── GOAL-REACHED ──
GOAL_REACHED_AREA_RATIO = 0.08   # YOLO stop in detect_panels (full stop)

# Two-phase approach
APPROACH_TRANSITION_AREA = 0.04  # Goal fills >4% center frame → enter FINAL STOP mode
FINAL_STOP_AREA          = 0.10  # Goal fills >10% in final-stop → STOP
FINAL_STOP_MAX_STEPS     = 15    # Hard timeout: 15 slow creep steps in final-stop mode
FINAL_STOP_STEP_INTERVAL = 3     # Only move forward every N cycles in final-stop (slow creep)

# ── PERSON SAFETY ──
SAFETY_OBJECTS = {"person"}
SAFETY_AREA_RATIO = 0.20   # raised threshold — avoids false triggers from chair/furniture misclassified as person

# "Go around" bypass — ADAPTIVE (see build_around_sequence)
AROUND_CENTER_HITS = 2  # require N consecutive center detections before bypass
CENTER_CONFIRM = 2      # require N consecutive center detections before going forward

# ── PATH OBSTACLE DETECTION ──
PATH_OBSTACLE_AREA_RATIO = 0.02   # detect obstacles from further away
PATH_OBSTACLE_CLASSES = {
    "bottle", "cup", "wine glass", "vase", "bowl",
    "backpack", "handbag", "suitcase",
    "sports ball", "frisbee", "skateboard", "umbrella",
    "book", "clock", "keyboard", "mouse",
}  # NEVER treat furniture/vehicles as obstacles (they are often the GOAL)

session = requests.Session()


# ══════════════════════════════════════════════════════════════════
# CAMERA — one per stream, threaded, latest frame only
# ══════════════════════════════════════════════════════════════════

STALE_TIMEOUT = 3.0   # if no new frame for this long, treat camera as dead

class Camera:
    def __init__(self, url, name):
        self.url = url
        self.name = name
        self._frame = None
        self._time = 0.0
        self._lock = threading.Lock()
        self._running = False

    def start(self):
        self._running = True
        threading.Thread(target=self._loop, daemon=True, name=f"cam_{self.name}").start()
        for _ in range(100):
            if self._frame is not None:
                return True
            time.sleep(0.1)
        return False

    def _loop(self):
        cap = cv2.VideoCapture(self.url)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        fails = 0
        try:
            while self._running:
                ret, frame = cap.read()
                if ret and frame is not None:
                    # Check if it's a near-black frame (dead camera sends black)
                    if frame.mean() < 2.0:
                        # Black frame from server = camera is dead
                        with self._lock:
                            self._frame = None
                            self._time = time.time()
                        fails += 1
                    else:
                        with self._lock:
                            self._frame = frame
                            self._time = time.time()
                        fails = 0
                else:
                    fails += 1
                    with self._lock:
                        self._frame = None  # immediately mark dead
                    if fails > 10:
                        try: cap.release()
                        except: pass
                        time.sleep(1)
                        cap = cv2.VideoCapture(self.url)
                        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                        fails = 0
                    time.sleep(0.01)
        finally:
            try: cap.release()
            except: pass

    def get(self):
        with self._lock:
            if self._frame is None:
                return None, 0.0
            # Check staleness
            if time.time() - self._time > STALE_TIMEOUT:
                self._frame = None
                return None, 0.0
            return self._frame.copy(), self._time

    def stop(self):
        self._running = False


# ══════════════════════════════════════════════════════════════════
# MOTOR — non-blocking
# ══════════════════════════════════════════════════════════════════

def send_action(action):
    """Fire-and-forget: one step then stop."""
    def _go():
        try:
            if action == "stop":
                session.post(f"{ROBOT_URL}/api/stop", timeout=1.0)
            else:
                session.post(f"{ROBOT_URL}/api/{action}", timeout=1.0)
                time.sleep(STEP_DURATION)
                session.post(f"{ROBOT_URL}/api/stop", timeout=1.0)
                time.sleep(STOP_DELAY)
        except:
            pass
    threading.Thread(target=_go, daemon=True).start()

def emergency_stop():
    for _ in range(3):
        try: session.post(f"{ROBOT_URL}/api/stop", timeout=0.5)
        except: pass
        time.sleep(0.05)


# ══════════════════════════════════════════════════════════════════
# VLM WARMUP — pre-load model into VRAM at startup
# ══════════════════════════════════════════════════════════════════

def warmup_vlm():
    """Send a tiny text-only request to Ollama to force model loading.
    This runs at startup so the first real inference is fast (~3-5s)
    instead of cold-start slow (~20s).
    """
    print("  Warming up VLM (loading model into VRAM)...")
    t0 = time.time()
    try:
        payload = {
            "model": VLM_MODEL,
            "messages": [
                {"role": "user", "content": "Say OK."}
            ],
            "stream": False,
            "keep_alive": "30m",
            "options": {
                "num_predict": 3,
                "temperature": 0.0,
                "num_ctx": 128,
            }
        }
        resp = session.post(VLM_API_URL, json=payload, timeout=60)
        elapsed = time.time() - t0
        if resp.status_code == 200:
            print(f"  ✓ VLM warm ({elapsed:.1f}s) — model loaded into VRAM")
        else:
            print(f"  ⚠ VLM warmup status {resp.status_code} ({elapsed:.1f}s)")
    except Exception as e:
        elapsed = time.time() - t0
        print(f"  ⚠ VLM warmup failed ({elapsed:.1f}s): {e}")
        print(f"    First inference will be slow (~20s cold start)")


# ══════════════════════════════════════════════════════════════════
# VLM — EgoState + StatefulVLM imported from ego_state.py
# ══════════════════════════════════════════════════════════════════
# EgoState:    Robot self-model (memory, observations, trajectory)
# StatefulVLM: 3-panel checker using /api/chat with rolling history
# Both imported at top of file.

# ══════════════════════════════════════════════════════════════════
# YOLO MULTI-CAMERA ENGINE
# ══════════════════════════════════════════════════════════════════

def build_around_sequence(area_ratio, panel_side="left"):
    """Build an adaptive bypass sequence based on how close the object is.
    area_ratio: bbox area / frame area (bigger = closer = wider bypass)
    panel_side: which panel the object was last seen in (affects dodge direction)
    """
    # Dodge direction: go opposite to where object is
    # If object is in center, dodge left by default
    if panel_side == "right":
        dodge = "left"
        return_dir = "right"
    else:
        dodge = "right"
        return_dir = "left"

    if area_ratio > 0.15:
        # VERY close → wide bypass
        fwd_count = 4
    elif area_ratio > 0.08:
        # Close → medium bypass
        fwd_count = 3
    else:
        # Far → short bypass
        fwd_count = 2

    seq = []
    seq.append(dodge)                    # turn away
    seq.extend(["forward"] * fwd_count)  # go past
    seq.append(return_dir)               # turn back
    seq.append(return_dir)               # extra turn back to face original direction
    seq.extend(["forward"] * fwd_count)  # continue past
    seq.append(dodge)                    # straighten out
    return seq


class YOLOMultiCam:
    """Runs YOLO on each camera frame independently.
    Returns: (action, panel, reason, detections_dict, ms)
    Priority: center > left > right (same as VLM).
    Caches results for YOLO_EVERY_N cycles.
    """

    def __init__(self, model_path=YOLO_MODEL_PATH, confidence=YOLO_CONFIDENCE):
        self.model = None
        self.confidence = confidence
        self.calls = 0
        self._cycle_counter = 0
        self._cached_result = (None, None, "no cache", {}, 0)
        self._last_goal = None
        self._last_area = 0
        self._last_area_ratio = 0.0
        if YOLO_AVAILABLE:
            try:
                print(f"  Loading YOLO: {model_path}...")
                self.model = YOLO(model_path)
                print(f"  ✓ YOLO loaded")
            except Exception as e:
                print(f"  ⚠ YOLO load failed: {e}")

    @property
    def available(self):
        return self.model is not None

    def clear_cache(self):
        """Clear cached detections (call on goal change)."""
        self._cached_result = (None, None, "cache cleared", {}, 0)
        self._cycle_counter = 0
        self._last_goal = None
        self._last_area_ratio = 0.0

    def detect_panels(self, frames, goal):
        """Run YOLO on all cameras, find which panel has the goal.
        Returns (action, panel, reason, all_detections, ms).
        Uses interval caching: only runs inference every YOLO_EVERY_N cycles.
        """
        if not self.available:
            return None, None, "YOLO not available", {}, 0

        # If goal changed, invalidate cache
        if goal != self._last_goal:
            self._cached_result = (None, None, "goal changed", {}, 0)
            self._cycle_counter = 0
            self._last_goal = goal

        # Return cached result if not time to re-detect
        self._cycle_counter += 1
        if self._cycle_counter % YOLO_EVERY_N != 1 and self._cached_result[0] is not None:
            return self._cached_result

        t0 = time.time()
        all_dets = {}  # {"left": [...], "center": [...], "right": [...]}
        goal_lower = goal.lower() if goal else ""

        # Synonym expansion for YOLO class names
        YOLO_SYNONYMS = {
            "human": ["person"], "man": ["person"], "woman": ["person"],
            "people": ["person"], "figure": ["person"],
            "phone": ["cell phone"], "mobile": ["cell phone"],
            "bag": ["backpack", "handbag", "suitcase"],
            "ball": ["sports ball"],
        }
        # What YOLO classes match the goal?
        target_classes = {goal_lower}
        if goal_lower in YOLO_SYNONYMS:
            target_classes.update(YOLO_SYNONYMS[goal_lower])
        # Also check if goal IS a YOLO class directly
        target_classes = {c for c in target_classes if c}
        # person as goal requires higher confidence (chairs misclassified as person)
        person_goal_min_conf = 0.55 if goal_lower in ("person","human","man","woman") else 0.0

        best_panel = None
        best_det = None
        best_area = 0

        # Check cameras in priority order
        for cam_name in ["center", "left", "right"]:
            frame = frames.get(cam_name)
            if frame is None:
                all_dets[cam_name] = []
                continue

            try:
                results = self.model(frame, conf=self.confidence, verbose=False)
            except Exception as e:
                print(f"  ⚠ YOLO error on {cam_name}: {e}")
                all_dets[cam_name] = []
                continue

            dets = []
            for result in results:
                if result.boxes is None:
                    continue
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    cls_name = result.names[cls_id]
                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    area = int((x2 - x1) * (y2 - y1))
                    cx = (x1 + x2) / 2
                    cy = (y1 + y2) / 2
                    dets.append({
                        "class": cls_name,
                        "confidence": round(conf, 3),
                        "bbox": (int(x1), int(y1), int(x2), int(y2)),
                        "cx": cx, "cy": cy,   # bbox center for position-based detection
                        "area": area,
                    })
                    # Check if this is our goal
                    if cls_name in target_classes \
                            and area > best_area \
                            and area >= YOLO_MIN_AREA \
                            and conf >= person_goal_min_conf:  # higher bar for person
                        # Center camera gets priority: if we already found
                        # the goal in center, only replace with a MUCH bigger
                        # detection from a side camera (2x area)
                        if best_panel == "center" and cam_name != "center":
                            if area < best_area * 2:
                                continue  # skip — center wins
                        best_panel = cam_name
                        best_det = {"class": cls_name, "confidence": conf,
                                    "area": area, "panel": cam_name}
                        best_area = area

            all_dets[cam_name] = dets

        total_ms = (time.time() - t0) * 1000
        self.calls += 1

        # Debug: log what each camera detected
        for cam_name in ["left", "center", "right"]:
            cam_dets = all_dets.get(cam_name, [])
            if cam_dets:
                det_str = ", ".join(f"{d['class']}({d['confidence']:.2f} a={d['area']})"
                                    for d in cam_dets[:3])
                print(f"    [{cam_name.upper():>6}] {det_str}")

        if best_panel and best_det:
            # Compute area ratio for goal-reached detection
            ref_frame = frames.get(best_panel)
            if ref_frame is not None:
                frame_area = ref_frame.shape[0] * ref_frame.shape[1]
            else:
                frame_area = 640 * 480  # fallback
            area_ratio = best_det['area'] / frame_area if frame_area > 0 else 0

            # GOAL-REACHED: object fills >12% of frame → stop!
            if area_ratio > GOAL_REACHED_AREA_RATIO and best_panel == "center":
                reason = (f"🎯 GOAL REACHED: {best_det['class']} fills "
                          f"{area_ratio:.1%} of frame — stopping")
                print(f"  {reason}")
                result = ("stop", best_panel, reason, all_dets, total_ms)
                self._cached_result = result
                self._last_area = best_det['area']
                self._last_area_ratio = area_ratio
                return result

            PANEL_ACTION = {"left": "left", "center": "forward", "right": "right"}
            action = PANEL_ACTION[best_panel]
            reason = (f"{best_det['class']} in {best_panel} "
                      f"(conf={best_det['confidence']:.2f}, "
                      f"area={best_det['area']}, ratio={area_ratio:.3f})")
            result = (action, best_panel, reason, all_dets, total_ms)
            self._cached_result = result
            self._last_area = best_det['area']
            self._last_area_ratio = area_ratio
            return result

        # Not found — cache the miss too so we don't re-run immediately
        result = (None, None, "not detected by YOLO", all_dets, total_ms)
        self._cached_result = result
        self._last_area = 0
        return result

# ══════════════════════════════════════════════════════════════════

class BlindSpotHandler:
    """When the goal disappears between panels, continue last action
    briefly instead of immediately searching.

    Logic:
      - Object was in center → disappeared → brief forward (might be close)
      - Object was in left → disappeared → brief left (crossing blind spot)
      - Object was in right → disappeared → brief right (crossing blind spot)
      - Timeout after BLIND_SPOT_TIMEOUT seconds → give up, search
    """

    def __init__(self):
        self.last_seen_panel = None
        self.last_seen_time = 0.0
        self.last_seen_action = None

    def update(self, panel_found):
        """Call when VLM finds object in a panel."""
        if panel_found is not None:
            self.last_seen_panel = panel_found
            self.last_seen_time = time.time()
            if panel_found == "center":
                self.last_seen_action = "forward"
            elif panel_found == "left":
                self.last_seen_action = "left"
            elif panel_found == "right":
                self.last_seen_action = "right"

    def get_blind_spot_action(self):
        """If object recently vanished, return the continuation action.
        Returns (action, True) if in blind spot, (None, False) otherwise.
        """
        if self.last_seen_panel is None:
            return None, False

        age = time.time() - self.last_seen_time
        if age < BLIND_SPOT_TIMEOUT:
            return self.last_seen_action, True
        else:
            # Timeout — clear memory
            self.last_seen_panel = None
            return None, False

    def reset(self):
        self.last_seen_panel = None
        self.last_seen_time = 0.0
        self.last_seen_action = None


# ══════════════════════════════════════════════════════════════════
# COMMAND PARSER — direct commands + object goals
# ══════════════════════════════════════════════════════════════════

# Direct motor commands — bypass VLM entirely
DIRECT_COMMANDS = {
    "forward": "forward", "straight": "forward", "ahead": "forward",
    "go straight": "forward", "go forward": "forward", "move forward": "forward",
    "go ahead": "forward",
    "left": "left", "turn left": "left", "go left": "left",
    "change left": "left", "move left": "left",
    "right": "right", "turn right": "right", "go right": "right",
    "change right": "right", "move right": "right",
    "stop": "stop", "halt": "stop", "brake": "stop",
    "backward": "backward", "reverse": "backward", "go back": "backward",
    "back": "backward", "move back": "backward",
}

FILLER = {"go", "to", "find", "move", "toward", "towards", "the", "a", "an",
          "please", "get", "fetch", "then", "and", "also"}
AVOID_KEYWORDS = {"avoid", "dodge", "away", "skip", "ignore"}
AROUND_KEYWORDS = {"around", "past", "beside", "circle"}

def parse_command(text):
    """Parse user input → (result, type).

    Returns one of:
      ('forward', 'direct')           — single direct command
      (['forward','left'], 'sequence')— multi-step sequence
      ('bottle', 'approach')          — object goal, uses VLM
      ('chair', 'avoid')              — avoid object, uses VLM
      (None, None)                    — could not parse
    """
    normalized = text.lower().strip()

    # Split on "then" / "and then" / "," for sequences
    import re
    parts = re.split(r'\s+then\s+|\s+and\s+then\s+|\s+and\s+|,\s*', normalized)
    parts = [p.strip() for p in parts if p.strip()]

    if len(parts) > 1:
        # Try to parse each part as a direct command
        actions = []
        for part in parts:
            act = _parse_single_direct(part)
            if act:
                actions.append(act)
            else:
                # If any part isn't a direct command, treat the whole thing
                # as a single non-sequence command
                break
        else:
            # All parts parsed as direct commands
            if len(actions) == 1:
                return actions[0], "direct"
            return actions, "sequence"

    # Single command
    return _parse_single(normalized)


def _parse_single_direct(text):
    """Try to parse text as a single direct motor command. Returns action or None."""
    normalized = text.lower().strip()
    for phrase in sorted(DIRECT_COMMANDS, key=len, reverse=True):
        if normalized == phrase or normalized.startswith(phrase + " "):
            return DIRECT_COMMANDS[phrase]
    # Bare direction words
    words = normalized.split()
    filler = {"go", "move", "turn", "change", "the", "a", "please"}
    meaningful = [w for w in words if w not in filler]
    if meaningful and meaningful[-1] in DIRECT_COMMANDS:
        return DIRECT_COMMANDS[meaningful[-1]]
    return None


def _parse_single(text):
    """Parse a single (non-sequence) command."""
    normalized = text.lower().strip()

    # Check direct commands (longest match first)
    for phrase in sorted(DIRECT_COMMANDS, key=len, reverse=True):
        if normalized == phrase or normalized.startswith(phrase + " "):
            return DIRECT_COMMANDS[phrase], "direct"

    # Otherwise parse as object goal
    words = normalized.split()
    mode = "approach"
    for w in words:
        if w in AROUND_KEYWORDS:
            mode = "around"
            break
        if w in AVOID_KEYWORDS:
            mode = "avoid"
            break

    all_filter = FILLER | AVOID_KEYWORDS | AROUND_KEYWORDS
    meaningful = [w for w in words if w not in all_filter]
    obj = meaningful[-1] if meaningful else None
    if obj and obj in ("forward", "left", "right", "stop", "backward",
                       "straight", "ahead", "back", "reverse", "halt"):
        return DIRECT_COMMANDS.get(obj, obj), "direct"
    return (obj, mode) if obj else (None, None)


# ══════════════════════════════════════════════════════════════════
# INPUT HANDLER — pynput + cv2 + voice
# ══════════════════════════════════════════════════════════════════

WASD_MAP = {'w': 'forward', 'a': 'left', 's': 'stop', 'd': 'right'}
OVERRIDE_DURATION = 0.8

class InputHandler:
    def __init__(self):
        self._lock = threading.Lock()
        self.goal = None
        self.goal_mode = "approach"
        self.direct_action = None
        self._direct_queue = deque()
        self._direct_step_start = 0.0
        self.cli_active = False
        self.cli_buffer = ""
        self.voice_active = False
        self.quit_requested = False
        self._override_action = None
        self._override_time = 0.0
        self._listener = None
        self.disabled_cameras = set()
        self.inference_mode = DEFAULT_INFERENCE_MODE  # "fast" or "slow"

    def start(self):
        if PYNPUT_AVAILABLE:
            self._start_pynput()
        self._start_stdin_reader()
        print(f"  Input: {'pynput + cv2 + stdin' if PYNPUT_AVAILABLE else 'cv2 + stdin'}")
        print("  ── Terminal commands: type goal (e.g. 'bottle'), 'fast', 'slow', 'q' ──")

    def _start_stdin_reader(self):
        """Background thread that reads goals from terminal stdin.
        Works regardless of cv2 focus or pynput permissions.
        Commands:
          <object>   → set goal (e.g. 'bottle', 'person')
          fast / f   → switch to YOLO mode
          slow / s   → switch to VLM mode
          clear / c  → clear goal
          q          → quit
          around <obj> → around mode
          avoid <obj>  → avoid mode
        """
        def _reader():
            print("  [stdin] Ready. Type a goal and press Enter:")
            while True:
                try:
                    line = input().strip()
                except EOFError:
                    break
                if not line:
                    continue
                low = line.lower()
                if low in ('q', 'quit'):
                    with self._lock:
                        self.quit_requested = True
                    print("  [stdin] Quitting...")
                    break
                elif low in ('fast', 'f', '3'):
                    self._set_inference_mode('fast')
                elif low in ('slow', 'l', '4'):
                    self._set_inference_mode('slow')
                elif low in ('clear', 'c', '0'):
                    self._clear_goal()
                else:
                    # Treat as goal/command
                    self._apply_parsed(line)
        threading.Thread(target=_reader, daemon=True, name="stdin-reader").start()

    def _start_pynput(self):
        def on_press(key):
            try:
                ch = key.char.lower() if hasattr(key, 'char') and key.char else None
                if ch is None:
                    return
                with self._lock:
                    if self.cli_active:
                        return
                if ch == 'q':
                    with self._lock:
                        self.quit_requested = True
                elif ch == '1':
                    self._activate_cli()
                elif ch == '2':
                    self._activate_voice()
                elif ch == '0':
                    self._clear_goal()
                elif ch in WASD_MAP:
                    with self._lock:
                        self._override_action = WASD_MAP[ch]
                        self._override_time = time.time()
                elif ch == 'e':
                    with self._lock:
                        self._override_action = None
                elif ch in ('7', '8', '9'):
                    cam_map = {'7': 'left', '8': 'center', '9': 'right'}
                    self._toggle_camera(cam_map[ch])
                elif ch == '3':
                    self._set_inference_mode('fast')
                elif ch == '4':
                    self._set_inference_mode('slow')
            except AttributeError:
                pass

        # Wait for pynput to finish loading (background thread, up to 15s)
        deadline = time.time() + 15
        while not PYNPUT_AVAILABLE and time.time() < deadline:
            time.sleep(0.1)

        if PYNPUT_AVAILABLE and pynput_keyboard is not None:
            self._listener = pynput_keyboard.Listener(on_press=on_press)
            self._listener.daemon = True
            self._listener.start()
            print("  ✓ pynput listener started")
        else:
            self._listener = None
            print("  ⚠ pynput not ready — using cv2 key fallback only")

    def _toggle_camera(self, cam_name):
        """Toggle a camera as disabled (blind spot)."""
        with self._lock:
            if cam_name in self.disabled_cameras:
                self.disabled_cameras.discard(cam_name)
                print(f"\n  📡 Camera {cam_name.upper()} ENABLED")
            else:
                self.disabled_cameras.add(cam_name)
                print(f"\n  🚧 Camera {cam_name.upper()} DISABLED (blind spot)")

    def _set_inference_mode(self, mode):
        """Switch between fast (YOLO) and slow (VLM) modes."""
        with self._lock:
            self.inference_mode = mode
        label = "⚡ FAST (YOLO ~50ms)" if mode == "fast" else "🧠 SLOW (VLM ~5s)"
        print(f"\n  {label}")

    def _activate_cli(self):
        with self._lock:
            self.cli_active = True
            self.cli_buffer = ""
        print("\n  📝 CLI ACTIVE — click the camera window, then type your goal + Enter")

    def _activate_voice(self):
        if not VOICE_AVAILABLE:
            print("\n  ⚠ Voice not available")
            return
        with self._lock:
            if self.voice_active:
                return
            self.voice_active = True
        threading.Thread(target=self._voice_worker, daemon=True).start()

    def _voice_worker(self):
        recognizer = sr.Recognizer()
        try:
            print("\n  🎤 Listening...")
            with sr.Microphone() as source:
                recognizer.adjust_for_ambient_noise(source, duration=0.5)
                audio = recognizer.listen(source, timeout=5, phrase_time_limit=5)
            text = recognizer.recognize_google(audio)
            print(f"  🎤 Heard: \"{text}\"")
            self._apply_parsed(text)
        except Exception as e:
            print(f"  ⚠ Voice error: {e}")
        finally:
            with self._lock:
                self.voice_active = False

    def _clear_goal(self):
        with self._lock:
            self.goal = None
            self.goal_mode = "approach"
            self.direct_action = None
            self._direct_queue.clear()
        print("\n  🗑 Goal cleared")

    def _apply_parsed(self, text):
        """Parse text and apply result to state."""
        normalized = text.lower().strip()

        # ── "turn around" → 50 left steps (180° spin) ──────────────────
        if normalized in ("turn around", "turnaround", "turn back", "180",
                          "spin around", "rotate", "u-turn", "uturn"):
            seq = ["left"] * 50
            with self._lock:
                self._direct_queue.clear()
                self._direct_queue.extend(seq)
                self.direct_action = self._direct_queue.popleft()
                self._direct_step_start = time.time()
                self.goal = None
                self.goal_mode = "direct"
            print(f"\n  🔄 TURN AROUND: executing 50 left steps (~180° spin)")
            return

        result, mode = parse_command(text)
        if not result:
            print(f"\n  ⚠ Could not parse: \"{text}\"")
            return
        with self._lock:
            if mode == "sequence":
                # Multi-step: load queue
                self._direct_queue.clear()
                self._direct_queue.extend(result)
                self.direct_action = self._direct_queue.popleft()
                self._direct_step_start = time.time()
                self.goal = None
                self.goal_mode = "direct"
                seq_str = " → ".join(result)
                print(f"\n  ✓ Sequence: {self.direct_action} → {seq_str}  ({len(result)+1} steps)")
            elif mode == "direct":
                self._direct_queue.clear()
                self.direct_action = result
                self._direct_step_start = time.time()
                self.goal = None
                self.goal_mode = "direct"
                print(f"\n  ✓ Command: DIRECT {result}")
            else:
                self._direct_queue.clear()
                self.direct_action = None
                self.goal = result
                self.goal_mode = mode
                print(f"\n  ✓ Command: {mode.upper()} {result}")

    def trigger_bypass_sequence(self, sequence):
        """Called by main loop when 'around' mode detects the object
        is close enough. Switches to direct sequence for bypass maneuver.
        """
        with self._lock:
            self._direct_queue.clear()
            self._direct_queue.extend(sequence)
            self.direct_action = self._direct_queue.popleft()
            self._direct_step_start = time.time()
            self.goal = None
            self.goal_mode = "direct"
        seq_str = " → ".join(sequence)
        print(f"\n  🔄 BYPASS: {self.direct_action} → {seq_str}  ({len(sequence)+1} steps)")

    def handle_cv2_key(self, key):
        if key == 255 or key == -1:
            return
        with self._lock:
            is_cli = self.cli_active
        if is_cli:
            if key == 27:
                with self._lock:
                    self.cli_active = False
                    self.cli_buffer = ""
                print("\n  CLI cancelled")
            elif key in (13, 10):
                with self._lock:
                    text = self.cli_buffer.strip()
                    self.cli_active = False
                    self.cli_buffer = ""
                if text:
                    self._apply_parsed(text)
            elif key in (8, 127):
                with self._lock:
                    self.cli_buffer = self.cli_buffer[:-1]
            elif 32 <= key < 127:
                with self._lock:
                    self.cli_buffer += chr(key)
        else:
            # Always handle cv2 keys — pynput may lack Accessibility permissions on macOS
            ch = chr(key).lower() if 0 <= key < 128 else None
            if ch == 'q':
                with self._lock:
                    self.quit_requested = True
            elif ch == '1':
                self._activate_cli()
            elif ch == '2':
                self._activate_voice()
            elif ch == '0':
                self._clear_goal()
            elif ch and ch in WASD_MAP:
                with self._lock:
                    self._override_action = WASD_MAP[ch]
                    self._override_time = time.time()
            elif ch in ('7', '8', '9'):
                cam_map = {'7': 'left', '8': 'center', '9': 'right'}
                self._toggle_camera(cam_map[ch])
            elif ch == '3':
                self._set_inference_mode('fast')
            elif ch == '4':
                self._set_inference_mode('slow')

    def get_state(self):
        with self._lock:
            # Advance direct command queue if current step has expired
            if self.direct_action and self.goal_mode == "direct":
                elapsed = time.time() - self._direct_step_start
                if elapsed >= DIRECT_STEP_DURATION:
                    if self._direct_queue:
                        prev = self.direct_action
                        self.direct_action = self._direct_queue.popleft()
                        self._direct_step_start = time.time()
                        remaining = len(self._direct_queue)
                        print(f"  [▶] {prev} → {self.direct_action}  "
                              f"({remaining} remaining)")
                    else:
                        # Sequence complete
                        print(f"  [✓] Sequence complete ({self.direct_action})")
                        self.direct_action = None
                        self.goal_mode = "approach"

            override = None
            if self._override_action:
                if time.time() - self._override_time < OVERRIDE_DURATION:
                    override = self._override_action
                else:
                    self._override_action = None

            queue_preview = list(self._direct_queue)
            return {
                "goal": self.goal,
                "goal_mode": self.goal_mode,
                "direct_action": self.direct_action,
                "direct_queue": queue_preview,
                "cli_active": self.cli_active,
                "cli_buffer": self.cli_buffer,
                "voice_active": self.voice_active,
                "quit": self.quit_requested,
                "override": override,
                "disabled_cameras": set(self.disabled_cameras),
                "inference_mode": self.inference_mode,
            }

    def stop(self):
        if self._listener:
            self._listener.stop()


# ══════════════════════════════════════════════════════════════════
# LOGGER
# ══════════════════════════════════════════════════════════════════

class Logger:
    def __init__(self):
        os.makedirs(LOG_DIR, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(LOG_DIR, f"egonav_{ts}.jsonl")
        self._f = open(path, "w")
        print(f"  Log: {path}")

    def log(self, entry):
        self._f.write(json.dumps(entry) + "\n")
        self._f.flush()

    def close(self):
        self._f.close()


# ══════════════════════════════════════════════════════════════════
# TRAJECTORY OVERLAY — future path drawn on camera frame
# ══════════════════════════════════════════════════════════════════

def draw_trajectory_overlay(frame, action, panel="center"):
    """Draw future trajectory prediction on a camera frame.
    Like the reference: red dots/dashes showing predicted robot path.
    """
    if frame is None or action == "stop":
        return frame

    overlay = frame.copy()
    h, w = overlay.shape[:2]
    cx = w // 2
    bot_y = h - 10  # start from bottom

    color = (0, 0, 255)     # red
    glow = (0, 80, 255)     # orange glow

    if action == "forward" and panel == "center":
        # Dashed line going straight up from bottom center
        for i in range(12):
            y1 = bot_y - i * 14
            y2 = y1 - 8
            if y2 < 40:
                break
            thickness = max(1, 4 - i // 3)
            alpha = max(0.3, 1.0 - i * 0.06)
            cv2.line(overlay, (cx, y1), (cx, y2), color, thickness)
            # Growing dot at each dash
            r = max(2, 5 - i // 3)
            cv2.circle(overlay, (cx, (y1 + y2) // 2), r, glow, -1)

        # Arrow label
        cv2.putText(overlay, "Go Straight", (cx - 45, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        cv2.putText(overlay, "Go Straight", (cx - 45, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)

    elif action == "left":
        # Curved dots going left from bottom center
        target_panel = panel in ("center", "left")
        if target_panel:
            for i in range(10):
                t = i / 9.0
                # Bezier-like curve: bottom-center to left
                px = int(cx - t * t * (w * 0.4))
                py = int(bot_y - t * (h * 0.55))
                if py < 30 or px < 5:
                    break
                r = max(2, 6 - i // 2)
                cv2.circle(overlay, (px, py), r, color, -1)
                cv2.circle(overlay, (px, py), r + 2, glow, 1)

            cv2.putText(overlay, "Change Left", (10, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
            cv2.putText(overlay, "Change Left", (10, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)

    elif action == "right":
        # Curved dots going right from bottom center
        target_panel = panel in ("center", "right")
        if target_panel:
            for i in range(10):
                t = i / 9.0
                px = int(cx + t * t * (w * 0.4))
                py = int(bot_y - t * (h * 0.55))
                if py < 30 or px > w - 5:
                    break
                r = max(2, 6 - i // 2)
                cv2.circle(overlay, (px, py), r, color, -1)
                cv2.circle(overlay, (px, py), r + 2, glow, 1)

            cv2.putText(overlay, "Change Right", (w - 140, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
            cv2.putText(overlay, "Change Right", (w - 140, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)

    elif action == "backward":
        # Dots going down
        if panel == "center":
            for i in range(6):
                y = bot_y - 10 + i * 12
                if y > h - 5:
                    break
                cv2.circle(overlay, (cx, y), 4, color, -1)
            cv2.putText(overlay, "Reverse", (cx - 30, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)

    # Blend overlay with original for slight transparency
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
    return frame


# ══════════════════════════════════════════════════════════════════
# DISPLAY — stitch 3 feeds + overlay
# ══════════════════════════════════════════════════════════════════

def draw_display(frames, action, goal, vlm_raw, vlm_reason, cycle, latency,
                 cli_active, cli_buffer, voice_active, override,
                 blind_spot_active, last_seen_panel, goal_mode="approach",
                 direct_queue=None):
    """Stitch the 3 camera frames side by side + overlay."""

    # Get frames, handle missing
    left = frames.get("left")
    center = frames.get("center")
    right = frames.get("right")

    # Make all same height
    target_h = 240
    resized = []
    for f in [left, center, right]:
        if f is not None:
            h, w = f.shape[:2]
            scale = target_h / h
            r = cv2.resize(f, (int(w * scale), target_h))
            resized.append(r)
        else:
            # Black placeholder
            placeholder = np.zeros((target_h, 320, 3), dtype=np.uint8)
            cv2.putText(placeholder, "NO FEED", (80, target_h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            resized.append(placeholder)

    disp = np.hstack(resized)
    h, w = disp.shape[:2]

    # Panel dividers
    pw = resized[0].shape[1]
    cv2.line(disp, (pw, 0), (pw, h), (0, 255, 255), 1)
    cv2.line(disp, (pw * 2, 0), (pw * 2, h), (0, 255, 255), 1)

    # Panel labels
    cv2.putText(disp, "LEFT", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
    cv2.putText(disp, "CENTER", (pw + 10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
    cv2.putText(disp, "RIGHT", (pw * 2 + 10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

    # Action
    ac = (255, 165, 0) if override else ((0, 200, 255) if blind_spot_active else (0, 255, 255))
    cv2.putText(disp, f"ACTION: {action.upper()}", (20, h - 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, ac, 2)

    # Goal / Mode label
    g = goal if goal else "none"
    if goal_mode == "direct":
        goal_color = (0, 200, 255)   # orange for direct
        mode_label = f"DIRECT: {action.upper()}"
    elif goal_mode == "avoid":
        goal_color = (0, 128, 255)
        mode_label = f"AVOID: {g}"
    elif goal_mode == "around":
        goal_color = (255, 200, 0)   # cyan for around
        mode_label = f"AROUND: {g}"
    else:
        goal_color = (0, 255, 0)
        mode_label = f"GOAL: {g}"
    cv2.putText(disp, mode_label, (20, h - 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, goal_color, 1)

    # Sequence queue indicator
    if goal_mode == "direct" and direct_queue:
        queue_str = action.upper() + " → " + " → ".join(q.upper() for q in direct_queue)
        cv2.putText(disp, f"SEQ: {queue_str}", (pw + 10, h - 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 255), 1)

    # VLM info
    if vlm_raw:
        cv2.putText(disp, f"VLM: {vlm_raw[:50]}", (20, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 0), 1)

    # Chain-of-thought reasoning
    if vlm_reason:
        # Semi-transparent background for reasoning
        cv2.rectangle(disp, (0, 35), (w, 60), (20, 20, 20), -1)
        cv2.putText(disp, f"WHY: {vlm_reason[:80]}", (10, 53),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 255, 180), 1)

    # Blind spot indicator
    if blind_spot_active:
        cv2.putText(disp, f"BLIND SPOT (last: {last_seen_panel})",
                    (w // 2 - 100, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 2)

    # CLI overlay
    if cli_active:
        cv2.rectangle(disp, (0, h - 40), (w, h), (30, 30, 30), -1)
        cv2.putText(disp, f"CLI> {cli_buffer}_", (20, h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    # Voice overlay
    if voice_active:
        cv2.putText(disp, "LISTENING...", (w // 2 - 60, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)

    # No goal and no direct command
    if not goal and not cli_active and not voice_active and goal_mode != "direct":
        cv2.rectangle(disp, (0, h // 2 - 20), (w, h // 2 + 20), (30, 30, 30), -1)
        cv2.putText(disp, "Press 1 (CLI) or 2 (Voice) to set goal or command",
                    (w // 2 - 240, h // 2 + 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 180, 255), 1)

    # Bottom info
    cv2.putText(disp, f"EgoNav | Cyc:{cycle} {latency:.0f}ms | 1:CLI 2:Voice 3:Fast 4:Slow Q:Quit",
                (10, h - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.28, (100, 100, 100), 1)

    return disp


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════

def main():

    print("=" * 55)
    print("  EgoNav — Hybrid VLA Navigation System")
    print("=" * 55)
    print(f"  Cameras: {CAM_LEFT}")
    print(f"           {CAM_CENTER}")
    print(f"           {CAM_RIGHT}")
    print(f"  Robot:   {ROBOT_URL}")
    print(f"  VLM:     {VLM_MODEL} ({VLM_API_FORMAT})")
    print(f"           {VLM_API_URL}")
    print()
    print("  Controls:")
    print("    1     → CLI goal")
    print("    2     → Voice goal")
    print("    3     → FAST mode (YOLO ~50ms)")
    print("    4     → SLOW mode (VLM ~5s)")
    print("    0     → Clear goal")
    print("    WASD  → Manual override")
    print("    7/8/9 → Toggle camera blind spot (L/C/R)")
    print("    Q     → Quit")
    print("=" * 55)
    print()

    # Test robot
    try:
        r = session.get(f"{ROBOT_URL}/api/status", timeout=2)
        print(f"  Robot: {'connected' if r.status_code == 200 else 'warning'}")
    except:
        print("  Robot: not reachable (continuing)")

    # Pre-load VLM into VRAM (runs while cameras connect)
    warmup_vlm()

    # Start 3 cameras
    print("  Starting cameras...")
    cam_left   = Camera(CAM_LEFT, "left")
    cam_center = Camera(CAM_CENTER, "center")
    cam_right  = Camera(CAM_RIGHT, "right")

    cams = {"left": cam_left, "center": cam_center, "right": cam_right}
    for name, cam in cams.items():
        ok = cam.start()
        print(f"    {name}: {'✓' if ok else '✗ FAILED'}")

    # Need at least center camera
    if cam_center._frame is None:
        print("ERROR: Center camera not connected")
        return

    # Components
    ego = EgoState(max_history=EGO_HISTORY_SIZE)
    ego.reset()
    vlm = StatefulVLM(ego, VLM_MODEL, VLM_API_URL, VLM_TIMEOUT, LOG_DIR, session,
                       api_format=VLM_API_FORMAT,
                       img_width=VLM_IMG_WIDTH,
                       num_ctx=VLM_NUM_CTX,
                       num_predict=VLM_NUM_PREDICT)
    yolo = YOLOMultiCam()
    blind = BlindSpotHandler()
    logger = Logger()
    inp = InputHandler()
    inp.start()

    cycle = 0
    last_action = "stop"
    consecutive_turns = 0
    vlm_last_call_id = 0
    vlm_acted = True
    around_center_hits = 0
    approach_center_hits = 0
    yolo_last_ms = 0
    vlm_forward_count = 0
    VLM_FORWARD_LIMIT = 4   # stop after 4 consecutive VLM-forward results (~16s)
    total_fwd_count = 0
    TOTAL_FWD_LIMIT = 35       # 35 × 0.08s = 2.8s of total forward across goal

    # Two-phase approach state
    approach_phase = "approach"   # "approach" | "final_stop"
    final_stop_steps = 0
    final_stop_cycle = 0
    prev_goal = None           # for goal-change detection
    around_last_panel = None   # track which side object was for bypass direction

    mode_label = "⚡ FAST (YOLO)" if DEFAULT_INFERENCE_MODE == "fast" else "🧠 SLOW (VLM)"
    print(f"\n  Ready. Mode: {mode_label}. Press 1 or 2 to set a goal.\n")

    try:
        while True:
            cycle += 1

            # ── 1. GET FRAMES ──
            frames = {}
            latest_time = 0
            for name, cam in cams.items():
                f, t = cam.get()
                frames[name] = f
                latest_time = max(latest_time, t)

            # Null out disabled cameras (manual blind spots)
            disabled = state.get("disabled_cameras", set()) if 'state' in dir() else set()

            if frames["center"] is None:
                time.sleep(0.1)
                continue

            # ── 2. INPUT ──
            state = inp.get_state()
            if state["quit"]:
                print("\n  ⏹ Quit requested.")
                break

            # Apply disabled cameras AFTER getting state
            for cam_name in state.get("disabled_cameras", set()):
                frames[cam_name] = None

            goal = state["goal"]
            goal_mode = state["goal_mode"]
            direct_action = state["direct_action"]
            cli_active = state["cli_active"]
            voice_active = state["voice_active"]
            override = state["override"]
            inference_mode = state.get("inference_mode", DEFAULT_INFERENCE_MODE)

            # ── GOAL-CHANGE DETECTION ──
            # When user switches target (bottle→human), clear all stale state
            if goal != prev_goal:
                if prev_goal is not None and goal is not None:
                    print(f"\n  🔄 GOAL CHANGED: {prev_goal} → {goal} — resetting state")
                blind.reset()
                yolo.clear_cache()
                consecutive_turns = 0
                around_center_hits = 0
                approach_center_hits = 0
                vlm_forward_count = 0
                total_fwd_count = 0
                approach_phase = "approach"
                final_stop_steps = 0
                final_stop_cycle = 0
                around_last_panel = None
                vlm_acted = True
                prev_goal = goal

            # ── AVOID MODE: invert actions ──
            AVOID_MAP = {"forward": "left", "left": "right", "right": "left", "stop": "forward"}

            # ── PATH OBSTACLE CHECK ──
            # When approaching a goal, detect non-goal objects blocking center path.
            # Returns (dodge_action, obstacle_info) or (None, None) if path is clear.
            def check_path_obstacle(yolo_dets, goal, frames):
                """Position-based obstacle detection.
                Blocks on ANY object whose bbox center is in the lower-center
                danger zone, regardless of class. Unknown small objects included.
                """
                center_dets = yolo_dets.get("center", [])
                if not center_dets:
                    return None, None

                ref_frame = frames.get("center")
                if ref_frame is None:
                    return None, None

                frame_h, frame_w = ref_frame.shape[:2]
                frame_area = frame_h * frame_w

                # Build skip set: goal + visual family
                goal_lower = goal.lower().strip() if goal else ""
                GOAL_SYNONYMS = {
                    "chair": {"chair","sofa","couch","bench","seat","stool","train","bus","car"},
                    "bottle": {"bottle","cup","wine glass","vase","flask"},
                    "door":   {"door","doorway"},
                    "person": {"person","man","woman","human"},
                    "box":    {"box","suitcase","backpack","luggage"},
                    "table":  {"table","desk","dining table"},
                    "tv":     {"tv","monitor","laptop","screen"},
                }
                skip_classes = {goal_lower} | GOAL_SYNONYMS.get(goal_lower, set())

                # Danger zone in lower-center of frame
                # Horizontal: 20%–80% of width (center column)
                # Vertical:   35%–100% of height (lower 65%)
                zone_x1 = frame_w * 0.20
                zone_x2 = frame_w * 0.80
                zone_y1 = frame_h * 0.15   # wider zone — catches bottle at cy≈145

                blocking = []
                for d in center_dets:
                    if d["class"] in skip_classes:
                        continue

                    # Class whitelist (if set)
                    if PATH_OBSTACLE_CLASSES and d["class"] not in PATH_OBSTACLE_CLASSES:
                        continue

                    # position-based check using bbox center
                    bx = d.get("cx", frame_w / 2)
                    by = d.get("cy", frame_h / 2)
                    in_zone = (zone_x1 <= bx <= zone_x2) and (by >= zone_y1)
                    ratio = d["area"] / frame_area if frame_area > 0 else 0

                    # Block if: in danger zone AND area > threshold
                    if in_zone and ratio > PATH_OBSTACLE_AREA_RATIO:
                        blocking.append((d, ratio, bx, by))

                    # Also block unknown/small objects by position alone
                    # Even if class is not in whitelist, if it's big enough in zone → block
                    elif in_zone and ratio > PATH_OBSTACLE_AREA_RATIO * 2:
                        blocking.append((d, ratio, bx, by))

                if not blocking:
                    return None, None

                blocking.sort(key=lambda x: x[1], reverse=True)
                biggest, obs_ratio, bx, by = blocking[0]

                # Dodge toward less cluttered side
                left_area  = sum(d["area"] for d in yolo_dets.get("left", []))
                right_area = sum(d["area"] for d in yolo_dets.get("right", []))
                dodge = "left" if left_area <= right_area else "right"

                info = (f"{biggest['class']} in lower-center "
                        f"({obs_ratio:.1%} @ cx={bx:.0f},cy={by:.0f}) → {dodge}")
                return dodge, info

            def _get_goal_area_ratio(yolo_dets, goal, frames, panel="center"):
                """Return area ratio of goal in given panel (for phase transition)."""
                if not goal or frames.get(panel) is None:
                    return 0.0
                goal_lower = goal.lower().strip()
                GOAL_SYN = {
                    "chair": {"chair","sofa","couch","bench"},
                    "bottle": {"bottle","cup","wine glass","vase"},
                    "table": {"table","desk","dining table"},
                    "person": {"person","man","woman"},
                    "tv": {"tv","monitor","laptop"},
                }
                goal_cls = {goal_lower} | GOAL_SYN.get(goal_lower, set())
                fa = frames[panel].shape[0] * frames[panel].shape[1]
                best = max((d["area"] for d in yolo_dets.get(panel, [])
                            if d["class"] in goal_cls), default=0)
                return best / fa if fa > 0 else 0.0

            # ── 3. PERCEPTION — dual mode ──
            # In fast mode: YOLO runs synchronously this cycle
            # In slow mode: VLM runs async on background thread

            yolo_action = None
            yolo_panel = None
            yolo_reason = ""
            yolo_dets = {}

            if inference_mode == "fast" and goal and goal_mode in ("approach", "avoid", "around") \
                    and not cli_active and not voice_active:
                yolo_action, yolo_panel, yolo_reason, yolo_dets, yolo_last_ms = \
                    yolo.detect_panels(frames, goal)
                is_fresh = (yolo._cycle_counter % YOLO_EVERY_N == 1)
                if is_fresh:
                    if yolo_action:
                        print(f"  [YOLO] {goal} → {yolo_panel} → {yolo_action} ({yolo_last_ms:.0f}ms)")
                    else:
                        # Pure if-else: YOLO not found → search left (no VLM escalation)
                        print(f"  [YOLO] {goal} not found → searching ({yolo_last_ms:.0f}ms)")

            if inference_mode == "slow" and goal and goal_mode in ("approach", "avoid", "around") \
                    and not vlm.running and not cli_active and not voice_active and vlm_acted:
                vlm.request(frames, goal, goal_mode=goal_mode,
                            last_action=last_action,
                            consecutive_turns=consecutive_turns)

            # ── 4. CHECK VLM RESULT (slow mode only) ──
            vlm_action, vlm_raw, vlm_reason, vlm_ms, vlm_panel = vlm.get()
            new_vlm = (vlm.calls > vlm_last_call_id)
            # Discard stale result computed for a DIFFERENT goal
            if new_vlm and vlm.last_goal is not None and vlm.last_goal != goal:
                print(f"  [DISCARD] Stale VLM result for '{vlm.last_goal}' "
                      f"(current goal: '{goal}') — skipping")
                vlm_last_call_id = vlm.calls
                new_vlm = False

            # ── 5. DECIDE ──
            # Unify: pick the active perception result based on mode
            # YOLO fast mode: only act on FRESH inference cycles.
            # On cached cycles, stop (step-by-step).
            if inference_mode == "fast":
                is_fresh_yolo = (yolo._cycle_counter % YOLO_EVERY_N == 1) if goal else False
                perc_action = yolo_action
                perc_panel = yolo_panel
                perc_reason = yolo_reason
                perc_source = "yolo"
                has_new_result = is_fresh_yolo and goal is not None
            else:
                perc_action = vlm_action if new_vlm else None
                perc_panel = vlm_panel if new_vlm else None
                perc_reason = vlm_reason if new_vlm else ""
                perc_source = "vlm"
                has_new_result = new_vlm

            blind_spot_active = False

            if cli_active or voice_active:
                action = "stop"
                source = "input_pause"
            elif goal_mode == "direct" and direct_action:
                action = direct_action
                source = "direct"
            elif not goal and goal_mode != "direct":
                action = "stop"
                source = "no_goal"
            elif override:
                action = override
                source = "keyboard"
            elif has_new_result and inference_mode == "fast":
                # ── FAST MODE: YOLO result available this cycle ──
                if new_vlm:
                    vlm_last_call_id = vlm.calls  # drain stale VLM results

                if perc_action is not None:
                    action = perc_action
                    source = perc_source

                    if goal_mode == "avoid":
                        if perc_panel is not None:
                            action = AVOID_MAP.get(perc_action, "forward")
                            source = "avoid"
                        else:
                            action = "forward"
                            source = "avoid_clear"
                    elif goal_mode == "around":
                        if perc_panel is not None:
                            around_last_panel = perc_panel
                        if perc_panel == "center":
                            around_center_hits += 1
                            if around_center_hits >= AROUND_CENTER_HITS:
                                # Compute area ratio for adaptive bypass
                                frame_h, frame_w = (frames.get("center") or frames.get("left") or frames.get("right")).shape[:2]
                                frame_area = frame_h * frame_w
                                area_ratio = yolo._last_area / frame_area if frame_area > 0 else 0.1
                                bypass_seq = build_around_sequence(area_ratio, around_last_panel or "center")
                                bypass_str = " → ".join(bypass_seq)
                                print(f"\n  🎯 Object '{goal}' centered {around_center_hits}x → ADAPTIVE BYPASS")
                                print(f"      area_ratio={area_ratio:.3f}  seq=[{bypass_str}]")
                                inp.trigger_bypass_sequence(bypass_seq)
                                around_center_hits = 0
                                action = "stop"
                                source = "around_trigger"
                            else:
                                blind.update(perc_panel)
                        elif perc_panel is not None:
                            around_center_hits = 0
                            blind.update(perc_panel)
                        else:
                            around_center_hits = 0
                    else:
                        # APPROACH mode
                        if perc_panel is not None:
                            blind.update(perc_panel)
                            if perc_panel == "center":
                                approach_center_hits += 1
                                if approach_center_hits >= CENTER_CONFIRM:
                                    # Before going forward, check for obstacles in path
                                    dodge, obs_info = check_path_obstacle(yolo_dets, goal, frames)
                                    if dodge:
                                        action = dodge
                                        source = "path_obstacle"
                                        print(f"  ⚠ PATH OBSTACLE: {obs_info}")
                                    else:
                                        action = "forward"
                                        source = perc_source
                                else:
                                    action = "stop"
                                    source = "centering"
                            else:
                                approach_center_hits = 0
                        else:
                            approach_center_hits = 0
                else:
                    # YOLO didn't find goal → check VLM backup
                    if new_vlm and vlm_action is not None:
                        # VLM backup returned a result! Use it.
                        action = vlm_action
                        source = "vlm_backup"
                        vlm_last_call_id = vlm.calls
                        vlm_acted = False
                        if vlm_panel is not None:
                            blind.update(vlm_panel)
                        print(f"  [VLM BACKUP] {goal} → {vlm_panel} → {action}")
                    elif cycle % 10 == 0:
                        # VLM hasn't responded yet — throttled search
                        if consecutive_turns >= 5 and last_action in ("left", "right"):
                            search_dir = "right" if last_action == "left" else "left"
                        else:
                            search_dir = "left"
                        action = search_dir
                        source = "yolo_search"
                    else:
                        action = "stop"
                        source = "waiting"

            elif new_vlm and vlm_action is not None and inference_mode == "slow":
                # ════════════════════════════════════════════════════════════
                # SLOW MODE: VLM gives direction, execute it
                # ════════════════════════════════════════════════════════════
                action = vlm_action
                source = "vlm"
                vlm_last_call_id = vlm.calls
                vlm_acted = False
                # do NOT reset total_fwd_count here — it must accumulate
                # across VLM calls, only goal-change resets it
                ego.record_action(action, source)
                print(f"  [VLM2] → MOVE={action}  reason: {vlm_reason[:60]}")

                # ── YOLO OBSTACLE GUARD (runs every VLM result) ──────────────
                if action == "forward" and goal_mode == "approach" and YOLO_AVAILABLE and yolo.available:
                    _, _, _, obs_dets, _ = yolo.detect_panels(frames, goal)
                    dodge, obs_info = check_path_obstacle(obs_dets, goal, frames)
                    if dodge:
                        action = dodge
                        source = "path_obstacle"
                        vlm_forward_count = 0
                        print(f"  ⚠ YOLO OBSTACLE GUARD: {obs_info}")

                # ── YOLO GOAL-REACHED ────────────────────────────────────────
                if action == "forward" and goal_mode == "approach" and YOLO_AVAILABLE and yolo.available:
                    ga, gp, _, _, _ = yolo.detect_panels(frames, goal)
                    if ga == "stop" and gp == "center":
                        print(f"  🏁 YOLO: {goal} fills frame → STOP")
                        action = "stop"
                        source = "goal_reached"
                        inp._clear_goal()
                        vlm_forward_count = 0

                # ── VLM proximity counter ─────────────────────────────────────
                if action == "forward" and goal_mode == "approach":
                    vlm_forward_count += 1
                    if vlm_forward_count >= VLM_FORWARD_LIMIT:
                        print(f"  🏁 VLM PROXIMITY ({vlm_forward_count} fwds): {goal} → STOP")
                        action = "stop"
                        source = "goal_reached"
                        inp._clear_goal()
                        vlm_forward_count = 0
                else:
                    vlm_forward_count = 0

            else:
                # ════════════════════════════════════════════════════════════
                # MOMENTUM: while VLM is thinking, KEEP MOVING
                # Key fix: use vlm.running (not vlm_acted) to gate momentum
                # ════════════════════════════════════════════════════════════
                if inference_mode == "slow" and goal and goal_mode == "approach"                         and vlm.running                         and last_action in ("forward", "left", "right"):

                    if last_action == "left" or last_action == "right":
                        # Cap turns at 2 momentum cycles, then go forward
                        total_fwd_count = max(0, total_fwd_count - 1)
                        if not hasattr(inp, "_mome_turn_count"):
                            inp._mome_turn_count = 0
                        inp._mome_turn_count += 1
                        if inp._mome_turn_count <= 2:
                            action = last_action
                            source = "momentum"
                        else:
                            action = "forward"
                            source = "momentum"
                            inp._mome_turn_count = 0
                    else:
                        # ── FORWARD MOMENTUM with two-phase approach ──
                        inp._mome_turn_count = 0
                        total_fwd_count += 1
                        final_stop_cycle += 1

                        # ── PHASE CHECK: transition to FINAL STOP mode ──
                        if approach_phase == "approach" and YOLO_AVAILABLE and yolo.available                                 and total_fwd_count % 5 == 0:
                            _, _, _, fsd, _ = yolo.detect_panels(frames, goal)
                            goal_ratio = _get_goal_area_ratio(fsd, goal, frames, "center")
                            if goal_ratio >= APPROACH_TRANSITION_AREA:
                                approach_phase = "final_stop"
                                final_stop_steps = 0
                                final_stop_cycle = 0
                                print(f"  🎯 FINAL STOP MODE: {goal} fills {goal_ratio:.1%} → slowing")

                        if approach_phase == "final_stop":
                            # ── FINAL STOP MODE: slow creep + precise stop ──
                            _, _, _, fsd, _ = yolo.detect_panels(frames, goal)
                            goal_ratio = _get_goal_area_ratio(fsd, goal, frames, "center")

                            if goal_ratio >= FINAL_STOP_AREA:
                                print(f"  🏁 FINAL STOP: {goal} fills {goal_ratio:.1%} → STOP")
                                action = "stop"
                                source = "goal_reached"
                                inp._clear_goal()
                                approach_phase = "approach"
                                total_fwd_count = 0
                            elif final_stop_steps >= FINAL_STOP_MAX_STEPS:
                                print(f"  🏁 FINAL STOP TIMEOUT ({final_stop_steps} steps): {goal} → STOP")
                                action = "stop"
                                source = "goal_reached"
                                inp._clear_goal()
                                approach_phase = "approach"
                                total_fwd_count = 0
                            elif final_stop_cycle % FINAL_STOP_STEP_INTERVAL == 0:
                                # Slow creep: move forward every N cycles only
                                action = "forward"
                                source = "final_stop_creep"
                                final_stop_steps += 1
                            else:
                                action = "stop"
                                source = "final_stop_pause"

                        elif total_fwd_count >= TOTAL_FWD_LIMIT:
                            # ── FALLBACK: hard cap (approach phase took too long) ──
                            print(f"  🏁 TOTAL FWD CAP ({total_fwd_count}): {goal} → STOP")
                            action = "stop"
                            source = "goal_reached"
                            inp._clear_goal()
                            total_fwd_count = 0
                            approach_phase = "approach"

                        elif YOLO_AVAILABLE and yolo.available and total_fwd_count % 3 == 0:
                            # ── APPROACH PHASE: YOLO obstacle + goal area check ──
                            _, _, _, obs_dets, _ = yolo.detect_panels(frames, goal)
                            # debug: print center detections to verify YOLO sees obstacle
                            ctr = obs_dets.get("center", [])
                            if ctr:
                                det_dbg = ", ".join(f"{d['class']}({d.get('conf',d.get('confidence',0)):.2f} cy={d.get('cy',0):.0f})" for d in ctr[:4])
                                print(f"  [YOLO-CTR] {det_dbg}")
                            dodge, obs_info = check_path_obstacle(obs_dets, goal, frames)
                            if dodge:
                                action = dodge
                                source = "path_obstacle"
                                total_fwd_count = 0
                                approach_phase = "approach"
                                print(f"  ⚠ MOMENTUM YOLO GUARD: {obs_info}")
                            else:
                                # Also check standard goal-reached
                                ga, gp, _, _, _ = yolo.detect_panels(frames, goal)
                                if ga == "stop" and gp == "center":
                                    print(f"  🏁 MOMENTUM YOLO: {goal} fills frame → STOP")
                                    action = "stop"
                                    source = "goal_reached"
                                    inp._clear_goal()
                                    total_fwd_count = 0
                                else:
                                    action = "forward"
                                    source = "momentum"
                        else:
                            action = "forward"
                            source = "momentum"

                elif inference_mode == "fast":
                    action = "stop"
                    source = "waiting"
                elif goal_mode == "avoid":
                    action = "forward" if goal and not vlm.running else "stop"
                    source = "avoid_cruise" if action == "forward" else "waiting"
                elif goal_mode == "around":
                    if goal and not vlm.running:
                        bs_action, bs_active = blind.get_blind_spot_action()
                        if bs_active and not vlm_acted:
                            action = bs_action
                            source = "blind_spot"
                            blind_spot_active = True
                        else:
                            action = "stop"
                            source = "waiting"
                    else:
                        action = "stop"
                        source = "waiting"
                else:
                    bs_action, bs_active = blind.get_blind_spot_action()
                    if bs_active and not vlm_acted:
                        action = bs_action
                        source = "blind_spot"
                        blind_spot_active = True
                    else:
                        action = "stop"
                        source = "waiting"

            # ── 6. SAFETY OVERRIDES ──
            # These run AFTER the decision but BEFORE execution.
            # They can veto dangerous actions regardless of inference mode.

            person_safety_triggered = False

            # A. PERSON SAFETY — always-on, both modes
            # Check all YOLO detections for nearby people
            if YOLO_AVAILABLE and yolo.available and action not in ("stop",) \
                    and goal not in ("person", "human", "man", "woman"):  # don't safety-stop when approaching person
                # Use latest YOLO dets if available, or run quick check
                check_dets = yolo_dets if yolo_dets else {}
                for cam_name, dets in check_dets.items():
                    for d in dets:
                        if d["class"] in SAFETY_OBJECTS:
                            ref_frame = frames.get(cam_name)
                            if ref_frame is not None:
                                fa = ref_frame.shape[0] * ref_frame.shape[1]
                                # require high confidence + large area to avoid chair=person false triggers
                                if d["area"] / fa > SAFETY_AREA_RATIO and d.get("conf", 1.0) > 0.75:
                                    print(f"  🚨 PERSON SAFETY: {d['class']} in {cam_name} "
                                          f"({d['area']/fa:.1%} of frame) → STOP")
                                    action = "stop"
                                    source = "person_safety"
                                    person_safety_triggered = True
                                    break
                    if person_safety_triggered:
                        break

            # ── 7. EXECUTE ──
            send_action(action)

            if source in ("vlm", "vlm_backup", "yolo", "blind_spot", "avoid", "avoid_clear", "avoid_cruise",
                          "direct", "centering", "yolo_search", "person_safety", "path_obstacle"):
                vlm_acted = True
                ego.record_action(action, source)

            if source in ("vlm", "vlm_backup", "yolo", "blind_spot", "avoid", "centering", "yolo_search", "path_obstacle") and action in ("left", "right"):
                if action == last_action:
                    consecutive_turns += 1
                else:
                    consecutive_turns = 1
            elif source in ("vlm", "vlm_backup", "yolo", "avoid", "avoid_clear", "centering", "person_safety"):
                consecutive_turns = 0

            if source in ("vlm", "vlm_backup", "yolo", "blind_spot", "avoid", "avoid_clear", "avoid_cruise",
                          "centering", "yolo_search", "person_safety", "path_obstacle"):
                last_action = action

            # ── 8. LOG ──
            now = time.time()
            latency = (now - latest_time) * 1000 if latest_time > 0 else 0
            logger.log({
                "cycle": cycle,
                "ts": now,
                "goal": goal,
                "goal_mode": goal_mode,
                "inference_mode": inference_mode,
                "action": action,
                "source": source,
                "vlm_raw": vlm_raw if inference_mode == "slow" else yolo_reason,
                "vlm_reason": vlm_reason if inference_mode == "slow" else yolo_reason,
                "vlm_panel": vlm_panel if inference_mode == "slow" else yolo_panel,
                "blind_spot": blind_spot_active,
                "person_safety": person_safety_triggered,
                "path_obstacle": source == "path_obstacle",
                "vlm_calls": vlm.calls,
                "yolo_calls": yolo.calls,
                "consecutive_turns": consecutive_turns,
                "lat_ms": round(latency, 1),
            })

            # ── 9. CONSOLE ──
            mode_icon = "⚡" if inference_mode == "fast" else "🧠"
            mode_tag = "🛡" if goal_mode == "avoid" else ("➡" if goal_mode == "direct" else ("🔄" if goal_mode == "around" else "★"))
            if source == "direct":
                if cycle % 5 == 0:
                    print(f"[{cycle:>4}] ➡ DIRECT → {action:>7}")
            elif source == "around_trigger":
                pass
            elif source == "centering":
                pass
            elif source in ("yolo", "smart_search"):
                panel_str = yolo_panel or "none"
                print(f"[{cycle:>4}] {mode_icon} YOLO → {action:>7}  "
                      f"panel={panel_str}  ({yolo_last_ms:.0f}ms)")
            elif source == "person_safety":
                pass  # already printed in safety block
            elif source == "path_obstacle":
                pass  # already printed inline
            elif source in ("vlm", "avoid", "avoid_clear"):
                print(f"[{cycle:>4}] {mode_icon} VLM → {action:>7}  [{ego.get_ego_summary()}]  "
                      f"\"{vlm_raw[:30]}\" ({latency:.0f}ms)")
                if vlm_reason:
                    print(f"       💬 {vlm_reason[:70]}")
            elif source == "avoid_cruise":
                if cycle % 15 == 0:
                    print(f"[{cycle:>4}]   ...cruising (avoid mode, path clear)...")
            elif source == "blind_spot":
                print(f"[{cycle:>4}] ⚡ BLIND SPOT → {action:>7}  "
                      f"(continuing from {blind.last_seen_panel})")
            elif source.startswith("wait"):
                if cycle % 15 == 0:
                    mode_str = "YOLO" if inference_mode == "fast" else "VLM"
                    print(f"[{cycle:>4}]   ...waiting for {mode_str}...")
            elif source not in ("waiting",):
                g = goal or "—"
                print(f"[{cycle:>4}] [{source[:4].upper()}] g:{g:<10} → {action:>7}")

            # ── 10. DRAW TRAJECTORY OVERLAY on frames ──
            display_frames = {}
            for name, f in frames.items():
                if f is not None:
                    df = f.copy()
                    if action != "stop":
                        df = draw_trajectory_overlay(df, action, panel=name)
                    # Centering crosshair on center panel
                    if source == "centering" and name == "center":
                        ch, cw = df.shape[:2]
                        cx, cy = cw // 2, ch // 2
                        # Crosshair
                        cv2.line(df, (cx - 20, cy), (cx + 20, cy), (0, 255, 0), 1)
                        cv2.line(df, (cx, cy - 20), (cx, cy + 20), (0, 255, 0), 1)
                        cv2.circle(df, (cx, cy), 15, (0, 255, 0), 1)
                        # Label
                        cv2.putText(df, "CENTERING...", (cx - 45, cy - 25),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                    display_frames[name] = df
                else:
                    display_frames[name] = None

            # ── 11. DISPLAY ──
            disp = draw_display(display_frames, action, goal, vlm_raw, vlm_reason,
                                cycle, latency,
                                cli_active, state["cli_buffer"],
                                voice_active, override,
                                blind_spot_active, blind.last_seen_panel,
                                goal_mode,
                                direct_queue=state.get("direct_queue"))
            cv2.imshow("EgoNav — Navigation", disp)

            # ── 11. KEYS ──
            key = cv2.waitKey(1) & 0xFF
            inp.handle_cv2_key(key)

            # ── 12. PACE ──
            time.sleep(CYCLE_DELAY)

    except KeyboardInterrupt:
        print("\n  Interrupted")
    finally:
        print("\n  Stopping...")
        emergency_stop()
        inp.stop()
        for cam in cams.values():
            cam.stop()
        cv2.destroyAllWindows()
        logger.close()
        session.close()
        print(f"  VLM calls: {vlm.calls}")
        print(f"  YOLO calls: {yolo.calls}")
        print("  Done.")


if __name__ == "__main__":
    main()
