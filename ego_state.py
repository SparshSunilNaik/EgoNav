"""
ego_state.py - Stateful Ego-Context Memory for EgoNav
=====================================================
Provides EgoState (robot memory) and StatefulVLM (VLM inference
with panoramic stitching). Supports Ollama and llama.cpp backends.
"""

import time
import threading
import os
import cv2
import base64
import numpy as np
import requests


class EgoState:
    """Maintains the robot's self-model across VLM calls.
    This is the 'memory' that makes the API stateful.
    """

    def __init__(self, max_history=6):
        self._lock = threading.Lock()
        self.max_history = max_history
        self.observations = []
        self.action_history = []
        self.chat_history = []
        self.goal_start_time = None
        self.total_searches = 0
        self.total_detections = 0

    def reset(self):
        """Called when the goal changes."""
        with self._lock:
            self.observations.clear()
            self.action_history.clear()
            self.chat_history.clear()
            self.goal_start_time = time.time()
            self.total_searches = 0
            self.total_detections = 0

    def record_observation(self, panel, found, reason):
        """Record what the VLM saw in a panel."""
        with self._lock:
            entry = {
                "t": round(time.time() - (self.goal_start_time or time.time()), 1),
                "panel": panel,
                "found": found,
                "reason": reason[:60],
            }
            self.observations.append(entry)
            if len(self.observations) > 12:
                self.observations.pop(0)
            if found:
                self.total_detections += 1
            else:
                self.total_searches += 1

    def record_action(self, action, source):
        """Record what action the robot took."""
        with self._lock:
            self.action_history.append({
                "t": round(time.time() - (self.goal_start_time or time.time()), 1),
                "action": action,
                "source": source,
            })
            if len(self.action_history) > 10:
                self.action_history.pop(0)

    def add_chat_turn(self, user_msg, assistant_response, images=None):
        """Add a query-response pair to the conversation history."""
        with self._lock:
            user_entry = {"role": "user", "content": user_msg}
            if images:
                user_entry["images"] = images
            self.chat_history.append(user_entry)
            self.chat_history.append({
                "role": "assistant",
                "content": assistant_response,
            })
            # Trim to max_history pairs (each pair = 2 messages)
            max_msgs = self.max_history * 2
            if len(self.chat_history) > max_msgs:
                self.chat_history = self.chat_history[-max_msgs:]

    def trim_history(self):
        """Aggressively trim history to reduce inference load on timeout."""
        with self._lock:
            if len(self.chat_history) > 4:
                removed = len(self.chat_history) - 4
                self.chat_history = self.chat_history[-4:]
                print(f"  [TRIM] Removed {removed} messages from history")

    def build_system_prompt(self, goal, goal_mode, last_action, consecutive_turns):
        """Compact action-output system prompt."""
        with self._lock:
            if goal_mode == "avoid":
                task_desc = f'AVOID "{goal}" — move away from it.'
            elif goal_mode == "around":
                task_desc = f'Go AROUND "{goal}" — treat it as a static obstacle.'
            else:
                task_desc = f'APPROACH "{goal}" — navigate toward it, stop when you arrive.'

            recent_actions = [a["action"] for a in self.action_history[-6:]]
            hist_str = ">".join(recent_actions) if recent_actions else "none"

            # Count consecutive forward steps in recent history
            fwd_count = 0
            for a in reversed(self.action_history):
                if a["action"] == "forward":
                    fwd_count += 1
                else:
                    break

            proximity_hint = ""
            if goal_mode == "approach" and fwd_count >= 4:
                proximity_hint = (
                    f"\nNOTE: You have moved FORWARD {fwd_count} consecutive times toward \"{goal}\"."
                    f" The robot is likely very close to or at the goal now."
                    f" If the goal is clearly visible right in front of you, say MOVE: stop."
                )

            return (
                "You are a robot navigation brain. You see a STITCHED panoramic image: LEFT | CENTER | RIGHT panels.\n"
                f"Task: {task_desc}\n"
                f"Recent moves: {hist_str} | Turn-streak: {consecutive_turns}\n"
                f"{proximity_hint}\n\n"
                "OBSTACLE AWARENESS: Look carefully at the LOWER HALF of the CENTER panel.\n"
                "If ANY object other than the goal is visible there, the path is BLOCKED — even if the goal is visible behind it.\n\n"
                "RULES (follow strictly):\n"
                f"1. Goal in CENTER, lower-center is EMPTY (no other objects) → MOVE: forward\n"
                f"2. Goal in CENTER BUT another object is between you and it (lower-center) → MOVE: left or right to dodge\n"
                f"3. Goal on LEFT → MOVE: left\n"
                f"4. Goal on RIGHT → MOVE: right\n"
                f"5. Goal NOT visible anywhere → MOVE: left (search)\n"
                f"6. You are right in front of the goal, very close → MOVE: stop\n"
                f"7. Stuck turning {consecutive_turns}+ times same direction → try opposite\n\n"
                "Reply in EXACTLY 2 lines:\n"
                "THINK: <which panel is goal in, is lower-center clear?>\n"
                "MOVE: forward OR left OR right OR stop"
            )

    def build_fast_prompt(self, goal):
        """Compact action prompt — references stitched panoramic layout."""
        # Build a goal-specific description hint
        GOAL_HINTS = {
            "chair":  "a chair (seat, backrest, legs — often rectangular)",
            "bottle": "a bottle (tall cylindrical container, possibly on the floor)",
            "person": "a person (human figure, face, body)",
            "door":   "a door (rectangular panel in a wall, doorframe)",
            "table":  "a table (flat surface on legs)",
            "tv":     "a TV or monitor (flat rectangular screen)",
            "bag":    "a bag or backpack",
            "cup":    "a cup or mug",
            "box":    "a box or cardboard container",
            "car":    "a car or vehicle",
        }
        goal_hint = GOAL_HINTS.get(goal.lower().strip(), f"the object called '{goal}'")
        return (
            f'You are looking at a stitched panoramic image from 3 robot cameras.\n'
            f'Panels (separated by CYAN lines): LEFT | CENTER | RIGHT.\n\n'
            f'Your GOAL: find {goal_hint}.\n'
            f'Look carefully at ALL panels. Do NOT confuse it with other objects.\n'
            f'A chair is NOT a person. A bottle is NOT a person.\n\n'
            f'THINK step-by-step:\n'
            f'  1. Which panel contains {goal_hint}?\n'
            f'  2. Is the path in CENTER clear of other objects?\n'
            f'  3. What is the correct move?\n\n'
            f'Reply in EXACTLY 2 lines:\n'
            f'THINK: <your reasoning>\n'
            f'MOVE: forward (goal in CENTER+clear) OR left (goal in LEFT) OR right (goal in RIGHT) OR stop (arrived)'
        )

    def get_chat_history(self):
        """Return a copy of the current chat history."""
        with self._lock:
            return list(self.chat_history)

    def get_ego_summary(self):
        """Return a short text summary of ego state for display."""
        with self._lock:
            n_obs = len(self.observations)
            n_hist = len(self.chat_history) // 2
            elapsed = round(time.time() - (self.goal_start_time or time.time()), 1)
            return (
                f"mem:{n_hist}/{self.max_history} obs:{n_obs} "
                f"det:{self.total_detections} srch:{self.total_searches} "
                f"t:{elapsed}s"
            )


# ══════════════════════════════════════════════════════════════════
# Per-camera check order and action map
# ══════════════════════════════════════════════════════════════════
PANEL_CHECK_ORDER = ["center", "left", "right"]
PANEL_ACTION_MAP = {"center": "forward", "left": "left", "right": "right"}


class StatefulVLM:
    """Stateful VLM that checks each camera panel for the goal object.
    Uses /api/chat with rolling conversation history (ego state).
    Priority: center (forward) > left (left) > right (right).
    If not found anywhere -> search.
    """

    def __init__(self, ego, model, api_url, timeout, log_dir, session,
                 api_format="ollama", img_width=None, num_ctx=4096,
                 num_predict=60):
        self._lock = threading.Lock()
        self.running = False
        self.last_action = None
        self.last_raw = ""
        self.last_reason = ""
        self.last_ms = 0.0
        self.last_panel = None
        self.last_goal = None
        self.calls = 0
        self.ego = ego
        self.model = model
        self.api_url = api_url
        self.timeout = timeout
        self.log_dir = log_dir
        self.session = session
        self.api_format = api_format  # "ollama" or "llamacpp"
        self.img_width = img_width    # resize width (None = no resize)
        self.num_ctx = num_ctx        # context window size
        self.num_predict = num_predict  # max output tokens

    def request(self, panels, goal, goal_mode="approach",
                last_action="stop", consecutive_turns=0):
        with self._lock:
            if self.running:
                return
            self.running = True
        panels_copy = {k: v.copy() for k, v in panels.items() if v is not None}
        threading.Thread(
            target=self._run,
            args=(panels_copy, goal, goal_mode, last_action, consecutive_turns),
            daemon=True
        ).start()

    def reset(self):
        """Clear cached VLM results. Call when goal changes."""
        with self._lock:
            self.last_action = None
            self.last_raw = ""
            self.last_reason = ""
            self.last_ms = 0.0
            self.last_panel = None
            self.last_goal = None

    def get(self):
        with self._lock:
            return self.last_action, self.last_raw, self.last_reason, \
                   self.last_ms, self.last_panel

    # ──────────────────────────────────────────────────────────────
    # Main dispatch
    # ──────────────────────────────────────────────────────────────

    def _run(self, panels, goal, goal_mode, last_action, consecutive_turns):
        try:
            if self.api_format == "llamacpp":
                self._run_percamera(panels, goal, goal_mode,
                                    last_action, consecutive_turns)
            else:
                self._run_panoramic(panels, goal, goal_mode,
                                    last_action, consecutive_turns)
        except Exception as e:
            print(f"  [ERR] VLM error: {e}")
        finally:
            with self._lock:
                self.running = False

    # ──────────────────────────────────────────────────────────────
    # llama.cpp: 3× per-camera "describe what you see" calls
    # ──────────────────────────────────────────────────────────────

    def _run_percamera(self, panels, goal, goal_mode,
                       last_action, consecutive_turns):
        """Ask each camera to list objects it sees.
        We never mention the goal — the model must independently name it.
        Then we check if any synonym of the goal appears in the output.
        Priority: center → left → right. Stop at first match.
        """
        t0 = time.time()

        if self.calls < 3:
            os.makedirs(self.log_dir, exist_ok=True)
            for name in panels:
                path = os.path.join(self.log_dir,
                                    f"debug_{self.calls}_{name}.jpg")
                cv2.imwrite(path, panels[name])

        prompt = (
            "List the main objects visible in this image. "
            "Be brief and specific. Only mention objects you are "
            "certain about.\n\n"
            "Objects:"
        )

        goal_keywords = self._get_goal_keywords(goal)

        found_panel = None
        found_reason = ""
        panel_results = {}

        for panel_name in PANEL_CHECK_ORDER:
            if panel_name not in panels:
                panel_results[panel_name] = "no feed"
                continue

            frame = panels[panel_name]
            _, buf = cv2.imencode('.jpg', frame,
                                  [cv2.IMWRITE_JPEG_QUALITY, 60])
            img_b64 = base64.b64encode(buf).decode()

            payload = {
                "prompt": prompt,
                "image_data": [{"data": img_b64, "id": 0}],
                "n_predict": 40,
                "temperature": 0.0,
                "stop": ["\n\n", "\nQuestion", "\nList"],
            }

            try:
                resp = self.session.post(self.api_url, json=payload,
                                         timeout=self.timeout)
                resp.raise_for_status()
                raw = resp.json().get("content", "").strip()
            except Exception as e:
                raw = f"error: {e}"

            panel_results[panel_name] = raw
            is_match = self._check_goal_in_description(raw, goal_keywords)

            elapsed = (time.time() - t0) * 1000
            tag = "✓ MATCH" if is_match else "✗ no"
            print(f"    [{panel_name:>6}] {tag}: \"{raw[:50]}\" "
                  f"({elapsed:.0f}ms)")

            if is_match:
                found_panel = panel_name
                found_reason = raw
                break

        total_ms = (time.time() - t0) * 1000
        self._store_result(panels, goal, found_panel, found_reason,
                           panel_results, total_ms,
                           last_action, consecutive_turns)

    # ──────────────────────────────────────────────────────────────
    # Ollama: single panoramic call
    # ──────────────────────────────────────────────────────────────

    def _run_panoramic(self, panels, goal, goal_mode,
                       last_action, consecutive_turns):
        """Single Ollama call with 3 separate images (no stitching).
        Each camera is a separate image — the prompt tells the model
        which image is LEFT, CENTER, RIGHT. No label confusion.
        """
        # ── Stitch 3 cameras into ONE labeled panoramic for VLM ──────────────
        stitched = self._stitch_for_vlm(panels)
        if stitched is None:
            print("  [ERR] Could not stitch panoramic")
            with self._lock:
                self.running = False
            return

        # Downscale to keep tokens low (target: ~672px wide)
        target_w = (self.img_width or 224) * 3
        if stitched.shape[1] > target_w:
            scale = target_w / stitched.shape[1]
            new_h = int(stitched.shape[0] * scale)
            stitched = cv2.resize(stitched, (target_w, new_h),
                                  interpolation=cv2.INTER_AREA)

        # Save debug stitched image
        if self.calls < 3:
            os.makedirs(self.log_dir, exist_ok=True)
            dbg = os.path.join(self.log_dir, f"debug_{self.calls}_stitched.jpg")
            cv2.imwrite(dbg, stitched)

        _, buf = cv2.imencode('.jpg', stitched, [cv2.IMWRITE_JPEG_QUALITY, 60])
        img_b64 = base64.b64encode(buf).decode()

        # Build prompts
        system_prompt = self.ego.build_system_prompt(
            goal, goal_mode, last_action, consecutive_turns
        )
        user_content = self.ego.build_fast_prompt(goal)

        # Single image message — no history (images in history = huge tokens)
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": user_content,
                "images": [img_b64],
            }
        ]
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "keep_alive": "30m",
            "options": {
                "num_predict": self.num_predict,
                "temperature": 0.0,
                "num_ctx": self.num_ctx,
            }
        }

        t0 = time.time()
        for attempt in range(3):
            try:
                resp = self.session.post(self.api_url, json=payload,
                                         timeout=self.timeout)
                resp.raise_for_status()
                break
            except (requests.exceptions.ReadTimeout,
                    requests.exceptions.ConnectionError) as retry_err:
                if attempt < 2:
                    wait = (attempt + 1) * 2
                    print(f"  [RETRY] VLM timeout, "
                          f"retry {attempt+1}/2 in {wait}s...")
                    self.ego.trim_history()
                    time.sleep(wait)
                else:
                    raise retry_err
        total_ms = (time.time() - t0) * 1000
        raw = resp.json().get("message", {}).get("content", "").strip()

        # ── Parse MOVE: output ──────────────────
        think = ""
        move = ""
        i_see = ""
        for line in raw.split("\n"):
            ll = line.strip().lower()
            if ll.startswith("think:"):
                think = line.strip()[6:].strip()
            elif ll.startswith("move:"):
                move = line.strip()[5:].strip().lower()
            elif ll.startswith("i see:"):
                i_see = line.strip()[6:].strip()

        # Fallback: scan entire response for move keywords if MOVE: tag missing
        if not move:
            for word in ["forward", "stop", "left", "right"]:
                if word in raw.lower():
                    move = word
                    break

        # Map to canonical action
        MOVE_MAP = {
            "forward": "forward", "ahead": "forward", "straight": "forward",
            "left": "left",
            "right": "right",
            "stop": "stop", "halt": "stop", "arrived": "stop",
        }
        action = None
        for k, v in MOVE_MAP.items():
            if k in move:
                action = v
                break

        # Default search if no valid action
        if action is None:
            if consecutive_turns >= 5 and last_action in ("left", "right"):
                action = "right" if last_action == "left" else "left"
            else:
                action = "left"
            think = think or "no clear move parsed — searching"
        reason = think or i_see or raw[:80] or "no response"
        found = action not in (None,)

        self.ego.record_observation(
            "center" if action == "forward" else (action if action in ("left", "right") else "all"),
            action != "left" or self.ego.total_detections > 0,
            reason
        )

        print(f"    [VLM2] raw: \"{raw[:80]}\"")
        print(f"           think: \"{think[:70]}\"")
        print(f"           move: \"{move}\" → action={action}  ({total_ms:.0f}ms)")

        with self._lock:
            self.last_action = action
            self.last_raw = move or raw[:40]
            self.last_reason = reason
            self.last_ms = total_ms
            self.last_panel = None   # panoramic mode: action is direct
            self.last_goal = goal
            self.calls += 1
        print(f"  [VLM2] {goal} → MOVE={action}  ({total_ms:.0f}ms)")

    # ──────────────────────────────────────────────────────────────
    # Shared helpers
    # ──────────────────────────────────────────────────────────────

    def _store_result(self, panels, goal, found_panel, found_reason,
                      panel_results, total_ms,
                      last_action, consecutive_turns):
        """Store the per-camera result into shared state."""
        if found_panel:
            action = PANEL_ACTION_MAP[found_panel]
            reason = found_reason[:80]
            self.ego.record_observation(found_panel, True, reason)

            with self._lock:
                self.last_action = action
                self.last_raw = f"{goal} in {found_panel}"
                self.last_reason = reason
                self.last_ms = total_ms
                self.last_panel = found_panel
                self.last_goal = goal
                self.calls += 1

            print(f"  [VLM] {goal} → {found_panel} → {action} "
                  f"({total_ms:.0f}ms)")
        else:
            if consecutive_turns >= 3 and last_action in ("left", "right"):
                search_dir = "right" if last_action == "left" else "left"
            else:
                search_dir = "left"

            reasons = [f"{k}: {v[:30]}" for k, v in panel_results.items()]
            reason = " | ".join(reasons)
            self.ego.record_observation("all", False, reason[:60])

            with self._lock:
                self.last_action = search_dir
                self.last_raw = f"not found → search {search_dir}"
                self.last_reason = reason[:80]
                self.last_ms = total_ms
                self.last_panel = None
                self.last_goal = goal
                self.calls += 1

            print(f"  [VLM] {goal} not found → search {search_dir} "
                  f"({total_ms:.0f}ms)")

    def _stitch_for_vlm(self, panels):
        """Stitch LEFT|CENTER|RIGHT into one panoramic image.
        Large text labels + thick cyan dividers so VLM can spatially localize.
        """
        target_h = 160
        target_w = 213   # ~4:3 aspect per panel
        label_h = 22     # header strip height
        parts = []
        names = ["left", "center", "right"]
        labels = ["LEFT", "CENTER", "RIGHT"]

        for name in names:
            if name in panels and panels[name] is not None:
                img = cv2.resize(panels[name], (target_w, target_h))
            else:
                img = np.zeros((target_h, target_w, 3), dtype=np.uint8)
                cv2.putText(img, "NO FEED", (30, target_h // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 180), 1)
            parts.append(img)

        # Stitch horizontally with 3-pixel cyan dividers
        divider = np.full((target_h, 3, 3), [255, 255, 0], dtype=np.uint8)
        body = np.hstack([parts[0], divider, parts[1], divider.copy(), parts[2]])

        # Build label header strip
        total_w = body.shape[1]
        header = np.zeros((label_h, total_w, 3), dtype=np.uint8)
        offsets = [target_w // 2 - 22,            # LEFT panel center
                   target_w + 3 + target_w // 2 - 30,   # CENTER panel center
                   (target_w + 3) * 2 + target_w // 2 - 25]  # RIGHT panel center
        for lbl, x in zip(labels, offsets):
            cv2.putText(header, lbl, (max(0, x), label_h - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        # Cyan dividers in header too
        for div_x in [target_w + 1, (target_w + 3) * 2 + 1]:
            header[:, div_x:div_x + 3] = [255, 255, 0]

        return np.vstack([header, body])

    # ── Synonym / keyword matching (for per-camera mode) ──

    SYNONYMS = {
        "human":   ["human", "person", "man", "woman", "people", "someone",
                     "boy", "girl", "child", "figure"],
        "person":  ["human", "person", "man", "woman", "people", "someone"],
        "bottle":  ["bottle", "water bottle", "flask"],
        "cup":     ["cup", "mug", "glass", "tumbler"],
        "chair":   ["chair", "seat", "stool", "sofa", "couch"],
        "table":   ["table", "desk", "counter"],
        "door":    ["door", "doorway", "entrance"],
        "car":     ["car", "vehicle", "automobile", "truck"],
        "dog":     ["dog", "puppy"],
        "cat":     ["cat", "kitten"],
        "phone":   ["phone", "smartphone", "mobile"],
        "laptop":  ["laptop", "computer", "notebook"],
        "tv":      ["tv", "television", "screen", "monitor"],
        "bag":     ["bag", "backpack", "purse", "handbag"],
        "carpet":  ["carpet", "rug", "mat"],
        "bed":     ["bed", "mattress"],
        "fan":     ["fan", "ceiling fan"],
        "light":   ["light", "lamp", "bulb"],
        "shoe":    ["shoe", "sneaker", "footwear"],
        "book":    ["book", "notebook"],
        "wall":    ["wall"],
        "window":  ["window"],
    }

    def _get_goal_keywords(self, goal):
        """Get all keywords that could match this goal, including synonyms."""
        goal_lower = goal.lower().strip()
        keywords = {goal_lower}

        for key, syns in self.SYNONYMS.items():
            if goal_lower == key or goal_lower in syns:
                keywords.update(syns)
                break

        # Add plural/singular variants
        extra = set()
        for kw in keywords:
            if kw.endswith("s"):
                extra.add(kw[:-1])
            else:
                extra.add(kw + "s")
        keywords.update(extra)

        return keywords

    def _check_goal_in_description(self, description, goal_keywords):
        """Check if any goal keyword appears in the VLM's description."""
        desc_lower = description.lower()
        desc_words = set(desc_lower.replace(",", " ").replace(".", " ")
                         .replace(";", " ").replace(":", " ").split())
        for kw in goal_keywords:
            if " " not in kw:
                if kw in desc_words:
                    return True
            else:
                if kw in desc_lower:
                    return True
        return False
