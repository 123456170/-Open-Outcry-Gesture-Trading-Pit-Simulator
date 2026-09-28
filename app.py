import os
import csv
import time
import math
import random
import threading
import tempfile
from collections import deque
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import cv2

try:
    import mediapipe as mp
    MEDIAPIPE_OK = True
except Exception:
    mp = None
    MEDIAPIPE_OK = False

st.set_page_config(
    page_title="Open-Outcry Gesture Trading Simulator",
    page_icon="🖐️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Optional fallback for older Streamlit versions.
if not hasattr(st, "fragment"):
    def _fragment_fallback(*args, **kwargs):
        def wrapper(func):
            return func
        return wrapper
    st.fragment = _fragment_fallback


TICKERS = ["PITX", "OPRT", "GEST", "HAND"]
STARTING_PRICES = {
    "PITX": 187.35,
    "OPRT": 96.70,
    "GEST": 251.10,
    "HAND": 58.42,
}
STARTING_POSITIONS = {
    "PITX": 40,
    "OPRT": 25,
    "GEST": 0,
    "HAND": 18,
}
STARTING_CASH = 250_000.0
ORDER_CSV = "open_outcry_order_log.csv"
VIDEO_FPS = 8
PENDING_SECONDS = 1.5
STABLE_SECONDS = 0.5
POINT_SECONDS = 0.5


def fmt_money(x):
    return f"${x:,.2f}"


def classify_signal(landmarks):
    """
    Classify MediaPipe hand landmarks into open-outcry style signals.

    Rules:
    - Palm facing out = SELL
    - Palm facing in = BUY
    - 1 to 5 extended fingers = quantity
    - Raised fist = cancel/hold
    - Index-only point = ticker selection
    """
    if not landmarks or len(landmarks) < 21:
        return {"hand_present": False}

    lm = np.array([[p.x, p.y, p.z] for p in landmarks], dtype=np.float32)
    wrist = lm[0]

    # Palm normal approximation from wrist -> index MCP and wrist -> pinky MCP.
    v1 = lm[5] - wrist
    v2 = lm[17] - wrist
    normal = np.cross(v1, v2)

    # Heuristic: negative z normal tends to mean palm faces the camera.
    palm_toward_camera = bool(normal[2] < 0.0)

    # Finger extension by distance from wrist.
    finger_defs = [(8, 6), (12, 10), (16, 14), (20, 18)]
    extended = []
    for tip, pip in finger_defs:
        d_tip = np.linalg.norm(lm[tip] - wrist)
        d_pip = np.linalg.norm(lm[pip] - wrist)
        extended.append(bool(d_tip > d_pip * 1.12))

    # Thumb heuristic for allowing 5-lot quantities.
    thumb_tip = lm[4]
    thumb_ip = lm[3]
    pinky_mcp = lm[17]
    thumb_extended = bool(
        np.linalg.norm(thumb_tip - pinky_mcp) > np.linalg.norm(thumb_ip - pinky_mcp) * 1.12
    )

    qty = int(sum(extended) + (1 if thumb_extended else 0))
    qty = max(0, min(5, qty))

    index_ext, middle_ext, ring_ext, pinky_ext = extended
    pointing = bool(index_ext and not middle_ext and not ring_ext and not pinky_ext)
    fist = qty == 0

    side = None
    if 1 <= qty <= 5:
        side = "SELL" if palm_toward_camera else "BUY"

    return {
        "hand_present": True,
        "palm_toward_camera": palm_toward_camera,
        "side": side,
        "qty": qty,
        "pointing": pointing,
        "point_x": float(lm[8][0]),
        "fist": bool(fist),
    }


class DemoScene:
    """Self-contained animated scene used for instant/demo video generation."""

    def __init__(self, state):
        self.tickers = TICKERS
        self.hist = {}
        with state.lock:
            for t in TICKERS:
                self.hist[t] = deque(list(state.history[t])[-100:], maxlen=120)

        self.prices = {t: float(self.hist[t][-1]) for t in TICKERS}
        self.active = random.choice(TICKERS)
        self.cash = STARTING_CASH
        self.positions = dict(STARTING_POSITIONS)
        self.orders = []
        self.pending = None
        self.gesture = "Demo starting"
        self.frame = 0
        self.next_event = 10

    def step(self):
        self.frame += 1

        for t in self.tickers:
            p = self.prices[t]
            p = max(1.0, p + random.gauss(0, p * 0.002))
            self.prices[t] = p
            self.hist[t].append(p)

        if self.pending and self.frame >= self.pending["confirm_frame"]:
            self._execute()

        if self.frame >= self.next_event and not self.pending:
            self._start_event()

    def _start_event(self):
        self.active = random.choice(self.tickers)
        side = random.choice(["BUY", "SELL"])
        qty = random.randint(1, 5)
        palm = "out" if side == "SELL" else "in"
        self.gesture = f"Palm {palm}, {qty} finger(s) -> {side} {qty} {self.active}"
        self.pending = {
            "ticker": self.active,
            "side": side,
            "qty": qty,
            "confirm_frame": self.frame + 8,
        }
        self.next_event = self.frame + random.randint(20, 35)

    def _execute(self):
        p = self.pending
        price = self.prices[p["ticker"]]

        if p["side"] == "BUY":
            self.cash -= p["qty"] * price
            self.positions[p["ticker"]] = self.positions.get(p["ticker"], 0) + p["qty"]
        else:
            self.cash += p["qty"] * price
            self.positions[p["ticker"]] = self.positions.get(p["ticker"], 0) - p["qty"]

        self.orders.insert(
            0,
            {
                "time": self.frame,
                "ticker": p["ticker"],
                "side": p["side"],
                "qty": p["qty"],
                "price": price,
            },
        )
        self.orders = self.orders[:8]
        self.gesture = f"Chop confirmed {p['side']} {p['qty']} {p['ticker']} @ {price:.2f}"
        self.pending = None


def render_demo_frame(scene):
    """Draw a colorful simulated dashboard frame for video output."""
    W, H = 760, 480
    img = np.full((H, W, 3), (18, 18, 24), dtype=np.uint8)

    cv2.putText(
        img,
        "OPEN-OUTCRY GESTURE PIT - SIMULATION ONLY",
        (16, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (80, 200, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        img,
        "Fictional prices / no real trading",
        (16, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (180, 180, 180),
        1,
        cv2.LINE_AA,
    )

    # Top ticker charts.
    for i, t in enumerate(TICKERS):
        x = 16 + i * 185
        y = 58
        w = 175
        h = 110
        active = t == scene.active
        color = (0, 200, 120) if active else (90, 90, 110)

        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2 if active else 1)
        cv2.putText(img, t, (x + 6, y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        price = scene.prices[t]
        cv2.putText(img, f"{price:.2f}", (x + 6, y + 36), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 255), 1, cv2.LINE_AA)

        pts = list(scene.hist[t])[-80:]
        if len(pts) > 1:
            mn, mx = min(pts), max(pts)
            rng = (mx - mn) or 1.0
            arr = []
            for j, val in enumerate(pts):
                px = x + 8 + int((w - 16) * j / (len(pts) - 1))
                py = y + h - 10 - int((h - 52) * (val - mn) / rng)
                arr.append([px, py])
            cv2.polylines(img, [np.array(arr, dtype=np.int32)], False, (0, 180, 255), 1, cv2.LINE_AA)

    # Left hand/gesture panel.
    cv2.rectangle(img, (16, 185), (380, 460), (40, 40, 55), -1)
    cv2.putText(img, "Hand signal", (26, 205), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(img, scene.gesture[:44], (26, 235), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 180), 1, cv2.LINE_AA)

    qty = scene.pending["qty"] if scene.pending else 0
    cx, cy = 120, 340
    cv2.circle(img, (cx, cy), 45, (0, 120, 220), -1)

    for f in range(qty):
        angle = -60 + f * 30
        rad = math.radians(angle)
        x2 = int(cx + 80 * math.cos(rad))
        y2 = int(cy - 80 * math.sin(rad))
        cv2.line(img, (cx, cy), (x2, y2), (0, 220, 255), 8, cv2.LINE_AA)

    if scene.pending:
        cv2.putText(
            img,
            f"PENDING {scene.pending['side']} {scene.pending['qty']} {scene.pending['ticker']}",
            (26, 420),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 200, 255),
            1,
            cv2.LINE_AA,
        )
    else:
        cv2.putText(img, "No pending order", (26, 420), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

    # Right portfolio/order panel.
    cv2.rectangle(img, (395, 185), (745, 460), (40, 40, 55), -1)
    equity = scene.cash + sum(scene.positions.get(t, 0) * scene.prices[t] for t in TICKERS)

    cv2.putText(img, "Portfolio", (405, 205), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(img, f"Cash: {scene.cash:,.0f}", (405, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 255, 200), 1, cv2.LINE_AA)
    cv2.putText(img, f"Equity: {equity:,.0f}", (405, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 255, 200), 1, cv2.LINE_AA)

    y = 280
    for order in scene.orders[:6]:
        line = f"{order['side']} {order['qty']} {order['ticker']} @ {order['price']:.2f}"
        cv2.putText(img, line, (405, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)
        y += 20

    return img


class AppState:
    """Holds price feed, portfolio, gesture state, camera state, and video state."""

    def __init__(self):
        self.lock = threading.RLock()

        self.tickers = TICKERS
        self.prices = dict(STARTING_PRICES)
        self.history = {t: deque(maxlen=220) for t in TICKERS}

        self.cash = STARTING_CASH
        self.positions = dict(STARTING_POSITIONS)
        self.orders = []

        self.pending = None
        self.active_ticker = TICKERS[0]
        self.status = "Simulation ready."
        self.hand_info = "No live hand signal. Autopilot/manual demo active."
        self.demo_event = "Auto demo started"

        self.camera_enabled = False
        self.autopilot = True
        self.swap_palm = False
        self.frame = None

        self.stop_event = threading.Event()
        self.feed_thread = None
        self.camera_thread = None
        self.autopilot_thread = None

        # Gesture stability tracking.
        self.stable_key = None
        self.stable_since = 0.0
        self.point_target = None
        self.point_since = 0.0
        self.fist_since = 0.0

        # Chop detection.
        self.chop_history = deque(maxlen=30)
        self.last_chop = 0.0

        # Video state.
        self.video_building = False
        self.recording = False
        self.record_source = None
        self.record_end = 0.0
        self.record_fps = VIDEO_FPS
        self.video_writer = None
        self.video_tmp_path = None
        self.video_path = None
        self.video_message = ""
        self.auto_video_started = False

        self._prefill()
        self.initial_equity = self.cash + sum(
            self.positions.get(t, 0) * self.prices.get(t, 0.0) for t in self.tickers
        )
        self._ensure_csv()

    def _prefill(self):
        for t in TICKERS:
            p = STARTING_PRICES[t] * random.uniform(0.96, 1.04)
            vals = []
            for _ in range(180):
                p = max(1.0, p + random.gauss(0, p * 0.0018))
                vals.append(p)
            self.history[t].extend(vals)
            self.prices[t] = vals[-1]

    def start(self):
        if self.feed_thread is None or not self.feed_thread.is_alive():
            self.feed_thread = threading.Thread(target=self._price_loop, daemon=True)
            self.feed_thread.start()

        if self.autopilot_thread is None or not self.autopilot_thread.is_alive():
            self.autopilot_thread = threading.Thread(target=self._autopilot_loop, daemon=True)
            self.autopilot_thread.start()

    def _price_loop(self):
        while not self.stop_event.is_set():
            with self.lock:
                for t in self.tickers:
                    p = self.prices[t]
                    p = max(1.0, p + random.gauss(0, p * 0.0022))
                    self.prices[t] = p
                    self.history[t].append(p)
                self._expire_pending_locked()
            time.sleep(1.0)

    def _autopilot_loop(self):
        while not self.stop_event.is_set():
            if self.autopilot:
                try:
                    with self.lock:
                        pending_exists = self.pending is not None

                    if not pending_exists:
                        ticker = random.choice(self.tickers)
                        self.set_active(ticker, source="autopilot point")
                        time.sleep(0.7)

                        side = random.choice(["BUY", "SELL"])
                        qty = random.randint(1, 5)
                        self.hand_info = (
                            f"Simulated pit signal: palm {'out' if side == 'SELL' else 'in'}, "
                            f"{qty} finger(s) on {ticker}"
                        )

                        self.create_pending(ticker, side, qty, source="autopilot")
                        time.sleep(0.45)
                        self.confirm_chop(source="autopilot chop")
                except Exception as e:
                    self.status = f"Autopilot error: {e}"

            time.sleep(random.uniform(4.0, 7.0))

    def _ensure_csv(self):
        if not os.path.exists(ORDER_CSV):
            with open(ORDER_CSV, "w", newline="") as f:
                csv.writer(f).writerow(
                    ["timestamp", "ticker", "direction", "quantity", "price", "source", "status"]
                )

    def _append_order(self, order):
        self.orders.insert(0, order)
        self.orders = self.orders[:200]
        try:
            with open(ORDER_CSV, "a", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=["timestamp", "ticker", "direction", "quantity", "price", "source", "status"],
                )
                writer.writerow(order)
        except Exception:
            pass

    def _expire_pending_locked(self):
        if self.pending and time.time() > self.pending["expires"]:
            self.pending = None
            self.status = "Pending order expired. Make the signal again."

    def expire_pending(self):
        with self.lock:
            self._expire_pending_locked()

    def set_active(self, ticker, source="manual"):
        with self.lock:
            if ticker in self.tickers and ticker != self.active_ticker:
                self.active_ticker = ticker
                self.status = f"{ticker} selected as active pit/product ({source})."

    def create_pending(self, ticker=None, side="BUY", qty=1, source="gesture"):
        with self.lock:
            self._expire_pending_locked()

            if self.pending:
                self.status = "Pending order already waiting for chop confirm."
                return

            ticker = ticker or self.active_ticker
            qty = max(1, min(5, int(qty)))

            self.pending = {
                "ticker": ticker,
                "side": side,
                "qty": qty,
                "created": time.time(),
                "expires": time.time() + PENDING_SECONDS,
                "source": source,
            }
            self.status = f"Pending {side} {qty} {ticker}. Confirm with downward chop."

    def cancel_pending(self, source="manual"):
        with self.lock:
            if self.pending:
                self.pending = None
                self.status = f"Pending order cancelled/hold by {source}."

    def confirm_chop(self, source="chop"):
        with self.lock:
            has_pending = self.pending is not None

        if has_pending:
            self.execute_pending(source=source)
        else:
            self.status = "No pending order to confirm."

    def execute_pending(self, source="gesture"):
        with self.lock:
            self._expire_pending_locked()
            if not self.pending:
                return False

            pending = self.pending
            self.pending = None

            ticker = pending["ticker"]
            side = pending["side"]
            qty = int(pending["qty"])
            price = self.prices[ticker]

            executed = False
            reason = ""

            if side == "BUY":
                cost = qty * price
                if cost <= self.cash:
                    self.cash -= cost
                    self.positions[ticker] = self.positions.get(ticker, 0) + qty
                    executed = True
                else:
                    reason = "insufficient cash"
            else:
                new_pos = self.positions.get(ticker, 0) - qty
                if new_pos >= -200:
                    self.cash += qty * price
                    self.positions[ticker] = new_pos
                    executed = True
                else:
                    reason = "short limit"

            status = "EXECUTED" if executed else "REJECTED"
            order = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "ticker": ticker,
                "direction": side,
                "quantity": qty,
                "price": round(price, 2),
                "source": source,
                "status": status,
            }
            self._append_order(order)

            if executed:
                self.status = f"Executed {side} {qty} {ticker} @ {price:.2f}"
                self.demo_event = self.status
            else:
                self.status = f"Rejected {side} {qty} {ticker}: {reason}"

            return executed

    def ticker_from_x(self, x):
        if x < 0.25:
            return TICKERS[0]
        if x < 0.50:
            return TICKERS[1]
        if x < 0.75:
            return TICKERS[2]
        return TICKERS[3]

    def set_camera(self, enabled):
        with self.lock:
            self.camera_enabled = enabled

        if enabled:
            if not MEDIAPIPE_OK:
                with self.lock:
                    self.camera_enabled = False
                self.status = "MediaPipe not available. Camera disabled."
                return

            if self.camera_thread is None or not self.camera_thread.is_alive():
                self.camera_thread = threading.Thread(target=self._camera_loop, daemon=True)
                self.camera_thread.start()
        else:
            self.frame = None
            self.status = "Camera disabled. Using autopilot/manual simulation."

    def _detect_chop(self, now):
        recent = [(t, y) for t, y in self.chop_history if now - t <= 0.35]
        if len(recent) < 5:
            return False

        ys = [y for _, y in recent]
        downward_delta = recent[-1][1] - recent[0][1]
        total_range = max(ys) - min(ys)

        if downward_delta > 0.06 and total_range > 0.07 and now - self.last_chop > 1.0:
            self.last_chop = now
            return True

        return False

    def _describe_signal(self, cls):
        if not cls or not cls.get("hand_present"):
            return "No hand detected."

        parts = []

        if cls.get("pointing"):
            parts.append(f"Pointing {self.ticker_from_x(cls.get('point_x', 0.5))}")

        if cls.get("fist"):
            parts.append("Raised fist = cancel/hold")

        if cls.get("side"):
            palm = "out" if cls["side"] == "SELL" else "in"
            parts.append(f"Palm {palm} = {cls['side']}")
            parts.append(f"{cls.get('qty', 0)} finger(s)")

        return " | ".join(parts) if parts else "Hand visible, no clear order signal."

    def _process_camera_signal(self, cls, hand):
        now = time.time()
        wrist_y = float(hand.landmark[0].y)
        self.chop_history.append((now, wrist_y))
        chop = self._detect_chop(now)

        if self.swap_palm and cls.get("side"):
            cls["side"] = "SELL" if cls["side"] == "BUY" else "BUY"
            cls["palm_toward_camera"] = not cls.get("palm_toward_camera", False)

        with self.lock:
            self.hand_info = self._describe_signal(cls)

        # Ticker selection by pointing.
        if cls.get("pointing"):
            target = self.ticker_from_x(cls.get("point_x", 0.5))
            if self.point_target == target:
                if now - self.point_since >= POINT_SECONDS:
                    self.set_active(target, source="point")
            else:
                self.point_target = target
                self.point_since = now
        else:
            self.point_target = None
            self.point_since = now

        # Fist cancel/hold.
        if cls.get("fist"):
            if self.fist_since == 0:
                self.fist_since = now
            elif now - self.fist_since >= STABLE_SECONDS:
                self.cancel_pending(source="fist")
                self.fist_since = 0
        else:
            self.fist_since = 0

        # Buy/sell + quantity, but not while pointing.
        if not cls.get("pointing") and cls.get("side") and int(cls.get("qty", 0)) >= 1:
            key = (cls["side"], int(cls["qty"]), self.active_ticker)

            if self.stable_key == key:
                if now - self.stable_since >= STABLE_SECONDS:
                    with self.lock:
                        has_pending = self.pending is not None

                    if not has_pending:
                        self.create_pending(self.active_ticker, cls["side"], int(cls["qty"]), source="camera")

                    self.stable_key = None
                    self.stable_since = 0
            else:
                self.stable_key = key
                self.stable_since = now
        else:
            self.stable_key = None
            self.stable_since = 0

        if chop:
            self.confirm_chop(source="camera chop")

    def _draw_hud(self, frame):
        h, w = frame.shape[:2]

        with self.lock:
            status = self.status
            active = self.active_ticker
            pending = self.pending

        cv2.rectangle(frame, (0, 0), (w, 46), (0, 0, 0), -1)
        cv2.putText(
            frame,
            "SIMULATION ONLY - FICTIONAL PRICES",
            (10, 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (80, 200, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"Active: {active} | {status[:80]}",
            (10, 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        if pending:
            txt = f"PENDING {pending['side']} {pending['qty']} {pending['ticker']} - CHOP TO CONFIRM"
            cv2.rectangle(frame, (10, h - 36), (w - 10, h - 8), (0, 180, 220), -1)
            cv2.putText(
                frame,
                txt,
                (16, h - 16),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 0),
                2,
                cv2.LINE_AA,
            )

        return frame

    def _camera_loop(self):
        cap = None
        hands = None

        try:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                raise RuntimeError("Camera not available")

            hands = mp.solutions.hands.Hands(
                max_num_hands=1,
                min_detection_confidence=0.6,
                min_tracking_confidence=0.5,
            )
            drawing_utils = mp.solutions.drawing_utils if MEDIAPIPE_OK else None
        except Exception as e:
            with self.lock:
                self.camera_enabled = False
            self.status = f"Camera error: {e}"
            return

        while not self.stop_event.is_set() and self.camera_enabled:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.05)
                continue

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            overlay = frame.copy()

            if results.multi_hand_landmarks:
                hand = results.multi_hand_landmarks[0]

                if drawing_utils:
                    drawing_utils.draw_landmarks(overlay, hand, mp.solutions.hands.HAND_CONNECTIONS)

                cls = classify_signal(hand.landmark)
                self._process_camera_signal(cls, hand)
            else:
                with self.lock:
                    self.hand_info = "No hand detected."
                self.stable_key = None
                self.point_target = None
                self.fist_since = 0

            overlay = self._draw_hud(overlay)
            self.frame = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

            if self.recording and self.record_source == "camera":
                self._write_video_frame(overlay)

            time.sleep(0.03)

        cap.release()
        if hands:
            hands.close()

        with self.lock:
            self.frame = None

    def _make_writer(self, path, fps, size):
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
        if writer.isOpened():
            return writer, path

        writer.release()
        alt = path.replace(".mp4", ".avi")
        writer = cv2.VideoWriter(alt, cv2.VideoWriter_fourcc(*"MJPG"), fps, size)
        if writer.isOpened():
            return writer, alt

        writer.release()
        return None, None

    def begin_video(self, seconds, source="synthetic"):
        with self.lock:
            if self.recording or self.video_building:
                return

            self.video_path = None
            self.recording = True
            self.video_building = True
            self.record_source = source
            self.record_end = time.time() + seconds
            self.record_fps = VIDEO_FPS
            self.video_message = f"Recording {source} demo for {seconds}s..."

        if source != "camera":
            threading.Thread(target=self._record_synthetic, args=(seconds,), daemon=True).start()
        else:
            threading.Thread(target=self._camera_record_watchdog, args=(seconds,), daemon=True).start()

    def _camera_record_watchdog(self, seconds):
        while time.time() < self.record_end and self.recording and not self.stop_event.is_set():
            time.sleep(0.2)

        if self.recording and self.record_source == "camera":
            self._finish_video()

    def _write_video_frame(self, frame):
        if not self.recording or frame is None:
            return

        if time.time() > self.record_end:
            self._finish_video()
            return

        frame = cv2.resize(frame, (760, 480))

        if self.video_writer is None:
            path = os.path.join(tempfile.gettempdir(), f"open_outcry_demo_{int(time.time())}.mp4")
            writer, actual_path = self._make_writer(path, self.record_fps, (760, 480))

            if writer is None:
                self.recording = False
                self.video_building = False
                self.video_message = "Video writer unavailable."
                return

            self.video_writer = writer
            self.video_tmp_path = actual_path

        self.video_writer.write(frame)

    def _record_synthetic(self, seconds):
        scene = DemoScene(self)
        n = int(seconds * self.record_fps)

        for _ in range(n):
            if not self.recording or self.stop_event.is_set():
                break

            scene.step()
            frame = render_demo_frame(scene)
            self._write_video_frame(frame)
            time.sleep(1.0 / self.record_fps)

        self._finish_video()

    def _finish_video(self):
        with self.lock:
            if not self.recording and not self.video_building:
                return

            if self.video_writer is not None:
                self.video_writer.release()
                self.video_writer = None

            if self.video_tmp_path and os.path.exists(self.video_tmp_path):
                self.video_path = self.video_tmp_path
                self.video_message = "Demo video ready."
            else:
                self.video_message = "No video frames captured."

            self.recording = False
            self.video_building = False

    def generate_instant_video(self, seconds=8):
        if self.video_building or self.recording:
            return

        self.video_building = True
        self.video_message = "Building instant demo video..."

        def worker():
            scene = DemoScene(self)
            fps = self.record_fps
            path = os.path.join(tempfile.gettempdir(), f"open_outcry_instant_{int(time.time())}.mp4")
            writer = None
            actual_path = path

            try:
                for _ in range(int(seconds * fps)):
                    scene.step()
                    frame = render_demo_frame(scene)

                    if writer is None:
                        writer, actual_path = self._make_writer(path, fps, (frame.shape[1], frame.shape[0]))
                        if writer is None:
                            self.video_message = "Video writer unavailable."
                            self.video_building = False
                            return

                    writer.write(frame)
            finally:
                if writer is not None:
                    writer.release()

            self.video_path = actual_path
            self.video_message = "Instant demo video ready."
            self.video_building = False

        threading.Thread(target=worker, daemon=True).start()


def get_state():
    if "app_state" not in st.session_state:
        st.session_state.app_state = AppState()
        st.session_state.app_state.start()

        if not st.session_state.app_state.auto_video_started:
            st.session_state.app_state.auto_video_started = True
            st.session_state.app_state.generate_instant_video(8)

    return st.session_state.app_state


@st.fragment(run_every=1.0)
def market_fragment():
    state = get_state()

    with state.lock:
        prices = state.prices.copy()
        histories = {t: list(state.history[t])[-90:] for t in TICKERS}
        active = state.active_ticker

    cols = st.columns(len(TICKERS))

    for i, t in enumerate(TICKERS):
        with cols[i]:
            series = histories[t]
            price = prices[t]
            delta = price - series[-2] if len(series) > 1 else 0.0

            label = f"{t} 🎯 ACTIVE" if t == active else t
            st.markdown(f"**{label}**")
            st.metric("Price", fmt_money(price), f"{delta:+.2f}")
            st.line_chart(pd.DataFrame({t: series}), height=150)


@st.fragment(run_every=1.0)
def portfolio_fragment():
    state = get_state()
    state.expire_pending()

    with state.lock:
        cash = state.cash
        positions = state.positions.copy()
        prices = state.prices.copy()
        initial_equity = state.initial_equity
        pending = state.pending

    equity = cash + sum(qty * prices.get(t, 0.0) for t, qty in positions.items())
    pnl = equity - initial_equity

    st.subheader("Portfolio")

    c1, c2, c3 = st.columns(3)
    c1.metric("Equity", fmt_money(equity), f"{pnl:+,.2f}")
    c2.metric("Cash", fmt_money(cash))
    c3.metric("Pending", pending["ticker"] if pending else "-")

    rows = []
    for t in TICKERS:
        pos = positions.get(t, 0)
        price = prices.get(t, 0)
        rows.append(
            {
                "Ticker": t,
                "Position": pos,
                "Price": round(price, 2),
                "Value": round(pos * price, 2),
            }
        )

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


@st.fragment(run_every=0.2)
def camera_fragment():
    state = get_state()

    st.subheader("Hand-signal feed")

    if state.camera_enabled and state.frame is not None:
        st.image(state.frame, use_container_width=True)
    else:
        st.info(state.hand_info)
        st.caption("Camera off: autopilot and manual fallback controls are simulating pit signals.")

    st.caption(state.status)


@st.fragment(run_every=0.2)
def orders_fragment():
    state = get_state()
    state.expire_pending()

    st.subheader("Order confirmation")

    with state.lock:
        pending = state.pending

    if pending:
        remaining = max(0.0, pending["expires"] - time.time())
        st.warning(
            f"PENDING: {pending['side']} {pending['qty']} {pending['ticker']} — "
            f"perform downward chop within {remaining:.1f}s"
        )
        st.progress(min(1.0, remaining / PENDING_SECONDS))
    else:
        st.info("No pending order. Show palm-in/out with 1-5 fingers for 0.5s.")

    st.subheader("Order history")

    with state.lock:
        orders = state.orders.copy()

    if orders:
        st.dataframe(pd.DataFrame(orders), height=240, use_container_width=True, hide_index=True)
    else:
        st.caption("No orders yet.")


@st.fragment(run_every=1.0)
def video_fragment():
    state = get_state()

    st.subheader("Demo video")

    if state.video_building or state.recording:
        st.info(state.video_message or "Preparing video...")

    if state.video_path and os.path.exists(state.video_path):
        if st.session_state.get("video_path_cache") != state.video_path:
            with open(state.video_path, "rb") as f:
                st.session_state.video_bytes = f.read()
            st.session_state.video_path_cache = state.video_path

        data = st.session_state.get("video_bytes", b"")
        if data:
            st.video(data)
            mime = "video/mp4" if state.video_path.endswith(".mp4") else "video/x-msvideo"
            st.download_button(
                "Download demo video",
                data=data,
                file_name=os.path.basename(state.video_path),
                mime=mime,
            )
    else:
        st.caption("Use the sidebar to generate or record a demo video.")


state = get_state()

st.title("🖐️ Open-Outcry Gesture Trading Pit Simulator")
st.caption(
    "SIMULATION ONLY: fictional prices, simulated orders, no real market data, "
    "no real brokerage, and no real execution."
)

with st.sidebar:
    st.warning("Simulation only. No API key required.")

    st.subheader("Controls")

    camera_on = st.checkbox("Enable camera hand tracking", value=state.camera_enabled)
    if camera_on != state.camera_enabled:
        state.set_camera(camera_on)

    state.autopilot = st.checkbox("Autopilot demo signals", value=state.autopilot)
    state.swap_palm = st.checkbox("Swap palm in/out mapping", value=state.swap_palm)

    st.subheader("Manual gesture fallback")

    active = st.selectbox(
        "Active ticker",
        TICKERS,
        index=TICKERS.index(state.active_ticker),
    )
    if active != state.active_ticker:
        state.set_active(active, source="manual")

    side = st.radio("Direction", ["BUY", "SELL"], horizontal=True)
    qty = st.slider("Quantity / fingers", 1, 5, 3)

    if st.button("Place order intent"):
        state.create_pending(active, side, qty, source="manual")

    if st.button("Confirm chop"):
        state.confirm_chop(source="manual")

    if st.button("Cancel / hold"):
        state.cancel_pending(source="manual")

    st.subheader("Demo video")

    seconds = st.slider("Recording length (seconds)", 5, 60, 10)
    source = st.radio(
        "Source",
        ["synthetic dashboard", "camera"],
        disabled=(not state.camera_enabled),
    )

    if st.button("Record live demo video"):
        state.begin_video(seconds, source="camera" if source == "camera" else "synthetic")

    if st.button("Generate instant demo video"):
        state.generate_instant_video(8)

market_fragment()

row = st.columns([2, 1])

with row[0]:
    camera_fragment()
    video_fragment()

with row[1]:
    portfolio_fragment()
    orders_fragment()