import cv2
import math
from collections import deque
import numpy as np
import mediapipe as mp

# =========================
# Parameters
# =========================
SMOOTH_N = 7
PINCH_FORCE_ZERO = 0.08
R_MIN_DEFAULT = 0.10
R_MAX_DEFAULT = 0.60
ANGLE_MIN, ANGLE_MAX = 0, 180

BAR_H, BAR_W = 220, 22
MARGIN = 40

# =========================
# MediaPipe setup
# =========================
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    model_complexity=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

# =========================
# State per hand
# keys: "Left", "Right" as reported by MediaPipe
# =========================
state = {
    "Left":  {
        "r_min": R_MIN_DEFAULT,
        "r_max": R_MAX_DEFAULT,
        "hist": deque(maxlen=SMOOTH_N),
        "angle": 0.0,
        "r": None
    },
    "Right": {
        "r_min": R_MIN_DEFAULT,
        "r_max": R_MAX_DEFAULT,
        "hist": deque(maxlen=SMOOTH_N),
        "angle": 0.0,
        "r": None
    }
}

def l2(p1, p2):
    return math.hypot(p1[0]-p2[0], p1[1]-p2[1])

def clamp(x, lo, hi):
    return max(lo, min(hi, x))

def map_norm_to_angle(r, rmin, rmax):
    if r is None:
        return None
    if r <= PINCH_FORCE_ZERO:
        return 0.0
    t = (r - rmin) / max(1e-6, (rmax - rmin))
    t = clamp(t, 0.0, 1.0)
    return ANGLE_MIN + t * (ANGLE_MAX - ANGLE_MIN)

def put_text(img, text, org, scale=0.8, thickness=2, color=(255,255,255)):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0,0,0), thickness+2, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)

def draw_bar(img, angle, x, y, label):
    # outline
    cv2.rectangle(img, (x, y), (x+BAR_W, y+BAR_H), (255,255,255), 2)
    # fill
    ang = 0.0 if angle is None else float(angle)
    fill_h = int((ang/180.0) * (BAR_H-4))
    cv2.rectangle(img, (x+2, y+BAR_H-2-fill_h), (x+BAR_W-2, y+BAR_H-2), (255,255,255), -1)
    # ticks
    for k in [0, 45, 90, 135, 180]:
        ty = y + BAR_H - int((k/180.0) * (BAR_H-4)) - 2
        cv2.line(img, (x+BAR_W+6, ty), (x+BAR_W+26, ty), (255,255,255), 1)
        put_text(img, f"{k}", (x+BAR_W+30, ty+4), scale=0.5, thickness=1)
    put_text(img, f"{label}", (x-2, y-10), scale=0.7, thickness=2)

def process_hand(frame, hand_label, lm_list, w, h):
    """
    Compute normalized pinch r and mapped angle for a hand.
    Landmarks of interest: thumb tip (4), index tip (8), index MCP (5), pinky MCP (17)
    """
    # Collect pixel coordinates
    idxs = [4, 8, 5, 17]
    pts = {}
    for i in idxs:
        x = int(lm_list[i].x * w)
        y = int(lm_list[i].y * h)
        pts[i] = (x, y)

    # Distances
    d_tip = l2(pts[4], pts[8])
    d_ref = max(10.0, l2(pts[5], pts[17]))  # hand width as scale
    r = d_tip / d_ref

    # Map to angle
    s = state[hand_label]
    angle = map_norm_to_angle(r, s["r_min"], s["r_max"])
    s["hist"].append(angle)
    angle_smooth = float(np.mean(s["hist"])) if len(s["hist"]) else angle

    # Save back
    s["r"] = r
    s["angle"] = angle_smooth

    # Draw landmarks & guides
    cv2.circle(frame, pts[4], 10, (0,255,255), -1)
    cv2.circle(frame, pts[8], 10, (0,255,255), -1)
    cv2.line(frame, pts[4], pts[8], (255,255,255), 2)

    cv2.circle(frame, pts[5], 6, (255,255,255), -1)
    cv2.circle(frame, pts[17],6, (255,255,255), -1)
    cv2.line(frame, pts[5], pts[17], (200,200,200), 1)

    return r, angle_smooth

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Cannot open camera 0")

try:
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        H, W = frame.shape[:2]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hands.process(rgb)

        # Reset per-frame r/angle visibility
        state["Left"]["r"] = None
        state["Right"]["r"] = None

        if res.multi_hand_landmarks:
            # Tie handedness to landmarks (same order)
            handed = []
            if res.multi_handedness:
                for hinfo in res.multi_handedness:
                    handed.append(hinfo.classification[0].label)  # "Left" or "Right"
            else:
                handed = ["Right"] * len(res.multi_hand_landmarks)

            for lm, label in zip(res.multi_hand_landmarks, handed):
                # Clamp to only expected keys
                if label not in ("Left", "Right"):
                    label = "Right"
                # Draw skeleton
                mp_drawing.draw_landmarks(
                    frame, lm, mp_hands.HAND_CONNECTIONS,
                    mp_styles.get_default_hand_landmarks_style(),
                    mp_styles.get_default_hand_connections_style()
                )
                process_hand(frame, label, lm.landmark, W, H)

        # Draw dual bars & text
        # Left-hand bar (Servo 2) on left
        draw_bar(frame, state["Left"]["angle"], MARGIN, MARGIN, "Left → Servo 2")
        # Right-hand bar (Servo 1) on right
        draw_bar(frame, state["Right"]["angle"], W - MARGIN - BAR_W - 40, MARGIN, "Right → Servo 1")

        # Text readouts (center-bottom)
        left_angle = int(round(state["Left"]["angle"])) if state["Left"]["r"] is not None else None
        right_angle = int(round(state["Right"]["angle"])) if state["Right"]["r"] is not None else None

        if left_angle is not None:
            put_text(frame, f"Left: angle={left_angle:3d} deg  r={state['Left']['r']:.3f}  "
                            f"(r_min={state['Left']['r_min']:.2f}, r_max={state['Left']['r_max']:.2f})",
                     (MARGIN, H-60), scale=0.75, thickness=2)
        else:
            put_text(frame, "Left: ---", (MARGIN, H-60), scale=0.75, thickness=2)

        if right_angle is not None:
            put_text(frame, f"Right: angle={right_angle:3d} deg  r={state['Right']['r']:.3f}  "
                            f"(r_min={state['Right']['r_min']:.2f}, r_max={state['Right']['r_max']:.2f})",
                     (MARGIN, H-30), scale=0.75, thickness=2)
        else:
            put_text(frame, "Right: ---", (MARGIN, H-30), scale=0.75, thickness=2)

        # Help
        put_text(frame, "Calib Left:  Z=ZERO  X=MAX   |   Calib Right:  N=ZERO  M=MAX   |   Q=Quit",
                 (MARGIN, 30), scale=0.7, thickness=1)

        cv2.imshow("Dual Pinch-to-Servo (Two Hands)", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

        # --- Calibration keys ---
        # Left hand (Servo 2)
        if key == ord('z'):
            if state["Left"]["r"] is not None:
                r_cur = state["Left"]["r"]
                state["Left"]["r_min"] = max(0.0, min(r_cur, state["Left"]["r_max"] - 0.02))
        elif key == ord('x'):
            if state["Left"]["r"] is not None:
                r_cur = state["Left"]["r"]
                state["Left"]["r_max"] = max(state["Left"]["r_min"] + 0.02, r_cur)

        # Right hand (Servo 1)
        elif key == ord('n'):
            if state["Right"]["r"] is not None:
                r_cur = state["Right"]["r"]
                state["Right"]["r_min"] = max(0.0, min(r_cur, state["Right"]["r_max"] - 0.02))
        elif key == ord('m'):
            if state["Right"]["r"] is not None:
                r_cur = state["Right"]["r"]
                state["Right"]["r_max"] = max(state["Right"]["r_min"] + 0.02, r_cur)

finally:
    cap.release()
    hands.close()
    cv2.destroyAllWindows()
