import cv2
import mediapipe as mp
import numpy as np

# -------- MODEL --------
model_path = r"F:\Coding\project\Drawing_with_hand_geusture\hand_landmarker.task"

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=1
)

landmarker = HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(1)

# -------- STATE --------
rectangles = []
temp_rect = None

draw_counter = 0
release_counter = 0

DRAW_THRESHOLD = 2
RELEASE_THRESHOLD = 2

alpha = 0.5   # smoothing
l = 2      # gap
threshold = 50


# -------- HELPERS --------
def is_fist(hand_landmarks):
    # Check if index, middle, ring, pinky tips are closer to wrist than their PIP joints
    for tip, pip in [(8, 6), (12, 10), (16, 14), (20, 18)]:
        tip_dist = (hand_landmarks[tip].x - hand_landmarks[0].x)**2 + (hand_landmarks[tip].y - hand_landmarks[0].y)**2
        pip_dist = (hand_landmarks[pip].x - hand_landmarks[0].x)**2 + (hand_landmarks[pip].y - hand_landmarks[0].y)**2
        if tip_dist > pip_dist:
            return False
    return True

def normalize(rect):
    x1, y1, x2, y2 = rect
    return min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)


def is_overlap(r1, r2, margin=0, area_thresh=0.1):
    x1, y1, x2, y2 = r1
    x3, y3, x4, y4 = r2

    # -------- EXPAND rectangles by margin --------
    x1 -= margin
    y1 -= margin
    x2 += margin
    y2 += margin

    x3 -= margin
    y3 -= margin
    x4 += margin
    y4 += margin

    # -------- INTERSECTION --------
    xi1 = max(x1, x3)
    yi1 = max(y1, y3)
    xi2 = min(x2, x4)
    yi2 = min(y2, y4)

    if xi2 <= xi1 or yi2 <= yi1:
        return False  # no overlap

    # -------- AREA CHECK --------
    inter_area = (xi2 - xi1) * (yi2 - yi1)

    area1 = (x2 - x1) * (y2 - y1)
    area2 = (x4 - x3) * (y4 - y3)

    min_area = min(area1, area2)

    # -------- STRONG CONDITION --------
    if inter_area / min_area > area_thresh:
        return True

    return False


def snap_rect(new_rect, old_rect):
    x1, y1, x2, y2 = new_rect
    x3, y3, x4, y4 = old_rect

    width = x2 - x1
    height = y2 - y1

    # BELOW → inherit WIDTH (top edge of new near bottom edge of old)
    if (x1 - x3)**2 + (y1 - y4)**2 < threshold**2 or (x2 - x4)**2 + (y1 - y4)**2 < threshold**2:
        new_x1, new_x2 = x3, x4
        new_y1 = y4 + l
        new_y2 = new_y1 + height
        return (new_x1, new_y1, new_x2, new_y2)

    # ABOVE → inherit WIDTH (bottom edge of new near top edge of old)
    if (x1 - x3)**2 + (y2 - y3)**2 < threshold**2 or (x2 - x4)**2 + (y2 - y3)**2 < threshold**2:
        new_x1, new_x2 = x3, x4
        new_y2 = y3 - l
        new_y1 = new_y2 - height
        return (new_x1, new_y1, new_x2, new_y2)

    # LEFT → inherit HEIGHT (right edge of new near left edge of old)
    if (x2 - x3)**2 + (y1 - y3)**2 < threshold**2 or (x2 - x3)**2 + (y2 - y4)**2 < threshold**2:
        new_y1, new_y2 = y3, y4
        new_x2 = x3 - l
        new_x1 = new_x2 - width
        return (new_x1, new_y1, new_x2, new_y2)

    # RIGHT → inherit HEIGHT (left edge of new near right edge of old)
    if (x1 - x4)**2 + (y1 - y3)**2 < threshold**2 or (x1 - x4)**2 + (y2 - y4)**2 < threshold**2:
        new_y1, new_y2 = y3, y4
        new_x1 = x4 + l
        new_x2 = new_x1 + width
        return (new_x1, new_y1, new_x2, new_y2)

    return None


# -------- MAIN LOOP --------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    result = landmarker.detect(mp_image)

    if result.hand_landmarks:
        h, w, _ = frame.shape
        hand = result.hand_landmarks[0]

        # -------- POINTS --------
        x1, y1 = int(hand[4].x * w), int(hand[4].y * h)
        x2, y2 = int(hand[8].x * w), int(hand[8].y * h)

        # -------- DISTANCE --------
        dist = ((x1 - x2)**2 + (y1 - y2)**2) ** 0.5

        if dist > 40:
            draw_counter += 1
            release_counter = 0
        else:
            release_counter += 1
            draw_counter = 0

        draw_mode = draw_counter >= DRAW_THRESHOLD

        # -------- DRAW --------
        if draw_mode:
            if temp_rect is None:
                temp_rect = (x1, y1, x2, y2)
            else:
                px1, py1, px2, py2 = temp_rect
                temp_rect = (
                    int(alpha * px1 + (1 - alpha) * x1),
                    int(alpha * py1 + (1 - alpha) * y1),
                    int(alpha * px2 + (1 - alpha) * x2),
                    int(alpha * py2 + (1 - alpha) * y2),
                )

        # -------- COMMIT --------
        if release_counter >= RELEASE_THRESHOLD and temp_rect is not None:

            nx1, ny1, nx2, ny2 = normalize(temp_rect)
            
            # Compensate for fingers closing during the commit motion
            pad = 20 
            new_rect = (nx1 - pad, ny1 - pad, nx2 + pad, ny2 + pad)

            # SNAP
            for rect in rectangles:
                snapped = snap_rect(new_rect, rect)
                if snapped is not None:
                    new_rect = normalize(snapped)
                    break

            # OVERLAP CHECK
            overlap = False
            for rect in rectangles:
                if is_overlap(new_rect, rect):
                    overlap = True
                    break

            if not overlap:
                rectangles.append(new_rect)

            temp_rect = None

        # -------- FIST DELETE --------
        if is_fist(hand):
            hx, hy = int(hand[9].x * w), int(hand[9].y * h)
            # Remove any rectangle containing the hand midpoint
            rectangles = [
                r for r in rectangles 
                if not (r[0] < hx < r[2] and r[1] < hy < r[3])
            ]

    # -------- DRAW STORED --------
    for x1, y1, x2, y2 in rectangles:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # -------- DRAW TEMP --------
    if temp_rect is not None:
        x1, y1, x2, y2 = normalize(temp_rect)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imshow("Hand Drawing System", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()