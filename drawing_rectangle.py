import cv2
import mediapipe as mp
import numpy as np

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

rectangles = []
current_rect = None
locked = False


# -------- NORMALIZE RECT --------
def normalize(rect):
    x1, y1, x2, y2 = rect
    return min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)


# -------- OVERLAP CHECK --------
def is_overlap(r1, r2):
    x1, y1, x2, y2 = r1
    x3, y3, x4, y4 = r2

    return not (x2 < x3 or x4 < x1 or y2 < y3 or y4 < y1)

l=2
# -------- SNAP FUNCTION --------
def snap_rect(new_rect, old_rect, threshold=20):
    x1, y1, x2, y2 = new_rect
    x3, y3, x4, y4 = old_rect

    width = abs(x3 - x4)
    height = abs(y3 - y4)

    # ABOVE → inherit WIDTH
    if (x1 - x3)**2 + (y1 - y4)**2 < threshold**2:
        width = x4 - x3  # inherit
        new_x1 = x3
        new_x2 = x4

        new_y2 = y4 - l
        new_y1 = new_y2 - (y2 - y1)  # keep height

        return (new_x1, new_y1, new_x2, new_y2)


    # BELOW → inherit WIDTH
    if (x1 - x3)**2 + (y2 - y3)**2 < threshold**2:
        width = x4 - x3
        new_x1 = x3
        new_x2 = x4

        new_y1 = y3 + l
        new_y2 = new_y1 + (y2 - y1)

        return (new_x1, new_y1, new_x2, new_y2)


    # LEFT → inherit HEIGHT
    if (x2 - x3)**2 + (y1 - y3)**2 < threshold**2:
        height = y4 - y3  # inherit
        new_y1 = y3
        new_y2 = y4

        new_x2 = x3 - l
        new_x1 = new_x2 - (x2 - x1)

        return (new_x1, new_y1, new_x2, new_y2)


    # RIGHT → inherit HEIGHT
    if (x1 - x4)**2 + (y1 - y3)**2 < threshold**2:
        height = y4 - y3
        new_y1 = y3
        new_y2 = y4

        new_x1 = x4 + l
        new_x2 = new_x1 + (x2 - x1)

        return (new_x1, new_y1, new_x2, new_y2)

    return None


# -------- MAIN LOOP --------
alpha = 0.7  # smoothing factor

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

        # -------- GESTURES --------
        thumb_up  = hand[4].y < hand[3].y
        index_up  = hand[8].y < hand[6].y
        middle_up = hand[12].y < hand[10].y
        ring_up   = hand[16].y < hand[14].y
        pinky_up  = hand[20].y < hand[18].y

        draw_mode = thumb_up and index_up
        fist = not index_up and not middle_up and not ring_up and not pinky_up

        # -------- DRAW --------
        if draw_mode:
            x1, y1 = int(hand[4].x * w), int(hand[4].y * h)
            x2, y2 = int(hand[8].x * w), int(hand[8].y * h)

            if current_rect is None:
                current_rect = (x1, y1, x2, y2)
            else:
                px1, py1, px2, py2 = current_rect
                current_rect = (
                    int(alpha * px1 + (1 - alpha) * x1),
                    int(alpha * py1 + (1 - alpha) * y1),
                    int(alpha * px2 + (1 - alpha) * x2),
                    int(alpha * py2 + (1 - alpha) * y2),
                )

        # -------- LOCK RECT --------
        if fist and current_rect is not None and not locked:

            current_rect = normalize(current_rect)

            # SNAP FIRST
            for rect in rectangles:
                snapped = snap_rect(current_rect, rect)
                if snapped is not None:
                    current_rect = normalize(snapped)
                    break

            # CHECK OVERLAP
            overlap = False
            for rect in rectangles:
                if is_overlap(current_rect, rect):
                    overlap = True
                    break

            # STORE
            if not overlap:
                rectangles.append(current_rect)

            current_rect = None
            locked = True

        if not fist:
            locked = False

    # -------- DRAW STORED --------
    for x1, y1, x2, y2 in rectangles:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # -------- DRAW CURRENT --------
    if current_rect is not None:
        x1, y1, x2, y2 = normalize(current_rect)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imshow("Hand Drawing System", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()