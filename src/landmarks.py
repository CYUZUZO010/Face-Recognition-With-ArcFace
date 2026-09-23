"""
Face landmarks + blink counting + smile detection

Pipeline:
camera
    -> Haar face detection
    -> MediaPipe Face Landmarker
    -> 5 keypoints for ArcFace alignment
    -> eye landmarks for blink detection
    -> mouth landmarks for smile detection

Run:
    python -m src.landmarks

Keys:
    q : quit
"""

import cv2
import numpy as np
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ============================================================
# CAMERA SETTINGS
# ============================================================

CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480


# ============================================================
# MEDIAPIPE MODEL
# ============================================================

MODEL_PATH = "models/face_landmarker.task"


# ============================================================
# 5 POINTS USED BY YOUR ARCFACE ALIGNMENT
# ============================================================

IDX_LEFT_EYE = 33
IDX_RIGHT_EYE = 263
IDX_NOSE_TIP = 1
IDX_MOUTH_LEFT = 61
IDX_MOUTH_RIGHT = 291


# ============================================================
# EYE LANDMARKS FOR BLINK DETECTION
# ============================================================

LEFT_EYE = [
    33,
    160,
    158,
    133,
    153,
    144,
]

RIGHT_EYE = [
    362,
    385,
    387,
    263,
    373,
    380,
]


# ============================================================
# MOUTH LANDMARKS
# ============================================================

MOUTH_LEFT = 61
MOUTH_RIGHT = 291
MOUTH_TOP = 13
MOUTH_BOTTOM = 14


# ============================================================
# THRESHOLDS
# ============================================================

# Lower EAR means the eye is more closed.
EAR_THRESHOLD = 0.20

# Minimum number of consecutive frames that must show
# closed eyes before we consider it a real blink.
BLINK_MIN_FRAMES = 2


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def distance(p1, p2):
    """
    Calculate Euclidean distance between two 2D points.
    """
    return np.linalg.norm(p1 - p2)


def eye_aspect_ratio(points):
    """
    Calculate Eye Aspect Ratio (EAR).

    EAR = (vertical distance 1 + vertical distance 2)
          / (2 * horizontal distance)

    Higher EAR  -> eye open
    Lower EAR   -> eye closed
    """

    p1, p2, p3, p4, p5, p6 = points

    vertical_1 = distance(p2, p6)
    vertical_2 = distance(p3, p5)

    horizontal = distance(p1, p4)

    if horizontal == 0:
        return 0.0

    ear = (vertical_1 + vertical_2) / (2.0 * horizontal)

    return float(ear)


def get_mouth_ratio(landmarks, width, height):
    """
    Calculate mouth opening ratio.

    This is used as a simple mouth-expression measurement.

    Higher value:
        mouth is more open

    Lower value:
        mouth is more closed
    """

    left = np.array(
        [
            landmarks[MOUTH_LEFT].x * width,
            landmarks[MOUTH_LEFT].y * height,
        ],
        dtype=np.float32,
    )

    right = np.array(
        [
            landmarks[MOUTH_RIGHT].x * width,
            landmarks[MOUTH_RIGHT].y * height,
        ],
        dtype=np.float32,
    )

    top = np.array(
        [
            landmarks[MOUTH_TOP].x * width,
            landmarks[MOUTH_TOP].y * height,
        ],
        dtype=np.float32,
    )

    bottom = np.array(
        [
            landmarks[MOUTH_BOTTOM].x * width,
            landmarks[MOUTH_BOTTOM].y * height,
        ],
        dtype=np.float32,
    )

    mouth_width = distance(left, right)
    mouth_height = distance(top, bottom)

    if mouth_width == 0:
        return 0.0

    return float(mouth_height / mouth_width)


def landmark_to_pixel(landmark, width, height):
    """
    Convert MediaPipe normalized coordinates to pixel coordinates.
    """

    x = int(landmark.x * width)
    y = int(landmark.y * height)

    return x, y


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # Haar face detector
    # --------------------------------------------------------

    cascade_path = (
        cv2.data.haarcascades
        + "haarcascade_frontalface_default.xml"
    )

    face = cv2.CascadeClassifier(cascade_path)

    if face.empty():
        raise RuntimeError(
            f"Failed to load cascade: {cascade_path}"
        )

    # --------------------------------------------------------
    # MediaPipe Face Landmarker
    # --------------------------------------------------------

    base_options = python.BaseOptions(
        model_asset_path=MODEL_PATH
    )

    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )

    landmarker = vision.FaceLandmarker.create_from_options(
        options
    )

    # --------------------------------------------------------
    # Camera
    # --------------------------------------------------------

    cap = cv2.VideoCapture(
        CAMERA_INDEX,
        cv2.CAP_DSHOW,
    )

    if not cap.isOpened():
        raise RuntimeError(
            "Camera not opened. Try camera index 0/1/2."
        )

    cap.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        FRAME_WIDTH,
    )

    cap.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        FRAME_HEIGHT,
    )

    print()
    print("==============================================")
    print("Face Landmarks + Blink + Smile Detection")
    print("==============================================")
    print()
    print("Camera opened.")
    print("Press Q to quit.")
    print()

    # --------------------------------------------------------
    # Blink state
    # --------------------------------------------------------

    blink_count = 0

    eyes_closed = False
    closed_frames = 0

    # --------------------------------------------------------
    # VIDEO TIMESTAMP
    # --------------------------------------------------------

    timestamp_ms = 0

    # --------------------------------------------------------
    # MAIN CAMERA LOOP
    # --------------------------------------------------------

    while True:

        ok, frame = cap.read()

        if not ok:
            print("Failed to read camera frame.")
            break

        # Mirror webcam
        frame = cv2.flip(frame, 1)

        H, W = frame.shape[:2]

        # ----------------------------------------------------
        # Haar detection
        # ----------------------------------------------------

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY,
        )

        faces = face.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(60, 60),
        )

        # Draw Haar face boxes
        for i, (x, y, w, h) in enumerate(
            faces,
            start=1,
        ):

            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2,
            )

            cv2.putText(
                frame,
                f"Face {i}",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

        # ----------------------------------------------------
        # Convert OpenCV frame to MediaPipe image
        # ----------------------------------------------------

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB,
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb,
        )

        # ----------------------------------------------------
        # Detect landmarks
        # ----------------------------------------------------

        result = landmarker.detect_for_video(
            mp_image,
            timestamp_ms,
        )

        timestamp_ms += 33

        # ----------------------------------------------------
        # Process first detected face
        # ----------------------------------------------------

        if result.face_landmarks:

            landmarks = result.face_landmarks[0]

            # =================================================
            # 5 POINTS FOR ARCFACE
            # =================================================

            idxs = [
                IDX_LEFT_EYE,
                IDX_RIGHT_EYE,
                IDX_NOSE_TIP,
                IDX_MOUTH_LEFT,
                IDX_MOUTH_RIGHT,
            ]

            kps = []

            for index in idxs:

                p = landmarks[index]

                px = p.x * W
                py = p.y * H

                kps.append(
                    [px, py]
                )

            kps = np.array(
                kps,
                dtype=np.float32,
            )

            # Ensure left/right ordering
            if kps[0, 0] > kps[1, 0]:
                kps[[0, 1]] = kps[[1, 0]]

            if kps[3, 0] > kps[4, 0]:
                kps[[3, 4]] = kps[[4, 3]]

            # Draw 5 ArcFace points
            for px, py in kps.astype(int):

                cv2.circle(
                    frame,
                    (int(px), int(py)),
                    4,
                    (0, 255, 0),
                    -1,
                )

            # =================================================
            # BLINK DETECTION
            # =================================================

            left_eye_points = []

            for index in LEFT_EYE:

                p = landmarks[index]

                left_eye_points.append(
                    np.array(
                        [
                            p.x * W,
                            p.y * H,
                        ],
                        dtype=np.float32,
                    )
                )

            right_eye_points = []

            for index in RIGHT_EYE:

                p = landmarks[index]

                right_eye_points.append(
                    np.array(
                        [
                            p.x * W,
                            p.y * H,
                        ],
                        dtype=np.float32,
                    )
                )

            left_ear = eye_aspect_ratio(
                left_eye_points
            )

            right_ear = eye_aspect_ratio(
                right_eye_points
            )

            ear = (
                left_ear + right_ear
            ) / 2.0

            # -----------------------------------------------
            # Determine whether eyes are closed
            # -----------------------------------------------

            if ear < EAR_THRESHOLD:

                closed_frames += 1

            else:

                # Eyes have opened again.
                # Check whether the previous state
                # was a real blink.

                if (
                    eyes_closed
                    and closed_frames >= BLINK_MIN_FRAMES
                ):
                    blink_count += 1

                closed_frames = 0
                eyes_closed = False

            # If eyes remain closed for enough frames,
            # remember that a closed-eye event occurred.

            if closed_frames >= BLINK_MIN_FRAMES:

                eyes_closed = True

            # =================================================
            # SMILE / MOUTH DETECTION
            # =================================================

            mouth_ratio = get_mouth_ratio(
                landmarks,
                W,
                H,
            )

            # Simple first-stage smile indicator.
            #
            # IMPORTANT:
            # This is a mouth-opening measurement,
            # so it will need calibration.
            #
            # We intentionally display the value first
            # instead of pretending one threshold works
            # perfectly for every face.

            smile_threshold = 0.30

            smiling = (
                mouth_ratio > smile_threshold
            )

            # =================================================
            # DRAW MOUTH LANDMARKS
            # =================================================

            mouth_indices = [
                MOUTH_LEFT,
                MOUTH_RIGHT,
                MOUTH_TOP,
                MOUTH_BOTTOM,
            ]

            for index in mouth_indices:

                p = landmarks[index]

                px, py = landmark_to_pixel(
                    p,
                    W,
                    H,
                )

                cv2.circle(
                    frame,
                    (px, py),
                    4,
                    (255, 0, 255),
                    -1,
                )

            # =================================================
            # DISPLAY INFORMATION
            # =================================================

            cv2.putText(
                frame,
                f"Blinks: {blink_count}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )

            cv2.putText(
                frame,
                f"EAR: {ear:.3f}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

            cv2.putText(
                frame,
                f"Mouth: {mouth_ratio:.3f}",
                (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

            if smiling:

                smile_text = "Smile: YES"

            else:

                smile_text = "Smile: NO"

            cv2.putText(
                frame,
                smile_text,
                (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
            )

        else:

            # No face detected.
            #
            # Reset temporary blink state so that
            # leaving/re-entering the camera does not
            # create a false blink.

            closed_frames = 0
            eyes_closed = False

            cv2.putText(
                frame,
                "No face detected",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
            )

        # ----------------------------------------------------
        # General information
        # ----------------------------------------------------

        cv2.putText(
            frame,
            f"Faces detected: {len(faces)}",
            (10, H - 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        cv2.putText(
            frame,
            "Press Q to quit",
            (10, H - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

        # ----------------------------------------------------
        # Show camera
        # ----------------------------------------------------

        cv2.imshow(
            "Face Landmarks + Blink + Smile",
            frame,
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    # --------------------------------------------------------
    # Cleanup
    # --------------------------------------------------------

    cap.release()

    cv2.destroyAllWindows()

    landmarker.close()


if __name__ == "__main__":
    main()
