from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import mediapipe as mp


# ============================================================
# Configuration
# ============================================================

FACE_LANDMARKER_MODEL = "models/face_landmarker.task"


# ArcFace standard 112x112 five-point template
ARCFACE_TEMPLATE = np.array(
    [
        [38.2946, 51.6963],  # left eye
        [73.5318, 51.5014],  # right eye
        [56.0252, 71.7366],  # nose
        [41.5493, 92.3655],  # left mouth
        [70.7299, 92.2041],  # right mouth
    ],
    dtype=np.float32,
)


# MediaPipe landmark indices
LEFT_EYE = 33
RIGHT_EYE = 263
NOSE = 1
LEFT_MOUTH = 61
RIGHT_MOUTH = 291


# ============================================================
# Face data structure
# ============================================================

@dataclass
class FaceKpsBox:
    bbox: np.ndarray
    kps: np.ndarray


# ============================================================
# Utility functions
# ============================================================

def _clip_box_xyxy(
    box: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray:

    x1, y1, x2, y2 = box.astype(int)

    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(0, min(x2, width - 1))
    y2 = max(0, min(y2, height - 1))

    return np.array(
        [x1, y1, x2, y2],
        dtype=np.int32,
    )


def _bbox_from_5pt(
    kps: np.ndarray,
) -> np.ndarray:

    x_min = float(np.min(kps[:, 0]))
    y_min = float(np.min(kps[:, 1]))
    x_max = float(np.max(kps[:, 0]))
    y_max = float(np.max(kps[:, 1]))

    width = x_max - x_min
    height = y_max - y_min

    pad_x = max(20.0, width * 0.65)
    pad_y = max(25.0, height * 0.85)

    return np.array(
        [
            x_min - pad_x,
            y_min - pad_y,
            x_max + pad_x,
            y_max + pad_y,
        ],
        dtype=np.float32,
    )


def _kps_span_ok(
    kps: np.ndarray,
) -> bool:

    if kps.shape != (5, 2):
        return False

    x_span = float(np.ptp(kps[:, 0]))
    y_span = float(np.ptp(kps[:, 1]))

    return (
        x_span >= 8.0
        and y_span >= 8.0
    )


# ============================================================
# ArcFace alignment
# ============================================================

def _estimate_norm_5pt(
    kps: np.ndarray,
    image_size: int = 112,
) -> np.ndarray:

    kps = np.asarray(
        kps,
        dtype=np.float32,
    ).reshape(5, 2)

    dst = ARCFACE_TEMPLATE.copy()

    if image_size != 112:
        dst *= float(image_size) / 112.0

    matrix, _ = cv2.estimateAffinePartial2D(
        kps,
        dst,
        method=cv2.LMEDS,
    )

    if matrix is None:

        matrix = cv2.getAffineTransform(
            kps[:3],
            dst[:3],
        )

    return matrix.astype(np.float32)


def align_face_5pt(
    frame: np.ndarray,
    kps: np.ndarray,
    image_size: int = 112,
) -> np.ndarray:

    matrix = _estimate_norm_5pt(
        kps,
        image_size,
    )

    aligned = cv2.warpAffine(
        frame,
        matrix,
        (
            image_size,
            image_size,
        ),
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )

    return aligned


# ============================================================
# MediaPipe Face Landmarker
# ============================================================

class FaceLandmarkerWrapper:

    def __init__(
        self,
        model_path: str = FACE_LANDMARKER_MODEL,
        num_faces: int = 5,
    ):

        BaseOptions = mp.tasks.BaseOptions

        FaceLandmarker = (
            mp.tasks.vision.FaceLandmarker
        )

        FaceLandmarkerOptions = (
            mp.tasks.vision.FaceLandmarkerOptions
        )

        RunningMode = (
            mp.tasks.vision.RunningMode
        )

        options = FaceLandmarkerOptions(
            base_options=BaseOptions(
                model_asset_path=str(model_path)
            ),
            running_mode=RunningMode.IMAGE,

            # IMPORTANT:
            # Allow multiple people.
            num_faces=num_faces,

            # Lower thresholds make detection
            # more tolerant in normal webcam conditions.
            min_face_detection_confidence=0.3,
            min_face_presence_confidence=0.3,
            min_tracking_confidence=0.3,

            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )

        self.landmarker = (
            FaceLandmarker.create_from_options(
                options
            )
        )

    def detect(
        self,
        image_bgr: np.ndarray,
    ):

        if (
            image_bgr is None
            or image_bgr.size == 0
        ):
            return []

        image_rgb = cv2.cvtColor(
            image_bgr,
            cv2.COLOR_BGR2RGB,
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=image_rgb,
        )

        result = self.landmarker.detect(
            mp_image
        )

        if not result.face_landmarks:
            return []

        return result.face_landmarks

    def close(self):

        try:
            self.landmarker.close()
        except Exception:
            pass


# ============================================================
# MediaPipe 5-point detector
# ============================================================

class Haar5ptDetector:

    """
    Face detector compatible with the rest of the project.

    Despite the historical class name Haar5ptDetector,
    detection is now performed directly by MediaPipe.

    Pipeline:

        Camera frame
              ↓
        MediaPipe Face Landmarker
              ↓
        five facial landmarks
              ↓
        bounding box
              ↓
        ArcFace alignment
    """

    def __init__(
        self,
        cascade_path: str | None = None,
        min_size: tuple[int, int] = (60, 60),
        smooth_alpha: float = 0.0,
        debug: bool = False,
    ):

        self.min_size = min_size
        self.smooth_alpha = smooth_alpha
        self.debug = debug

        self.landmarker = FaceLandmarkerWrapper(
            FACE_LANDMARKER_MODEL,
            num_faces=5,
        )

    # --------------------------------------------------------
    # Extract five points
    # --------------------------------------------------------

    def _extract_5pt(
        self,
        landmarks,
        image_width: int,
        image_height: int,
    ) -> np.ndarray | None:

        indices = [
            LEFT_EYE,
            RIGHT_EYE,
            NOSE,
            LEFT_MOUTH,
            RIGHT_MOUTH,
        ]

        points = []

        for index in indices:

            lm = landmarks[index]

            points.append(
                [
                    lm.x * image_width,
                    lm.y * image_height,
                ]
            )

        points = np.asarray(
            points,
            dtype=np.float32,
        )

        if not _kps_span_ok(points):
            return None

        return points

    # --------------------------------------------------------
    # Main detector
    # --------------------------------------------------------

    def detect(
        self,
        frame: np.ndarray,
        max_faces: int = 5,
    ) -> list[FaceKpsBox]:

        if (
            frame is None
            or frame.size == 0
        ):
            return []

        height, width = frame.shape[:2]

        all_landmarks = (
            self.landmarker.detect(frame)
        )

        if not all_landmarks:
            return []

        results = []

        for landmarks in all_landmarks[:max_faces]:

            kps = self._extract_5pt(
                landmarks,
                width,
                height,
            )

            if kps is None:
                continue

            bbox = _bbox_from_5pt(kps)

            bbox = _clip_box_xyxy(
                bbox,
                width,
                height,
            )

            results.append(
                FaceKpsBox(
                    bbox=bbox,
                    kps=kps,
                )
            )

        return results

    def close(self):

        self.landmarker.close()


# ============================================================
# Camera test
# ============================================================

def main():

    print("=" * 65)
    print("MEDIA PIPE 5-POINT DETECTOR TEST")
    print("=" * 65)

    print(
        f"MediaPipe: {mp.__version__}"
    )

    print(
        "Opening external USB camera: index 2"
    )

    cap = cv2.VideoCapture(
        2,
        cv2.CAP_DSHOW,
    )

    cap.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        640,
    )

    cap.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        480,
    )

    if not cap.isOpened():

        raise RuntimeError(
            "Could not open external USB camera index 2."
        )

    print(
        "Camera opened successfully."
    )

    detector = Haar5ptDetector(
        min_size=(60, 60),
        smooth_alpha=0.0,
        debug=False,
    )

    print(
        "Detector initialized."
    )

    print()
    print(
        "Show your face to the camera."
    )

    print(
        "Five red points should appear."
    )

    print(
        "Press Q to quit."
    )

    try:

        while True:

            ok, frame = cap.read()

            if not ok:
                continue

            faces = detector.detect(
                frame,
                max_faces=5,
            )

            for index, face in enumerate(faces):

                x1, y1, x2, y2 = (
                    face.bbox.astype(int)
                )

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2,
                )

                for point in face.kps:

                    px, py = point.astype(int)

                    cv2.circle(
                        frame,
                        (px, py),
                        5,
                        (0, 0, 255),
                        -1,
                    )

                cv2.putText(
                    frame,
                    f"Face {index + 1}",
                    (
                        x1,
                        max(25, y1 - 10),
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )

            cv2.putText(
                frame,
                f"Faces detected: {len(faces)}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )

            cv2.putText(
                frame,
                "Q = Quit",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

            cv2.imshow(
                "MediaPipe 5-Point Detector",
                frame,
            )

            key = (
                cv2.waitKey(1)
                & 0xFF
            )

            if key == ord("q"):
                break

    finally:

        cap.release()
        cv2.destroyAllWindows()
        detector.close()


if __name__ == "__main__":
    main()