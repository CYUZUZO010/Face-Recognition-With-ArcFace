from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .haar_5pt import (
    Haar5ptDetector,
    align_face_5pt,
)


# ============================================================
# Configuration
# ============================================================

DB_PATH = Path("data/db/face_db.npz")

MODEL_PATH = Path(
    "models/embedder_arcface.onnx"
)

# External USB camera
CAMERA_INDEX = 2

# ------------------------------------------------------------
# Recognition threshold
#
# Lower distance = more similar.
#
# A match is accepted when:
#
#     distance <= THRESHOLD
# ------------------------------------------------------------

THRESHOLD = 0.34

# Maximum number of faces to process
MAX_FACES = 5

# ArcFace input size
IMAGE_SIZE = 112

# ArcFace embedding dimension
EMBEDDING_DIM = 512

# ------------------------------------------------------------
# Performance
#
# ArcFace is expensive to run on every frame.
#
# With 5:
#
# Frame 1 -> detection
# Frame 2 -> detection
# Frame 3 -> detection
# Frame 4 -> detection
# Frame 5 -> detection + ArcFace recognition
#
# Then repeat.
# ------------------------------------------------------------

RECOGNITION_INTERVAL = 5


# ============================================================
# Face Match Result
# ============================================================

@dataclass
class MatchResult:

    name: str

    similarity: float

    distance: float

    accepted: bool


# ============================================================
# ArcFace ONNX Embedder
# ============================================================

class ArcFaceEmbedderONNX:

    def __init__(
        self,
        model_path: str | Path,
    ):

        import onnxruntime as ort

        self.model_path = str(
            model_path
        )

        # ----------------------------------------------------
        # Check model exists
        # ----------------------------------------------------

        if not Path(
            self.model_path
        ).exists():

            raise FileNotFoundError(
                f"ArcFace model not found:\n"
                f"{self.model_path}"
            )

        # ----------------------------------------------------
        # Load ONNX model
        # ----------------------------------------------------

        self.session = (
            ort.InferenceSession(
                self.model_path,
                providers=[
                    "CPUExecutionProvider"
                ],
            )
        )

        # ----------------------------------------------------
        # Get input/output names
        # ----------------------------------------------------

        self.input_name = (
            self.session
            .get_inputs()[0]
            .name
        )

        self.output_name = (
            self.session
            .get_outputs()[0]
            .name
        )

        # ----------------------------------------------------
        # Print model information
        # ----------------------------------------------------

        input_shape = (
            self.session
            .get_inputs()[0]
            .shape
        )

        output_shape = (
            self.session
            .get_outputs()[0]
            .shape
        )

        print(
            "ArcFace model loaded."
        )

        print(
            f"Input shape: {input_shape}"
        )

        print(
            f"Output shape: {output_shape}"
        )

    # ========================================================
    # Generate embedding
    # ========================================================

    def embed(
        self,
        image_bgr: np.ndarray,
    ) -> np.ndarray:

        # ----------------------------------------------------
        # Validate image
        # ----------------------------------------------------

        if (
            image_bgr is None
            or image_bgr.size == 0
        ):

            raise ValueError(
                "Input face image is empty."
            )

        # ----------------------------------------------------
        # Resize to 112 x 112
        # ----------------------------------------------------

        image = cv2.resize(
            image_bgr,
            (
                IMAGE_SIZE,
                IMAGE_SIZE,
            ),
        )

        # ----------------------------------------------------
        # Convert BGR -> RGB
        # ----------------------------------------------------

        rgb = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB,
        )

        # ----------------------------------------------------
        # Convert to float32
        # ----------------------------------------------------

        rgb = rgb.astype(
            np.float32
        )

        # ----------------------------------------------------
        # ArcFace normalization
        #
        # (pixel - 127.5) / 128
        # ----------------------------------------------------

        rgb = (
            rgb - 127.5
        ) / 128.0

        # ----------------------------------------------------
        # HWC -> CHW
        # ----------------------------------------------------

        tensor = np.transpose(
            rgb,
            (2, 0, 1),
        )

        # ----------------------------------------------------
        # Add batch dimension
        #
        # CHW -> NCHW
        # ----------------------------------------------------

        tensor = np.expand_dims(
            tensor,
            axis=0,
        ).astype(
            np.float32
        )

        # ----------------------------------------------------
        # Run ArcFace
        # ----------------------------------------------------

        output = self.session.run(
            [self.output_name],
            {
                self.input_name: tensor
            },
        )[0]

        # ----------------------------------------------------
        # Flatten output
        # ----------------------------------------------------

        embedding = np.asarray(
            output[0],
            dtype=np.float32,
        ).reshape(-1)

        # ----------------------------------------------------
        # Verify embedding dimension
        # ----------------------------------------------------

        if embedding.size != EMBEDDING_DIM:

            raise ValueError(
                f"Unexpected ArcFace embedding "
                f"dimension: {embedding.size}. "
                f"Expected {EMBEDDING_DIM}."
            )

        # ----------------------------------------------------
        # L2 normalization
        # ----------------------------------------------------

        norm = np.linalg.norm(
            embedding
        )

        if norm < 1e-8:

            raise ValueError(
                "ArcFace produced a zero embedding."
            )

        embedding = (
            embedding / norm
        )

        return embedding.astype(
            np.float32
        )


# ============================================================
# Load Face Database
# ============================================================

def load_db(
    db_path: Path,
) -> dict[str, np.ndarray]:

    print()

    print("=" * 70)

    print(
        "LOADING FACE DATABASE"
    )

    print("=" * 70)

    # --------------------------------------------------------
    # Check database
    # --------------------------------------------------------

    if not db_path.exists():

        print(
            "ERROR: Database not found:"
        )

        print(
            db_path
        )

        return {}

    print(
        f"Database: {db_path}"
    )

    # --------------------------------------------------------
    # Load NPZ database
    # --------------------------------------------------------

    data = np.load(
        db_path,
        allow_pickle=True,
    )

    print(
        f"Database arrays: {data.files}"
    )

    # --------------------------------------------------------
    # Check names
    # --------------------------------------------------------

    if "names" not in data.files:

        print(
            "ERROR: Database does not contain "
            "'names'."
        )

        return {}

    # --------------------------------------------------------
    # Check embeddings
    # --------------------------------------------------------

    if "embeddings" not in data.files:

        print(
            "ERROR: Database does not contain "
            "'embeddings'."
        )

        return {}

    # --------------------------------------------------------
    # Read names
    # --------------------------------------------------------

    names_array = data["names"]

    names = [
        str(name)
        for name in names_array.tolist()
    ]

    # --------------------------------------------------------
    # Read embeddings
    # --------------------------------------------------------

    embeddings = np.asarray(
        data["embeddings"],
        dtype=np.float32,
    )

    # --------------------------------------------------------
    # Make embeddings 2D
    # --------------------------------------------------------

    if embeddings.ndim == 1:

        embeddings = embeddings.reshape(
            1,
            -1,
        )

    # --------------------------------------------------------
    # Validate dimensions
    # --------------------------------------------------------

    if embeddings.ndim != 2:

        print(
            "ERROR: Embeddings must be a "
            "2-dimensional array."
        )

        print(
            f"Shape: {embeddings.shape}"
        )

        return {}

    # --------------------------------------------------------
    # Validate 512 dimensions
    # --------------------------------------------------------

    if embeddings.shape[1] != EMBEDDING_DIM:

        print(
            "ERROR: Wrong embedding dimension."
        )

        print(
            f"Expected: {EMBEDDING_DIM}"
        )

        print(
            f"Actual: {embeddings.shape[1]}"
        )

        return {}

    # --------------------------------------------------------
    # Validate number of people
    # --------------------------------------------------------

    if len(names) != embeddings.shape[0]:

        print(
            "ERROR: Number of names does not "
            "match number of embeddings."
        )

        print(
            f"Names: {len(names)}"
        )

        print(
            f"Embeddings: {embeddings.shape[0]}"
        )

        return {}

    # --------------------------------------------------------
    # Build database dictionary
    # --------------------------------------------------------

    db: dict[str, np.ndarray] = {}

    for index, name in enumerate(names):

        embedding = embeddings[index]

        # ----------------------------------------------------
        # Calculate norm
        # ----------------------------------------------------

        norm = np.linalg.norm(
            embedding
        )

        # ----------------------------------------------------
        # Skip invalid embedding
        # ----------------------------------------------------

        if norm < 1e-8:

            print(
                f"WARNING: Skipping invalid "
                f"embedding for '{name}'."
            )

            continue

        # ----------------------------------------------------
        # Normalize embedding
        # ----------------------------------------------------

        embedding = (
            embedding / norm
        ).astype(
            np.float32
        )

        # ----------------------------------------------------
        # Store
        # ----------------------------------------------------

        db[name] = embedding

    # --------------------------------------------------------
    # Print loaded people
    # --------------------------------------------------------

    print(
        f"Identities loaded: {len(db)}"
    )

    if db:

        print(
            "Known people:"
        )

        for name in db:

            print(
                f"  - {name}"
            )

    print("=" * 70)

    return db


# ============================================================
# Face Database Matcher
# ============================================================

class FaceDBMatcher:

    def __init__(
        self,
        db: dict[str, np.ndarray],
        threshold: float = 0.34,
    ):

        self.db = db

        self.threshold = threshold

        self.names = list(
            db.keys()
        )

        # ----------------------------------------------------
        # Build matrix
        #
        # Example:
        #
        # 3 people:
        #
        # (3, 512)
        # ----------------------------------------------------

        if self.names:

            self.matrix = np.stack(
                [
                    db[name]
                    for name in self.names
                ],
                axis=0,
            ).astype(
                np.float32
            )

        else:

            self.matrix = np.empty(
                (
                    0,
                    EMBEDDING_DIM,
                ),
                dtype=np.float32,
            )

    # ========================================================
    # Match embedding
    # ========================================================

    def match(
        self,
        embedding: np.ndarray,
    ) -> MatchResult:

        # ----------------------------------------------------
        # No enrolled people
        # ----------------------------------------------------

        if len(self.names) == 0:

            return MatchResult(
                name="Unknown",
                similarity=0.0,
                distance=1.0,
                accepted=False,
            )

        # ----------------------------------------------------
        # Normalize incoming embedding
        # ----------------------------------------------------

        embedding = np.asarray(
            embedding,
            dtype=np.float32,
        ).reshape(-1)

        norm = np.linalg.norm(
            embedding
        )

        if norm < 1e-8:

            return MatchResult(
                name="Unknown",
                similarity=0.0,
                distance=1.0,
                accepted=False,
            )

        embedding = (
            embedding / norm
        )

        # ----------------------------------------------------
        # Cosine similarity
        #
        # Both vectors are normalized.
        #
        # Therefore:
        #
        # dot product = cosine similarity
        # ----------------------------------------------------

        similarities = (
            self.matrix
            @ embedding
        )

        # ----------------------------------------------------
        # Find best match
        # ----------------------------------------------------

        best_index = int(
            np.argmax(
                similarities
            )
        )

        similarity = float(
            similarities[
                best_index
            ]
        )

        # ----------------------------------------------------
        # Convert similarity to distance
        # ----------------------------------------------------

        distance = (
            1.0 - similarity
        )

        # ----------------------------------------------------
        # Candidate name
        # ----------------------------------------------------

        candidate_name = (
            self.names[
                best_index
            ]
        )

        # ----------------------------------------------------
        # Threshold decision
        # ----------------------------------------------------

        accepted = (
            distance <= self.threshold
        )

        if accepted:

            name = candidate_name

        else:

            name = "Unknown"

        return MatchResult(
            name=name,
            similarity=similarity,
            distance=distance,
            accepted=accepted,
        )


# ============================================================
# Draw Face Information
# ============================================================

def draw_face(
    frame: np.ndarray,
    face,
    result: MatchResult,
):

    # --------------------------------------------------------
    # Bounding box
    # --------------------------------------------------------

    x1, y1, x2, y2 = (
        face.bbox.astype(int)
    )

    # --------------------------------------------------------
    # Frame dimensions
    # --------------------------------------------------------

    height, width = frame.shape[:2]

    # --------------------------------------------------------
    # Keep coordinates inside frame
    # --------------------------------------------------------

    x1 = max(
        0,
        min(x1, width - 1),
    )

    y1 = max(
        0,
        min(y1, height - 1),
    )

    x2 = max(
        0,
        min(x2, width - 1),
    )

    y2 = max(
        0,
        min(y2, height - 1),
    )

    # --------------------------------------------------------
    # Box color
    #
    # Green = recognized
    # Red   = unknown
    # --------------------------------------------------------

    if result.accepted:

        box_color = (
            0,
            255,
            0,
        )

    else:

        box_color = (
            0,
            0,
            255,
        )

    # --------------------------------------------------------
    # Draw bounding box
    # --------------------------------------------------------

    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        box_color,
        2,
    )

    # --------------------------------------------------------
    # Draw five landmarks
    # --------------------------------------------------------

    for point in face.kps:

        px, py = (
            point.astype(int)
        )

        cv2.circle(
            frame,
            (px, py),
            4,
            (0, 0, 255),
            -1,
        )

    # --------------------------------------------------------
    # Draw name
    # --------------------------------------------------------

    label = result.name

    label_y = max(
        25,
        y1 - 10,
    )

    cv2.putText(
        frame,
        label,
        (
            x1,
            label_y,
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        box_color,
        2,
        cv2.LINE_AA,
    )

    # --------------------------------------------------------
    # Draw similarity
    # --------------------------------------------------------

    similarity_text = (
        f"similarity: "
        f"{result.similarity:.3f}"
    )

    cv2.putText(
        frame,
        similarity_text,
        (
            x1,
            min(
                height - 35,
                y2 + 20,
            ),
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        box_color,
        1,
        cv2.LINE_AA,
    )

    # --------------------------------------------------------
    # Draw distance
    # --------------------------------------------------------

    distance_text = (
        f"distance: "
        f"{result.distance:.3f}"
    )

    cv2.putText(
        frame,
        distance_text,
        (
            x1,
            min(
                height - 15,
                y2 + 38,
            ),
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        box_color,
        1,
        cv2.LINE_AA,
    )


# ============================================================
# Draw Face While Waiting for Recognition
# ============================================================

def draw_checking_face(
    frame: np.ndarray,
    face,
):

    # --------------------------------------------------------
    # Bounding box
    # --------------------------------------------------------

    x1, y1, x2, y2 = (
        face.bbox.astype(int)
    )

    height, width = frame.shape[:2]

    # --------------------------------------------------------
    # Keep coordinates inside frame
    # --------------------------------------------------------

    x1 = max(
        0,
        min(x1, width - 1),
    )

    y1 = max(
        0,
        min(y1, height - 1),
    )

    x2 = max(
        0,
        min(x2, width - 1),
    )

    y2 = max(
        0,
        min(y2, height - 1),
    )

    # --------------------------------------------------------
    # Yellow/cyan checking box
    # --------------------------------------------------------

    checking_color = (
        255,
        255,
        0,
    )

    # --------------------------------------------------------
    # Draw box
    # --------------------------------------------------------

    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        checking_color,
        2,
    )

    # --------------------------------------------------------
    # Draw five points
    # --------------------------------------------------------

    for point in face.kps:

        px, py = (
            point.astype(int)
        )

        cv2.circle(
            frame,
            (px, py),
            4,
            (0, 0, 255),
            -1,
        )

    # --------------------------------------------------------
    # Draw checking text
    # --------------------------------------------------------

    cv2.putText(
        frame,
        "Checking...",
        (
            x1,
            max(
                25,
                y1 - 10,
            ),
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        checking_color,
        2,
        cv2.LINE_AA,
    )


# ============================================================
# Main Recognition System
# ============================================================

def main():

    global THRESHOLD

    print()

    print("=" * 70)

    print(
        "LIVE MULTI-FACE ARCFACE RECOGNITION"
    )

    print("=" * 70)

    # ========================================================
    # 1. Load database
    # ========================================================

    db = load_db(
        DB_PATH
    )

    if not db:

        print()

        print(
            "WARNING: No valid identities "
            "were loaded."
        )

        print(
            "Run enrollment first:"
        )

        print(
            "python -m src.enroll"
        )

        print()

    # ========================================================
    # 2. Load MediaPipe detector
    # ========================================================

    print()

    print(
        "Initializing MediaPipe face detector..."
    )

    detector = Haar5ptDetector(
        min_size=(60, 60),
        smooth_alpha=0.0,
        debug=False,
    )

    print(
        "Face detector initialized."
    )

    # ========================================================
    # 3. Load ArcFace
    # ========================================================

    print()

    print(
        "Loading ArcFace ONNX model..."
    )

    embedder = ArcFaceEmbedderONNX(
        MODEL_PATH
    )

    print(
        "ArcFace initialized."
    )

    # ========================================================
    # 4. Create matcher
    # ========================================================

    matcher = FaceDBMatcher(
        db,
        threshold=THRESHOLD,
    )

    # ========================================================
    # 5. Open camera
    # ========================================================

    print()

    print(
        f"Opening external USB camera "
        f"index {CAMERA_INDEX}..."
    )

    cap = cv2.VideoCapture(
        CAMERA_INDEX,
        cv2.CAP_DSHOW,
    )

    # --------------------------------------------------------
    # Camera resolution
    # --------------------------------------------------------

    cap.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        640,
    )

    cap.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        480,
    )

    # --------------------------------------------------------
    # Check camera
    # --------------------------------------------------------

    if not cap.isOpened():

        detector.close()

        raise RuntimeError(
            f"Could not open external "
            f"USB camera index {CAMERA_INDEX}."
        )

    print(
        "Camera opened successfully."
    )

    # ========================================================
    # Controls
    # ========================================================

    print()

    print("=" * 70)

    print(
        "RECOGNITION CONTROLS"
    )

    print("=" * 70)

    print(
        "Q = quit"
    )

    print(
        "+ / = = increase distance threshold"
    )

    print(
        "- / _ = decrease distance threshold"
    )

    print()

    print(
        f"Current distance threshold: "
        f"{THRESHOLD:.3f}"
    )

    print(
        f"Recognition interval: "
        f"every {RECOGNITION_INTERVAL} frames"
    )

    print()

    print(
        "Look at the camera."
    )

    print(
        "The system will recognize enrolled "
        "people automatically."
    )

    print("=" * 70)

    # ========================================================
    # FPS variables
    # ========================================================

    previous_time = time.time()

    fps = 0.0

    # --------------------------------------------------------
    # Count camera frames
    # --------------------------------------------------------

    frame_count = 0

    # --------------------------------------------------------
    # Store most recent recognition results
    # --------------------------------------------------------
    #
    # Example:
    #
    # Face 1 -> cyuzuzo
    # Face 2 -> Unknown
    #
    # These results stay visible while we wait
    # for the next ArcFace recognition cycle.
    # --------------------------------------------------------

    last_results: list[MatchResult] = []

    # ========================================================
    # Recognition loop
    # ========================================================

    try:

        while True:

            # ------------------------------------------------
            # Read frame
            # ------------------------------------------------

            ok, frame = cap.read()

            if not ok:

                print(
                    "Could not read camera frame."
                )

                break

            # ------------------------------------------------
            # Increase frame counter
            # ------------------------------------------------

            frame_count += 1

            # ------------------------------------------------
            # Detect faces
            #
            # MediaPipe still runs continuously so the
            # camera display remains responsive.
            # ------------------------------------------------

            faces = detector.detect(
                frame,
                max_faces=MAX_FACES,
            )

            # =================================================
            # Run ArcFace recognition periodically
            # =================================================

            if (
                frame_count
                % RECOGNITION_INTERVAL
                == 0
            ):

                # ------------------------------------------------
                # New recognition cycle
                # ------------------------------------------------

                current_results: list[
                    MatchResult
                ] = []

                # ------------------------------------------------
                # Process every detected face
                # ------------------------------------------------

                for face_index, face in enumerate(
                    faces
                ):

                    try:

                        # ========================================
                        # Align face
                        # ========================================

                        aligned = (
                            align_face_5pt(
                                frame,
                                face.kps,
                                image_size=IMAGE_SIZE,
                            )
                        )

                        # ========================================
                        # Generate ArcFace embedding
                        # ========================================

                        embedding = (
                            embedder.embed(
                                aligned
                            )
                        )

                        # ========================================
                        # Compare with database
                        # ========================================

                        result = matcher.match(
                            embedding
                        )

                        current_results.append(
                            result
                        )

                    except Exception as e:

                        print(
                            f"Face {face_index + 1} "
                            f"processing error: {e}"
                        )

                        current_results.append(
                            MatchResult(
                                name="Unknown",
                                similarity=0.0,
                                distance=1.0,
                                accepted=False,
                            )
                        )

                # ------------------------------------------------
                # Save latest recognition results
                # ------------------------------------------------

                last_results = (
                    current_results
                )

            # =================================================
            # Draw results
            # =================================================

            recognized_count = 0

            # ------------------------------------------------
            # Draw every detected face
            # ------------------------------------------------

            for face_index, face in enumerate(
                faces
            ):

                # ------------------------------------------------
                # If we have a recognition result for this face
                # ------------------------------------------------

                if (
                    face_index
                    < len(last_results)
                ):

                    result = (
                        last_results[
                            face_index
                        ]
                    )

                    # --------------------------------------------
                    # Count recognized people
                    # --------------------------------------------

                    if result.accepted:

                        recognized_count += 1

                    # --------------------------------------------
                    # Draw recognized/unknown result
                    # --------------------------------------------

                    draw_face(
                        frame,
                        face,
                        result,
                    )

                else:

                    # ------------------------------------------------
                    # New face waiting for recognition
                    # ------------------------------------------------

                    draw_checking_face(
                        frame,
                        face,
                    )

            # =================================================
            # FPS calculation
            # =================================================

            current_time = time.time()

            elapsed = (
                current_time
                - previous_time
            )

            if elapsed > 0:

                current_fps = (
                    1.0 / elapsed
                )

                if fps == 0:

                    fps = current_fps

                else:

                    fps = (
                        0.9 * fps
                        + 0.1 * current_fps
                    )

            previous_time = (
                current_time
            )

            # =================================================
            # Header information
            # =================================================

            cv2.putText(
                frame,
                f"Faces: {len(faces)}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                frame,
                f"Recognized: "
                f"{recognized_count}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                frame,
                f"Threshold: "
                f"{THRESHOLD:.3f}",
                (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                frame,
                f"FPS: {fps:.1f}",
                (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                frame,
                f"Recognition: "
                f"1/{RECOGNITION_INTERVAL} frames",
                (10, 150),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

            cv2.putText(
                frame,
                "Q = Quit | +/- = Threshold",
                (10, 455),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

            # =================================================
            # Display camera
            # =================================================

            cv2.imshow(
                "Live Face Recognition",
                frame,
            )

            # =================================================
            # Keyboard input
            # =================================================

            key = (
                cv2.waitKey(1)
                & 0xFF
            )

            # ------------------------------------------------
            # Q = quit
            # ------------------------------------------------

            if key == ord("q"):

                break

            # ------------------------------------------------
            # + or =
            #
            # Increase distance threshold
            # ------------------------------------------------

            if key in (
                ord("+"),
                ord("="),
            ):

                THRESHOLD = min(
                    1.0,
                    THRESHOLD + 0.01,
                )

                matcher.threshold = (
                    THRESHOLD
                )

                print(
                    f"Distance threshold: "
                    f"{THRESHOLD:.3f}"
                )

            # ------------------------------------------------
            # - or _
            #
            # Decrease distance threshold
            # ------------------------------------------------

            if key in (
                ord("-"),
                ord("_"),
            ):

                THRESHOLD = max(
                    0.0,
                    THRESHOLD - 0.01,
                )

                matcher.threshold = (
                    THRESHOLD
                )

                print(
                    f"Distance threshold: "
                    f"{THRESHOLD:.3f}"
                )

    finally:

        # =====================================================
        # Cleanup
        # =====================================================

        cap.release()

        cv2.destroyAllWindows()

        detector.close()

        print()

        print(
            "Recognition stopped."
        )


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":

    main()