# src/enroll.py

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import json
import re
import time

import cv2
import numpy as np

from .haar_5pt import Haar5ptDetector, align_face_5pt
from .embed import ArcFaceEmbedderONNX, EmbeddingResult


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass
class EnrollConfig:
    out_db_npz: Path = Path("data/db/face_db.npz")
    out_db_json: Path = Path("data/db/face_db.json")

    crops_dir: Path = Path("data/enroll")

    save_crops: bool = True

    samples_needed: int = 15

    auto_capture_every_s: float = 0.5


# ============================================================
# HELPERS
# ============================================================

def sanitize_name(name: str) -> str:

    name = name.strip()

    name = re.sub(
        r'[<>:"/\\|?*]',
        "_",
        name,
    )

    name = re.sub(
        r"\s+",
        "_",
        name,
    )

    return name


def get_embedding_array(result) -> np.ndarray:
    """
    ArcFaceEmbedderONNX.embed() returns EmbeddingResult.

    We need only result.embedding.
    """

    if isinstance(result, EmbeddingResult):

        embedding = result.embedding

    elif hasattr(result, "embedding"):

        embedding = result.embedding

    else:

        embedding = result

    embedding = np.asarray(
        embedding,
        dtype=np.float32,
    ).reshape(-1)

    norm = np.linalg.norm(embedding)

    if norm > 1e-12:

        embedding = embedding / norm

    return embedding.astype(
        np.float32
    )


def mean_embedding(
    embeddings: List[np.ndarray],
) -> np.ndarray:

    if not embeddings:

        raise ValueError(
            "No embeddings were provided."
        )

    matrix = np.vstack(
        [
            get_embedding_array(e)
            for e in embeddings
        ]
    ).astype(np.float32)

    mean = np.mean(
        matrix,
        axis=0,
    ).astype(np.float32)

    norm = np.linalg.norm(mean)

    if norm > 1e-12:

        mean = mean / norm

    return mean.astype(
        np.float32
    )


# ============================================================
# FACE BOX HELPER
# ============================================================

def get_face_box(face, frame_shape):
    """
    Build a bounding box from the five facial landmarks.

    This works with the current FaceKpsBox structure
    without requiring x1/y1/x2/y2 attributes.
    """

    h, w = frame_shape[:2]

    kps = np.asarray(
        face.kps,
        dtype=np.float32,
    )

    x_min = int(
        np.min(kps[:, 0])
    )

    y_min = int(
        np.min(kps[:, 1])
    )

    x_max = int(
        np.max(kps[:, 0])
    )

    y_max = int(
        np.max(kps[:, 1])
    )

    # Add padding around the five landmarks
    face_width = max(
        1,
        x_max - x_min,
    )

    face_height = max(
        1,
        y_max - y_min,
    )

    pad_x = int(
        face_width * 0.8
    )

    pad_y = int(
        face_height * 0.9
    )

    x1 = max(
        0,
        x_min - pad_x,
    )

    y1 = max(
        0,
        y_min - pad_y,
    )

    x2 = min(
        w - 1,
        x_max + pad_x,
    )

    y2 = min(
        h - 1,
        y_max + pad_y,
    )

    return x1, y1, x2, y2


# ============================================================
# DATABASE
# ============================================================

def load_database(
    npz_path: Path,
):

    if not npz_path.exists():

        return (
            [],
            np.empty(
                (0, 512),
                dtype=np.float32,
            ),
        )

    try:

        data = np.load(
            npz_path,
            allow_pickle=True,
        )

        names = data[
            "names"
        ].tolist()

        embeddings = data[
            "embeddings"
        ].astype(
            np.float32
        )

        return names, embeddings

    except Exception as e:

        print(
            f"[WARNING] Could not load database: {e}"
        )

        return (
            [],
            np.empty(
                (0, 512),
                dtype=np.float32,
            ),
        )


def save_database(
    npz_path: Path,
    json_path: Path,
    names: List[str],
    embeddings: np.ndarray,
):

    npz_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    json_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    embeddings = np.asarray(
        embeddings,
        dtype=np.float32,
    )

    np.savez(
        npz_path,
        names=np.asarray(
            names,
            dtype=object,
        ),
        embeddings=embeddings,
    )

    json_data = {
        "names": names,
        "embeddings": embeddings.tolist(),
    }

    with open(
        json_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            json_data,
            f,
            indent=2,
        )

    print()
    print("=" * 65)
    print("DATABASE SAVED")
    print("=" * 65)

    print(
        f"NPZ : {npz_path}"
    )

    print(
        f"JSON: {json_path}"
    )

    print(
        f"People: {len(names)}"
    )

    print(
        f"Embedding shape: {embeddings.shape}"
    )

    print("=" * 65)


# ============================================================
# MAIN
# ============================================================

def main():

    cfg = EnrollConfig()

    print("=" * 65)
    print("LIVE FACE ENROLLMENT")
    print("=" * 65)

    name_input = input(
        "Enter the person's name: "
    )

    name = sanitize_name(
        name_input
    )

    if not name:

        print(
            "Invalid name."
        )

        return

    print()
    print(
        f"Enrolling: {name}"
    )

    # --------------------------------------------------------
    # DATABASE
    # --------------------------------------------------------

    existing_names, existing_embeddings = (
        load_database(
            cfg.out_db_npz
        )
    )

    print(
        f"Existing identities: "
        f"{len(existing_names)}"
    )

    # --------------------------------------------------------
    # DETECTOR
    # --------------------------------------------------------

    detector = Haar5ptDetector(
        min_size=(70, 70),
        smooth_alpha=0.80,
        debug=False,
    )

    # --------------------------------------------------------
    # ARC FACE
    # --------------------------------------------------------

    embedder = ArcFaceEmbedderONNX(
        model_path="models/embedder_arcface.onnx",
        debug=False,
    )

    print(
        "ArcFace embedding dimension: 512"
    )

    # --------------------------------------------------------
    # CAMERA
    # --------------------------------------------------------

    camera_index = 2

    print()
    print(
        f"Opening external USB camera: index {camera_index}..."
    )

    cap = cv2.VideoCapture(
        camera_index,
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

        detector.close()

        raise RuntimeError(
            f"Could not open camera {camera_index}."
        )

    print(
        "Camera opened successfully."
    )

    # --------------------------------------------------------
    # STORAGE
    # --------------------------------------------------------

    captured_embeddings: List[
        np.ndarray
    ] = []

    person_crop_dir = (
        cfg.crops_dir / name
    )

    if cfg.save_crops:

        person_crop_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    flash_until = 0

    message = ""

    message_until = 0

    # --------------------------------------------------------
    # CONTROLS
    # --------------------------------------------------------

    print()
    print("=" * 65)
    print("ENROLLMENT CONTROLS")
    print("=" * 65)
    print()
    print("SPACE = capture one sample")
    print("A     = automatic capture")
    print("R     = reset captured samples")
    print("S     = SAVE enrollment")
    print("Q     = quit")
    print()

    try:

        while True:

            ok, frame = cap.read()

            if not ok:

                print(
                    "[ERROR] Could not read camera frame."
                )

                break

            vis = frame.copy()

            # ------------------------------------------------
            # DETECTION
            # ------------------------------------------------

            faces = detector.detect(
                frame,
                max_faces=1,
            )

            current_face = None

            if faces:

                current_face = faces[0]

                # Get bbox from landmarks
                x1, y1, x2, y2 = get_face_box(
                    current_face,
                    frame.shape,
                )

                # Green bounding box
                cv2.rectangle(
                    vis,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2,
                )

                # Five red landmarks
                for x, y in current_face.kps.astype(
                    int
                ):

                    cv2.circle(
                        vis,
                        (
                            int(x),
                            int(y),
                        ),
                        5,
                        (0, 0, 255),
                        -1,
                    )

                cv2.putText(
                    vis,
                    "FACE DETECTED",
                    (
                        10,
                        30,
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )

            else:

                cv2.putText(
                    vis,
                    "NO FACE DETECTED",
                    (
                        10,
                        30,
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )

            # ------------------------------------------------
            # INFO
            # ------------------------------------------------

            count = len(
                captured_embeddings
            )

            cv2.putText(
                vis,
                f"Person: {name}",
                (
                    10,
                    65,
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
            )

            cv2.putText(
                vis,
                f"Samples: {count}/{cfg.samples_needed}",
                (
                    10,
                    95,
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
            )

            cv2.putText(
                vis,
                "SPACE Capture | S Save | Q Quit",
                (
                    10,
                    vis.shape[0] - 20,
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )

            # ------------------------------------------------
            # FLASH
            # ------------------------------------------------

            if time.time() < flash_until:

                cv2.rectangle(
                    vis,
                    (0, 0),
                    (
                        vis.shape[1] - 1,
                        vis.shape[0] - 1,
                    ),
                    (255, 255, 255),
                    8,
                )

                cv2.putText(
                    vis,
                    "SAMPLE CAPTURED!",
                    (
                        150,
                        230,
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 0),
                    3,
                )

            # ------------------------------------------------
            # MESSAGE
            # ------------------------------------------------

            if time.time() < message_until:

                cv2.putText(
                    vis,
                    message,
                    (
                        10,
                        130,
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2,
                )

            # ------------------------------------------------
            # DISPLAY
            # ------------------------------------------------

            cv2.imshow(
                "Face Enrollment",
                vis,
            )

            key = (
                cv2.waitKey(1)
                & 0xFF
            )

            # ------------------------------------------------
            # QUIT
            # ------------------------------------------------

            if key == ord("q"):

                print()
                print(
                    "Enrollment cancelled."
                )

                break

            # ------------------------------------------------
            # RESET
            # ------------------------------------------------

            elif key == ord("r"):

                captured_embeddings.clear()

                print(
                    "Captured samples reset."
                )

                message = (
                    "Samples reset"
                )

                message_until = (
                    time.time() + 2
                )

            # ------------------------------------------------
            # CAPTURE WITH SPACE
            # ------------------------------------------------

            elif key == 32:

                if current_face is None:

                    message = (
                        "No face detected"
                    )

                    message_until = (
                        time.time() + 2
                    )

                    continue

                if len(
                    captured_embeddings
                ) >= cfg.samples_needed:

                    message = (
                        "15 samples captured - press S to save"
                    )

                    message_until = (
                        time.time() + 3
                    )

                    continue

                # --------------------------------------------
                # ALIGN
                # --------------------------------------------

                aligned = align_face_5pt(
                    frame,
                    current_face.kps,
                    image_size = 112,
                )

                # --------------------------------------------
                # EMBED
                # --------------------------------------------

                result = embedder.embed(
                    aligned
                )

                embedding = get_embedding_array(
                    result
                )

                # --------------------------------------------
                # VERIFY
                # --------------------------------------------

                if embedding.size != 512:

                    print(
                        f"[ERROR] Expected 512 values, "
                        f"got {embedding.size}"
                    )

                    continue

                # --------------------------------------------
                # STORE
                # --------------------------------------------

                captured_embeddings.append(
                    embedding
                )

                # --------------------------------------------
                # SAVE CROP
                # --------------------------------------------

                if cfg.save_crops:

                    timestamp = (
                        time.strftime(
                            "%Y%m%d_%H%M%S"
                        )
                        + "_"
                        + str(
                            int(
                                time.time() * 1000
                            ) % 1000
                        )
                    )

                    crop_path = (
                        person_crop_dir
                        / f"{timestamp}.jpg"
                    )

                    cv2.imwrite(
                        str(crop_path),
                        aligned,
                    )

                count = len(
                    captured_embeddings
                )

                print(
                    f"Captured sample "
                    f"{count}/{cfg.samples_needed} "
                    f"| embedding shape: "
                    f"{embedding.shape}"
                )

                message = (
                    f"Sample {count}/{cfg.samples_needed} captured!"
                )

                message_until = (
                    time.time() + 2
                )

                flash_until = (
                    time.time() + 0.15
                )

            # ------------------------------------------------
            # AUTOMATIC CAPTURE
            # ------------------------------------------------

            elif key == ord("a"):

                print()
                print(
                    "Automatic capture started..."
                )

                while len(
                    captured_embeddings
                ) < cfg.samples_needed:

                    ok2, frame2 = cap.read()

                    if not ok2:

                        break

                    faces2 = detector.detect(
                        frame2,
                        max_faces=1,
                    )

                    if not faces2:

                        cv2.imshow(
                            "Face Enrollment",
                            frame2,
                        )

                        if (
                            cv2.waitKey(1)
                            & 0xFF
                        ) == ord("q"):

                            break

                        continue

                    face2 = faces2[0]

                    aligned2 = align_face_5pt(
                        frame2,
                        face2.kps,
                        image_size=112,
                    )

                    result2 = (
                        embedder.embed(
                            aligned2
                        )
                    )

                    emb2 = (
                        get_embedding_array(
                            result2
                        )
                    )

                    if emb2.size != 512:

                        continue

                    captured_embeddings.append(
                        emb2
                    )

                    if cfg.save_crops:

                        timestamp = (
                            time.strftime(
                                "%Y%m%d_%H%M%S"
                            )
                            + "_"
                            + str(
                                int(
                                    time.time()
                                    * 1000
                                )
                                % 1000
                            )
                        )

                        crop_path = (
                            person_crop_dir
                            / f"{timestamp}.jpg"
                        )

                        cv2.imwrite(
                            str(
                                crop_path
                            ),
                            aligned2,
                        )

                    print(
                        f"Auto captured "
                        f"{len(captured_embeddings)}/"
                        f"{cfg.samples_needed}"
                    )

                    time.sleep(
                        cfg.auto_capture_every_s
                    )

                print(
                    "Automatic capture finished."
                )

            # ------------------------------------------------
            # SAVE
            # ------------------------------------------------

            elif key == ord("s"):

                count = len(
                    captured_embeddings
                )

                if count < 3:

                    print()
                    print(
                        "Not enough samples."
                    )
                    print(
                        "Capture at least 3 samples first."
                    )

                    message = (
                        "Need at least 3 samples"
                    )

                    message_until = (
                        time.time() + 3
                    )

                    continue

                print()
                print(
                    "=" * 65
                )
                print(
                    "CREATING MEAN EMBEDDING..."
                )
                print(
                    "=" * 65
                )

                template = mean_embedding(
                    captured_embeddings
                )

                print(
                    f"Mean embedding shape: "
                    f"{template.shape}"
                )

                print(
                    f"Mean embedding norm: "
                    f"{np.linalg.norm(template):.4f}"
                )

                # --------------------------------------------
                # ADD OR UPDATE
                # --------------------------------------------

                if name in existing_names:

                    index = existing_names.index(
                        name
                    )

                    existing_embeddings[
                        index
                    ] = template

                    print(
                        f"Updated identity: {name}"
                    )

                else:

                    existing_names.append(
                        name
                    )

                    if existing_embeddings.size == 0:

                        existing_embeddings = (
                            template.reshape(
                                1,
                                -1,
                            )
                        )

                    else:

                        existing_embeddings = (
                            np.vstack(
                                [
                                    existing_embeddings,
                                    template,
                                ]
                            )
                        )

                    print(
                        f"Added identity: {name}"
                    )

                # --------------------------------------------
                # SAVE
                # --------------------------------------------

                save_database(
                    cfg.out_db_npz,
                    cfg.out_db_json,
                    existing_names,
                    existing_embeddings,
                )

                print()
                print(
                    "=" * 65
                )
                print(
                    f"SUCCESS! {name} has been enrolled."
                )
                print(
                    "=" * 65
                )

                print()
                print(
                    "You can now test live recognition."
                )

                cv2.waitKey(1500)

                break

    finally:

        cap.release()

        cv2.destroyAllWindows()

        detector.close()


if __name__ == "__main__":

    main()