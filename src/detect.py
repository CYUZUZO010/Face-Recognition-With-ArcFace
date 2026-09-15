# src/detect.py

import cv2


CAMERA_INDEX = 1
FRAME_WIDTH = 640
FRAME_HEIGHT = 480


def main():
    cascade_path = (
        cv2.data.haarcascades
        + "haarcascade_frontalface_default.xml"
    )

    face = cv2.CascadeClassifier(cascade_path)

    if face.empty():
        raise RuntimeError(
            f"Failed to load cascade: {cascade_path}"
        )

    # Camera 1 = your USB camera
    cap = cv2.VideoCapture(
        CAMERA_INDEX,
        cv2.CAP_DSHOW,
    )

    if not cap.isOpened():
        raise RuntimeError(
            "USB camera could not be opened."
        )

    # Lower resolution = less lag
    cap.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        FRAME_WIDTH,
    )

    cap.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        FRAME_HEIGHT,
    )

    print("USB camera face detection running.")
    print("Press 'q' to quit.")

    while True:
        ok, frame = cap.read()

        if not ok:
            print("Failed to read camera frame.")
            break

        # Mirror the camera like a normal webcam
        frame = cv2.flip(frame, 1)

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

        # Draw EVERY detected face
        for i, (x, y, w, h) in enumerate(faces, start=1):

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
            "Press Q to quit",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

        cv2.imshow(
            "Face Detection",
            frame,
        )

        if (cv2.waitKey(1) & 0xFF) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()