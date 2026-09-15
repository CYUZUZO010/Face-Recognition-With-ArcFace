import cv2
import time


print("=" * 60)
print("SEARCHING FOR CAMERAS")
print("=" * 60)


for index in range(6):

    print()
    print(f"Testing camera index {index}...")

    cap = cv2.VideoCapture(
        index,
        cv2.CAP_DSHOW
    )

    if not cap.isOpened():

        print("  Could not open")

        cap.release()

        continue

    print("  Opened!")

    cap.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        640
    )

    cap.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        480
    )

    # Give camera time to initialize
    time.sleep(1)

    good_frame = None

    start = time.time()

    while time.time() - start < 2:

        ok, frame = cap.read()

        if ok:
            good_frame = frame

    if good_frame is None:

        print("  No frame received")

        cap.release()

        continue

    mean = good_frame.mean()

    print(
        f"  Frame received"
    )

    print(
        f"  Brightness: {mean:.2f}"
    )

    cv2.putText(
        good_frame,
        f"CAMERA INDEX {index}",
        (20, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.putText(
        good_frame,
        f"Brightness: {mean:.2f}",
        (20, 85),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.imshow(
        f"Camera {index}",
        good_frame
    )

    print(
        f"  Camera {index} is displayed."
    )

    print(
        "  Look at the window."
    )

    print(
        "  Press ANY KEY in the camera window to continue."
    )

    cv2.waitKey(0)

    cv2.destroyAllWindows()

    cap.release()


print()
print("=" * 60)
print("CAMERA SEARCH COMPLETE")
print("=" * 60)