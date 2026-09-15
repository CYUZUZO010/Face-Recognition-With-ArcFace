import cv2
import time

CAMERA_INDEX = 2

print(f"Opening camera index {CAMERA_INDEX}...")

cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("ERROR: Camera could not be opened.")
    exit()

print("Camera opened successfully.")
print("Move your hand or face in front of the camera.")
print("Press Q to quit.")

# Give the camera a moment to initialize
time.sleep(2)

while True:
    ret, frame = cap.read()

    if not ret:
        print("Could not read frame.")
        continue

    cv2.putText(
        frame,
        "USB CAMERA - INDEX 3",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("USB Camera Test", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()