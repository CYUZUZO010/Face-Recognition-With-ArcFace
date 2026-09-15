import cv2

CAMERA_INDEX = 2  # CHANGE THIS to your USB camera index

cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)

if not cap.isOpened():
    print(f"Could not open camera {CAMERA_INDEX}")
    exit()

# Faster resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Try automatic exposure
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)

# Increase brightness
cap.set(cv2.CAP_PROP_BRIGHTNESS, 60)

# Slightly increase contrast
cap.set(cv2.CAP_PROP_CONTRAST, 20)

print("Camera opened")
print("Width:", cap.get(cv2.CAP_PROP_FRAME_WIDTH))
print("Height:", cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print("Exposure:", cap.get(cv2.CAP_PROP_EXPOSURE))
print("Brightness:", cap.get(cv2.CAP_PROP_BRIGHTNESS))
print("Contrast:", cap.get(cv2.CAP_PROP_CONTRAST))

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to read frame")
        break

    cv2.imshow("USB Camera - Q to quit", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()