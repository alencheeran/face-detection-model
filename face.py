# face_detection.py
import cv2
import time

# 1. Load pre-trained Haar Cascade classifier for frontal faces
cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(cascade_path)

# 2. Start the webcam (0 = default camera)
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Could not open webcam. Check camera permissions or index.")

# Optional: set a smaller frame size for speed (uncomment if needed)
 cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
 cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# For FPS calculation
prev_time = 0
frame_count = 0

# Main loop
while True:
    ret, frame = cap.read()           # A. Capture a frame from webcam
    if not ret:
        print("Failed to grab frame")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)   # B. Convert to grayscale (faster detection)

    # C. Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    # D. Draw rectangles and labels
    for i, (x, y, w, h) in enumerate(faces):
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, f"Face {i+1}", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # E. Calculate and show FPS
    frame_count += 1
    cur_time = time.time()
    elapsed = cur_time - prev_time
    if elapsed > 0.5:  # update twice a second
        fps = frame_count / elapsed
        prev_time = cur_time
        frame_count = 0
    # display FPS on the frame (top-left corner)
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # F. Show the result
    cv2.imshow("Face Detection - Press 'q' to quit, 's' to save", frame)

    # G. Keyboard controls
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        # Save a snapshot when user presses 's'
        timestamp = int(time.time())
        filename = f"face_snapshot_{timestamp}.png"
        cv2.imwrite(filename, frame)
        print(f"Saved {filename}")

# Cleanup
cap.release()
cv2.destroyAllWindows()
