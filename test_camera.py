"""Simple camera test to diagnose OpenCV issues."""
import cv2
import time

print("Testing camera access...")

# Try to open camera with DSHOW backend on Windows
import platform
if platform.system() == 'Windows':
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
else:
    cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not open camera!")
    print("Possible causes:")
    print("  1. Camera is in use by another application")
    print("  2. No camera connected")
    print("  3. Camera permissions not granted")
    exit(1)

print("Camera opened successfully!")

# Set camera properties
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Warmup: read several frames to let camera auto-adjust exposure/brightness
print("Warming up camera (please wait)...")
for i in range(30):
    ret, frame = cap.read()
    time.sleep(0.05)

# Read a frame
ret, frame = cap.read()
if not ret or frame is None:
    print("ERROR: Could not read frame from camera!")
    cap.release()
    exit(1)

print(f"Frame captured: {frame.shape[1]}x{frame.shape[0]}")

# Create and show window
window_name = "Camera Test - Press Q to quit"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.imshow(window_name, frame)

print("Window created! If you can see this message but no window,")
print("try checking your taskbar or Alt+Tab to find the window.")
print("Press 'q' to quit...")

# Main loop
frame_count = 0
start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame!")
        break
    
    frame_count += 1
    
    # Add text overlay
    fps = frame_count / (time.time() - start_time) if frame_count > 0 else 0
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, "Press Q to quit", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    cv2.imshow(window_name, frame)
    
    # Print status every 100 frames
    if frame_count % 100 == 0:
        print(f"Frames: {frame_count}, FPS: {fps:.1f}")
    
    # Check for quit
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:  # Q or ESC
        print("Quitting...")
        break

cap.release()
cv2.destroyAllWindows()
print("Done!")
