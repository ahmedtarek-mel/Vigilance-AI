"""Comprehensive camera diagnostic - tests multiple backends and indices."""
import cv2
import time
import platform

print("="*60)
print("CAMERA DIAGNOSTIC TOOL")
print("="*60)

# Test different backends
backends = [
    ("DSHOW (DirectShow)", cv2.CAP_DSHOW),
    ("MSMF (Media Foundation)", cv2.CAP_MSMF),
    ("ANY (Auto-detect)", cv2.CAP_ANY),
]

# Test camera indices 0, 1, 2
camera_indices = [0, 1, 2]

working_config = None

for cam_idx in camera_indices:
    for backend_name, backend_id in backends:
        print(f"\nTrying Camera {cam_idx} with {backend_name}...")
        
        try:
            cap = cv2.VideoCapture(cam_idx, backend_id)
            
            if not cap.isOpened():
                print(f"  [X] Could not open")
                cap.release()
                continue
            
            # Set properties
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            # Try to read frames
            success_count = 0
            for i in range(10):
                ret, frame = cap.read()
                if ret and frame is not None:
                    # Check if frame is not all black
                    if frame.mean() > 5:  # Average pixel value > 5 means not black
                        success_count += 1
                time.sleep(0.1)
            
            cap.release()
            
            if success_count > 0:
                print(f"  [OK] SUCCESS! Got {success_count}/10 non-black frames")
                working_config = (cam_idx, backend_id, backend_name)
                break
            else:
                print(f"  [!] Opened but frames are black")
                
        except Exception as e:
            print(f"  [X] Error: {e}")

    if working_config:
        break

print("\n" + "="*60)
if working_config:
    cam_idx, backend_id, backend_name = working_config
    print(f"[OK] WORKING CONFIG FOUND!")
    print(f"   Camera Index: {cam_idx}")
    print(f"   Backend: {backend_name}")
    print("="*60)
    
    # Show working camera
    print("\nOpening camera window (press Q to quit)...")
    cap = cv2.VideoCapture(cam_idx, backend_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        cv2.putText(frame, f"Camera {cam_idx} - {backend_name}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, "Press Q to quit", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow("Working Camera", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
else:
    print("[X] NO WORKING CAMERA CONFIGURATION FOUND")
    print("="*60)
    print("\nPossible solutions:")
    print("1. Restart your computer to reset camera driver")
    print("2. Check Windows Settings > Privacy > Camera")
    print("3. Update camera driver in Device Manager")
    print("4. Try the WEB VERSION instead (uses browser camera)")

print("\nDone!")
