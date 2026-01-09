"""Camera port scanner - finds working camera index with DirectShow backend."""
import cv2

def list_camera_ports():
    """
    Test the first 5 indexes to see which ones work.
    """
    print("Scanning for cameras... (this may take a few seconds)")
    available_ports = []

    for index in range(5):
        # We use CAP_DSHOW to avoid the MSMF error you saw earlier
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                # Check if frame is not black (using ASCII markers)
                if frame is not None and frame.mean() > 5:
                    print(f"[OK] Camera found at Index {index} (with valid frames)")
                    available_ports.append(index)
                else:
                    print(f"[!] Index {index} opened, but frames are BLACK.")
            else:
                print(f"[!] Index {index} opened, but failed to grab a frame.")
            cap.release()
        else:
            print(f"[X] Index {index} is not available.")
    
    return available_ports

# Run the scanner
working_ports = list_camera_ports()

print("\n--------------------------------")
if working_ports:
    print(f"Success! Use index {working_ports[0]} in your code.")
else:
    print("No working cameras found. Please check physical connection or privacy settings.")
