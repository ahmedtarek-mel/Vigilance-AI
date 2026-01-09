import base64
import os

def file_to_b64(path):
    if not os.path.exists(path):
        print(f"Error: {path} not found")
        return ""
    with open(path, "rb") as f:
        return "data:video/mp4;base64," + base64.b64encode(f.read()).decode('utf-8')

# Paths are relative to CWD
t1 = file_to_b64("web/assets/test1.mp4")
t2 = file_to_b64("web/assets/test2.mp4")

if t1 and t2:
    js_content = f"window.PRESETS = {{ test1: '{t1}', test2: '{t2}' }};"
    with open("web/presets.js", "w") as f:
        f.write(js_content)
    print("Success: web/presets.js created")
else:
    print("Failed to encode videos")
