<div align="center">

# 🛡️ Vigilance AI
### Intelligent Driver Drowsiness Detection System

[![Live Demo](https://img.shields.io/badge/🚀_TRY_LIVE_DEMO-4F46E5?style=for-the-badge&logoColor=white)](https://ahmedtarek-mel.github.io/Vigilance-AI/web)
[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![TensorFlow.js](https://img.shields.io/badge/TensorFlow.js-4.10+-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/js)
[![License](https://img.shields.io/badge/License-Copyright-red?style=flat-square)](LICENSE)

**Prevent accidents before they happen with AI-powered driver state analysis.**

[Launch Web App](https://ahmedtarek-mel.github.io/Vigilance-AI/web) 

</div>



---

## 🎬 Demo

<div align="center">
  <!-- GitHub supports video playback for mp4 -->
  <video src="demo/Demo.mp4" width="100%" controls></video>
  <p><em>Real-time detection of eye closure (Blinks) and mouth opening (Yawns)</em></p>
</div>

---

##  Features

###  Precision Monitoring
- **Eye Tracking** — EAR (Eye Aspect Ratio) algorithm for micro-sleep detection
- **Yawn Detection** — MAR (Mouth Aspect Ratio) analysis for signs of fatigue
- **Head Pose** — (Python Version) 3D orientation tracking for distraction

###  Interactive Dashboard
- **Real-time Metrics** — Live EAR/MAR values and Drowsiness Score
- **Dynamic Charts** — Visual history of driver state
- **Visual & Audio Alerts** — Multisensory warnings when thresholds are breached

###  Dual Architecture
- **Web App (Edge AI)** — Runs locally in browser via TensorFlow.js (Privacy focused)
- **Desktop App (Python)** — Robust dlib/OpenCV backend for high-performance setups

###  Configurable System
- **Adjustable Thresholds** — Customize sensitivity for different drivers
- **Dark/Light Themes** — Optimized for day/night driving conditions
- **Anti-Tamper** — Protected against unauthorized code inspection

---

##  Quick Start

### 🌐 Web Version (Recommended)
No installation needed! Runs directly in your browser.

1.  **[Click Here to Open](https://ahmedtarek-mel.github.io/Driver-Drowsiness-Detection/web)**
2.  Allow camera access.
3.  Start driving!

### 🐍 Python Version

```bash
# Clone the repository
git clone https://github.com/ahmedtarek-mel/Driver-Drowsiness-Detection.git
cd Driver-Drowsiness-Detection

# Install dependencies
pip install -r requirements.txt

# Run the application
python main.py
```

---

##  How It Works

The system uses a **facial landmark geometry** approach to quantify fatigue:

```
EAR (Eye Aspect Ratio) = (|p2-p6| + |p3-p5|) / (2 × |p1-p4|)
```

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Face Mesh** | MediaPipe / Dlib | Detect 468 facial landmarks in real-time |
| **Eye Analysis** | EAR Algorithm | Calculate eye openness to detect drowsiness |
| **Mouth Analysis** | MAR Algorithm | measure mouth aspect ratio to detect yawns |

<details>
<summary><strong>🔬 Technical Deep Dive</strong></summary>

### 1. Face Detection
I utilize **MediaPipe Face Mesh** (Web) and **dlib HOG** (Python) to locate facial keypoints with high sub-pixel accuracy, ensuring robust performance even in low light.

### 2. State Estimation
Instead of black-box ML, I use geometric ratios (EAR/MAR). This makes the system **explainable** and **computationally efficient**, allowing it to run smoothly on mobile devices or Raspberry Pi.

### 3. Smart Alerting
A "Perclos-like" temporal analysis filters out normal blinks. Alerts are only triggered when the drowsiness score (a weighted moving average of closure duration) breaches critical levels.

</details>

---

##  Tech Stack

| Layer | Technology |
|-------|------------|
| **Web Frontend** | HTML5, CSS3, TensorFlow.js, Chart.js |
| **Python Backend** | OpenCV, dlib, PyGame, NumPy |
| **Analysis** | Facial Landmark Geometry (EAR/MAR) |
| **Deployment** | GitHub Pages (Static Web App) |

---

##  Project Structure

```
Driver-Drowsiness-Detection/
├── web/                   # 🌐 TensorFlow.js Application
│   ├── index.html
│   ├── style.css
│   └── app.js
├── src/                   # 🐍 Python Core Logic
│   ├── detection/         # Face & Eye algorithms
│   └── core/              # Video stream handling
├── config.yaml            # System Configuration
├── main.py                # Python Entry Point
└── requirements.txt       # Dependencies
```

---

## 👤 Author

<div align="center">

**Ahmed Tarek**

*Computer Vision & AI Engineer*

[![GitHub](https://img.shields.io/badge/GitHub-ahmedtarek--mel-181717?style=flat-square&logo=github)](https://github.com/ahmedtarek-mel)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Ahmed_Tarek-0A66C2?style=flat-square&logo=linkedin)](https://linkedin.com/in/ahmed-tarek-mel)
[![Email](https://img.shields.io/badge/Email-Contact_Me-EA4335?style=flat-square&logo=gmail)](mailto:your-email@example.com)

*"Turning pixel data into life-saving intelligence."*

</div>

---

## 📄 License

**Copyright © 2026 Ahmed Tarek. All Rights Reserved.**

This project is for demonstration purposes. Unauthorized copying or commercial use is strictly prohibited. See [LICENSE](LICENSE) for details.

---

<div align="center">

**⭐ Star this repo if you find it interesting!**

Made with ❤️ by [Ahmed Tarek](https://github.com/ahmedtarek-mel)

</div>
