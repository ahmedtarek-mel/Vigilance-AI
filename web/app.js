/**
 * Driver Drowsiness Detection System
 * Browser-based real-time detection using TensorFlow.js
 * 
 * Features:
 * - Eye Aspect Ratio (EAR) calculation
 * - Mouth Aspect Ratio (MAR) for yawn detection
 * - Real-time drowsiness scoring
 * - Audio/visual alerts
 */

// ============================================
// Configuration
// ============================================
const CONFIG = {
    EAR_THRESHOLD: 0.25,
    MAR_THRESHOLD: 0.75,
    DROWSY_TIME_MS: 2000,
    FPS_SAMPLE_SIZE: 30,
    CHART_MAX_POINTS: 50,
};

// ============================================
// State
// ============================================
const state = {
    isRunning: false,
    detector: null,
    video: null,
    canvas: null,
    ctx: null,

    // Metrics
    currentEAR: 0,
    currentMAR: 0,
    drowsinessScore: 0,
    blinkCount: 0,
    yawnCount: 0,
    alertCount: 0,

    // Tracking
    eyeClosedStartTime: null,
    isEyeClosed: false,
    mouthOpenStartTime: null,
    isYawning: false,
    wasYawning: false,
    wasBlinking: false,

    // Alerts
    isAlerting: false,
    soundEnabled: true,

    // Performance
    frameCount: 0,
    lastFpsUpdate: 0,
    fps: 0,
    fpsHistory: [],

    // Session
    sessionStartTime: null,
    sessionTimerInterval: null,

    // Chart
    chart: null,
    earHistory: [],
    marHistory: [],

    // Theme
    isDarkTheme: true,
    isCameraMode: false,
};

// ============================================
// Face Landmark Indices (MediaPipe)
// ============================================
const LANDMARKS = {
    // Left eye (6 points for EAR)
    leftEye: [362, 385, 387, 263, 373, 380],
    // Right eye (6 points for EAR)  
    rightEye: [33, 160, 158, 133, 153, 144],
    // Lips for MAR
    upperLip: [13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 78, 191, 80, 81, 82],
    lowerLip: [14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191, 78, 95, 88, 178, 87],
    innerLipTop: [13, 312, 311, 310, 415, 308],
    innerLipBottom: [14, 87, 178, 88, 95, 78],
    // Mouth corners for width
    mouthLeft: 61,
    mouthRight: 291,
    mouthTop: 13,
    mouthBottom: 14,
};

// ============================================
// DOM Elements
// ============================================
let elements = {};

function initElements() {
    elements = {
        video: document.getElementById('video'),
        videoContainer: document.querySelector('.video-container'),
        overlay: document.getElementById('overlay'),
        startBtn: document.getElementById('start-btn'),
        stopBtn: document.getElementById('stop-btn'),
        soundBtn: document.getElementById('sound-btn'),
        uploadBtn: document.getElementById('upload-btn'),
        videoUpload: document.getElementById('video-upload'),
        // ... (other elements remain same, just patching initElements is safer if I replace the whole function or add properties carefully)
        statusBadge: document.getElementById('status-badge'),
        fpsBadge: document.getElementById('fps-badge'),
        themeToggle: document.getElementById('theme-toggle'),
        loadingScreen: document.getElementById('loading-screen'),
        loadingStatus: document.querySelector('.loading-status'),
        app: document.getElementById('app'),
        alertBanner: document.getElementById('alert-banner'),
        // Stats
        earValue: document.getElementById('ear-value'),
        marValue: document.getElementById('mar-value'),
        earBar: document.getElementById('ear-bar'),
        marBar: document.getElementById('mar-bar'),
        blinkCount: document.getElementById('blink-count'),
        yawnCount: document.getElementById('yawn-count'),
        drowsyCount: document.getElementById('drowsy-count'),
        alertCount: document.getElementById('alert-count'),
        scoreValue: document.getElementById('score-value'),
        scoreFill: document.querySelector('.score-fill'),
        scoreLabel: document.getElementById('score-label'),
        sessionTime: document.getElementById('session-time'),
        metricsChart: document.getElementById('metrics-chart'),
        // Settings
        earThreshold: document.getElementById('ear-threshold'),
        marThreshold: document.getElementById('mar-threshold'),
        drowsyTime: document.getElementById('drowsy-time'),
        // Value displays
        earThresholdDisplay: document.getElementById('ear-threshold-value'),
        marThresholdDisplay: document.getElementById('mar-threshold-value'),
        drowsyTimeDisplay: document.getElementById('drowsy-time-value'),

        // Debug
        debugText: document.getElementById('debug-text-overlay'),

        // Audio
        alertSound: document.getElementById('alert-sound'),
    };
}

function logDebug(message) {
    // Debug disabled for production
}

// ============================================
// Initialization
// ============================================
async function init() {
    initElements();
    setupEventListeners();
    setupChart();
    loadTheme();
    setupSecurity();

    try {
        updateLoadingStatus('Loading TensorFlow.js...');
        await tf.ready();

        updateLoadingStatus('Loading Face Detection Model...');
        // Initialize detector
        state.detector = await faceLandmarksDetection.createDetector(
            faceLandmarksDetection.SupportedModels.MediaPipeFaceMesh,
            {
                runtime: 'mediapipe', // Switch to mediapipe runtime for better stability
                refineLandmarks: true,
                maxFaces: 1,
                solutionPath: 'https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh',
            }
        );

        updateLoadingStatus('Ready!');
        await delay(500);

        showApp();
    } catch (error) {
        console.error('Initialization error:', error);
        logDebug(`Init Error: ${error.message}`);
        updateLoadingStatus('Error loading model. Please refresh.');
    }
}

function updateLoadingStatus(status) {
    if (elements.loadingStatus) {
        elements.loadingStatus.textContent = status;
    }
}

function showApp() {
    elements.loadingScreen.classList.add('hidden');
    elements.app.classList.remove('hidden');
}

function delay(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

// ============================================
// Event Listeners
// ============================================
function setupEventListeners() {
    elements.startBtn.addEventListener('click', startDetection);
    elements.stopBtn.addEventListener('click', stopDetection);
    elements.soundBtn.addEventListener('click', toggleSound);
    elements.themeToggle.addEventListener('click', toggleTheme);

    // Video Upload Handlers
    elements.uploadBtn.addEventListener('click', () => elements.videoUpload.click());
    elements.videoUpload.addEventListener('change', handleVideoUpload);

    // Preset Tests
    document.getElementById('test1-btn').addEventListener('click', () => loadPresetVideo('test1'));
    document.getElementById('test2-btn').addEventListener('click', () => loadPresetVideo('test2'));

    // Settings
    elements.earThreshold.addEventListener('input', (e) => {
        CONFIG.EAR_THRESHOLD = parseFloat(e.target.value);
        elements.earThresholdDisplay.textContent = e.target.value;
    });

    elements.marThreshold.addEventListener('input', (e) => {
        CONFIG.MAR_THRESHOLD = parseFloat(e.target.value);
        elements.marThresholdDisplay.textContent = e.target.value;
    });

    elements.drowsyTime.addEventListener('input', (e) => {
        CONFIG.DROWSY_TIME_MS = parseFloat(e.target.value) * 1000;
        elements.drowsyTimeDisplay.textContent = e.target.value;
    });
}

function loadPresetVideo(key) {
    try {
        if (!window.PRESETS || !PRESETS[key]) {
            alert('Preset video not found. Please run encode_presets.py first.');
            return;
        }

        // Show loading state
        elements.statusBadge.querySelector('span').textContent = 'Loading...';

        // Stop existing stream
        if (elements.video.srcObject) {
            const tracks = elements.video.srcObject.getTracks();
            tracks.forEach(track => track.stop());
            elements.video.srcObject = null;
        }

        // Use Base64 data URI directly
        elements.video.src = PRESETS[key];
        elements.video.loop = true;
        state.isCameraMode = false;

        elements.statusBadge.querySelector('span').textContent = 'Test Mode';

        elements.video.onloadedmetadata = () => {
            state.canvas.width = elements.video.videoWidth;
            state.canvas.height = elements.video.videoHeight;

            // Force container aspect ratio to prevent black bars/misalignment
            if (elements.videoContainer) {
                elements.videoContainer.style.aspectRatio = `${state.canvas.width}/${state.canvas.height}`;
            }

            // Show "Ready" status
            elements.statusBadge.querySelector('span').textContent = 'Ready';
            elements.statusBadge.classList.replace('badge-success', 'badge-secondary');
        };
    } catch (err) {
        console.error('Error loading preset:', err);
        elements.statusBadge.querySelector('span').textContent = 'Error';
    }
}

// Handle Video Upload
async function handleVideoUpload(e) {
    const file = e.target.files[0];
    if (!file) return;

    // Stop existing stream if any
    if (elements.video.srcObject) {
        const tracks = elements.video.srcObject.getTracks();
        tracks.forEach(track => track.stop());
        elements.video.srcObject = null;
    }

    // Set video source to file
    const url = URL.createObjectURL(file);
    elements.video.src = url;
    elements.video.loop = true;
    state.isCameraMode = false;

    elements.statusBadge.querySelector('span').textContent = 'Test Mode';

    elements.video.onloadedmetadata = () => {
        state.canvas.width = elements.video.videoWidth;
        state.canvas.height = elements.video.videoHeight;

        // Force container aspect ratio to prevent black bars/misalignment
        if (elements.videoContainer) {
            elements.videoContainer.style.aspectRatio = `${state.canvas.width}/${state.canvas.height}`;
        }

        // Show "Ready" status
        elements.statusBadge.querySelector('span').textContent = 'Ready';
        elements.statusBadge.classList.replace('badge-success', 'badge-secondary');
        // Do not auto-start. User must click Start.
    };
}

// ============================================
// Camera & Detection
// ============================================
async function startDetection() {
    try {
        if (!elements.video.src && !elements.video.srcObject) {
            // Only request camera if not playing a file
            const stream = await navigator.mediaDevices.getUserMedia({
                video: {
                    width: { ideal: 1280 },
                    height: { ideal: 720 },
                    facingMode: 'user',
                },
                audio: false,
            });
            elements.video.srcObject = stream;
        }

        // Wait for video metadata to load (handled in upload for file, but need check here too)
        if (elements.video.readyState < 2) {
            await new Promise((resolve) => {
                elements.video.onloadeddata = () => resolve();
            });
        }

        if (elements.video.paused) {
            await elements.video.play();
        }

        // Wait a bit for video to stabilize
        await delay(500);

        // Setup canvas with actual video dimensions
        state.canvas = elements.overlay;
        state.ctx = state.canvas.getContext('2d');

        // Use actual video dimensions
        const videoWidth = elements.video.videoWidth || 640;
        const videoHeight = elements.video.videoHeight || 480;

        // Force container aspect ratio to prevent black bars/misalignment
        if (elements.videoContainer) {
            elements.videoContainer.style.aspectRatio = `${videoWidth}/${videoHeight}`;
        }

        [
            {
                "StartLine": 345,
                "EndLine": 346,
                "TargetContent": "        console.log(`Video dimensions: ${videoWidth}x${videoHeight}`);\n        logDebug(`Started. Video: ${videoWidth}x${videoHeight}`);",
                "ReplacementContent": "",
                "AllowMultiple": false
            },
            {
                "StartLine": 415,
                "EndLine": 417,
                "TargetContent": "        if (state.frameCount % 30 === 0) {\n            logDebug(`Video: ${elements.video.videoWidth}x${elements.video.videoHeight}, Ready: ${elements.video.readyState}`);\n        }",
                "ReplacementContent": "",
                "AllowMultiple": false
            },
            {
                "StartLine": 423,
                "EndLine": 427,
                "TargetContent": "        if (state.frameCount % 30 === 0) {\n            const faceCount = faces ? faces.length : 0;\n            if (faceCount === 0) logDebug(\"No faces detected.\");\n            else logDebug(`Faces detected: ${faceCount}`);\n        }",
                "ReplacementContent": "",
                "AllowMultiple": false
            },
            {
                "StartLine": 439,
                "EndLine": 442,
                "TargetContent": "                // Debug first keypoint\n                if (state.frameCount % 100 === 0) {\n                    logDebug(`KP[0]: ${JSON.stringify(keypoints[0])}`);\n                }",
                "ReplacementContent": "",
                "AllowMultiple": false
            },
            {
                "StartLine": 447,
                "EndLine": 449,
                "TargetContent": "                if (state.frameCount % 100 === 0) {\n                    logDebug(`EAR: ${state.currentEAR}, MAR: ${state.currentMAR}`);\n                }",
                "ReplacementContent": "",
                "AllowMultiple": false
            },
            {
                "StartLine": 488,
                "EndLine": 490,
                "TargetContent": "    if (state.frameCount % 100 === 0) {\n        logDebug(`Processing ${keypoints.length} keypoints`);\n    }",
                "ReplacementContent": "",
                "AllowMultiple": false
            }
        ]
        elements.startBtn.classList.add('hidden');
        elements.stopBtn.classList.remove('hidden');
        elements.statusBadge.classList.remove('badge-danger');
        elements.statusBadge.classList.add('badge-success');
        elements.statusBadge.querySelector('span').textContent = 'Active';
        state.isCameraMode = true;

        // Start
        state.isRunning = true;
        state.sessionStartTime = Date.now();
        detectFrame();
        startSessionTimer();

    } catch (error) {
        console.error('Camera error:', error);
        alert('Could not access camera. Please allow camera permissions and make sure no other app is using the camera.');
    }
}

function stopDetection() {
    state.isRunning = false;

    // Stop camera
    const stream = elements.video.srcObject;
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
        elements.video.srcObject = null;
    }

    // Pause video file if playing
    if (elements.video.src && !elements.video.paused) {
        elements.video.pause();
        elements.video.currentTime = 0;
    }

    // Clear canvas
    if (state.ctx) {
        state.ctx.clearRect(0, 0, state.canvas.width, state.canvas.height);
    }

    // Update UI
    elements.startBtn.classList.remove('hidden');
    elements.stopBtn.classList.add('hidden');
    elements.statusBadge.classList.remove('badge-success');
    elements.statusBadge.classList.add('badge-danger');
    elements.statusBadge.querySelector('span').textContent = 'Stopped';

    // Stop alert
    stopAlert();

    // Stop session timer
    stopSessionTimer();
}

async function detectFrame() {
    if (!state.isRunning) return;

    // Validate detector and video
    if (!state.detector || !elements.video || elements.video.readyState < 2) {
        requestAnimationFrame(detectFrame);
        return;
    }

    const startTime = performance.now();

    try {
        // Debug logs (throttled)


        const faces = await state.detector.estimateFaces(elements.video, {
            flipHorizontal: false,
        });



        // Clear canvas
        if (state.ctx) {
            state.ctx.clearRect(0, 0, state.canvas.width, state.canvas.height);
        }

        if (faces && faces.length > 0) {
            const face = faces[0];
            const keypoints = face.keypoints;

            if (keypoints && keypoints.length > 0) {


                // Calculate metrics
                calculateMetrics(keypoints);



                // Draw visualization
                drawVisualization(keypoints);

                // Update drowsiness
                updateDrowsinessState();
            }
        } else {
            // No face detected - show message
            if (state.ctx) {
                state.ctx.fillStyle = 'rgba(255, 255, 255, 0.8)';
                state.ctx.font = '16px Inter, sans-serif';
                state.ctx.fillText('No face detected - look at camera', 10, 30);
            }
            state.currentEAR = 0;
            state.currentMAR = 0;
        }

        // Update display
        updateDisplay();
        updateChart();

        // Calculate FPS
        updateFPS(startTime);

    } catch (error) {
        console.error('Detection error:', error);
        logDebug(`Error: ${error.message}`);
    }

    // Next frame
    requestAnimationFrame(detectFrame);
}

// ============================================
// Metrics Calculation
// ============================================
function calculateMetrics(keypoints) {


    // Get landmark positions as array
    const points = keypoints.map(kp => [kp.x, kp.y]);

    // Calculate EAR
    const leftEAR = calculateEAR(points, LANDMARKS.leftEye);
    const rightEAR = calculateEAR(points, LANDMARKS.rightEye);
    state.currentEAR = (leftEAR + rightEAR) / 2;

    // Calculate MAR
    state.currentMAR = calculateMAR(points);



    // Detect blinks
    detectBlink();

    // Detect yawns
    detectYawn();
}

function calculateEAR(points, eyeIndices) {
    // EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)
    const p1 = points[eyeIndices[0]];
    const p2 = points[eyeIndices[1]];
    const p3 = points[eyeIndices[2]];
    const p4 = points[eyeIndices[3]];
    const p5 = points[eyeIndices[4]];
    const p6 = points[eyeIndices[5]];

    if (!p1 || !p2 || !p3 || !p4 || !p5 || !p6) return 0.3;

    const A = distance(p2, p6);
    const B = distance(p3, p5);
    const C = distance(p1, p4);

    if (C === 0) return 0.3;

    return (A + B) / (2 * C);
}

function calculateMAR(points) {
    // Use mouth landmarks for MAR
    const mouthTop = points[LANDMARKS.mouthTop];
    const mouthBottom = points[LANDMARKS.mouthBottom];
    const mouthLeft = points[LANDMARKS.mouthLeft];
    const mouthRight = points[LANDMARKS.mouthRight];

    if (!mouthTop || !mouthBottom || !mouthLeft || !mouthRight) return 0;

    const vertical = distance(mouthTop, mouthBottom);
    const horizontal = distance(mouthLeft, mouthRight);

    if (horizontal === 0) return 0;

    return vertical / horizontal;
}

function distance(p1, p2) {
    if (!p1 || !p2) return 0;
    return Math.sqrt(Math.pow(p2[0] - p1[0], 2) + Math.pow(p2[1] - p1[1], 2));
}

function detectBlink() {
    const isEyeCurrentlyClosed = state.currentEAR < CONFIG.EAR_THRESHOLD;

    if (!state.wasBlinking && isEyeCurrentlyClosed) {
        // Eyes just closed
        state.wasBlinking = true;
    } else if (state.wasBlinking && !isEyeCurrentlyClosed) {
        // Eyes just opened - count as blink
        state.blinkCount++;
        state.wasBlinking = false;
    }
}

function detectYawn() {
    const isCurrentlyYawning = state.currentMAR > CONFIG.MAR_THRESHOLD;

    if (!state.wasYawning && isCurrentlyYawning) {
        // Mouth just opened wide
        state.wasYawning = true;
        state.mouthOpenStartTime = Date.now();
    } else if (state.wasYawning && !isCurrentlyYawning) {
        // Mouth just closed
        const duration = Date.now() - state.mouthOpenStartTime;
        if (duration > 1000) {
            // Yawn if mouth was open for > 1 second
            state.yawnCount++;
        }
        state.wasYawning = false;
    }
}

function updateDrowsinessState() {
    const isEyeCurrentlyClosed = state.currentEAR < CONFIG.EAR_THRESHOLD;

    if (isEyeCurrentlyClosed) {
        if (!state.eyeClosedStartTime) {
            state.eyeClosedStartTime = Date.now();
        }

        const closedDuration = Date.now() - state.eyeClosedStartTime;
        state.drowsinessScore = Math.min((closedDuration / CONFIG.DROWSY_TIME_MS) * 100, 100);

        if (closedDuration >= CONFIG.DROWSY_TIME_MS && !state.isAlerting) {
            triggerAlert();
        }
    } else {
        state.eyeClosedStartTime = null;
        state.drowsinessScore = Math.max(state.drowsinessScore - 5, 0);

        if (state.drowsinessScore < 30 && state.isAlerting) {
            stopAlert();
        }
    }
}

// ============================================
// Visualization
// ============================================
function drawVisualization(keypoints) {
    const ctx = state.ctx;

    // Use a simpler scale factor that guarantees visibility
    // On high-res camera (1280 wide), scale is 2. On 640, it's 1.
    const scale = Math.max(state.canvas.width / 640, 1.0);

    // Draw face mesh - Increase contrast and size
    ctx.fillStyle = '#3b82f6';
    keypoints.forEach(point => {
        ctx.beginPath();
        // Min 2px radius for visibility
        ctx.arc(point.x, point.y, Math.max(1.5 * scale, 2), 0, 2 * Math.PI);
        ctx.fill();
    });

    // Draw eye contours
    const points = keypoints.map(kp => [kp.x, kp.y]);
    const lineWidth = Math.max(2 * scale, 3); // Ensure min width of 3px

    drawEyeContour(ctx, points, LANDMARKS.leftEye, state.currentEAR < CONFIG.EAR_THRESHOLD, lineWidth);
    drawEyeContour(ctx, points, LANDMARKS.rightEye, state.currentEAR < CONFIG.EAR_THRESHOLD, lineWidth);
    drawMouthContour(ctx, points, lineWidth);

    // Red alert overlay
    if (state.isAlerting) {
        ctx.fillStyle = 'rgba(239, 68, 68, 0.2)';
        ctx.fillRect(0, 0, state.canvas.width, state.canvas.height);
    }

    // Copyright Watermark - ONLY if NOT in camera mode
    if (!state.isCameraMode) {
        ctx.save();
        ctx.translate(state.canvas.width, 0);
        ctx.scale(-1, 1);

        // Revert to fixed small professional size
        ctx.font = '500 14px Inter, sans-serif';
        ctx.fillStyle = 'rgba(255, 255, 255, 0.8)';
        ctx.textAlign = 'right';
        ctx.fillText('© Vigilance AI | Ahmed Tarek', state.canvas.width - 15, state.canvas.height - 15);
        ctx.restore();
    }
}

function drawEyeContour(ctx, points, indices, isClosed, lineWidth) {
    ctx.beginPath();
    const eyePoints = indices.map(i => points[i]).filter(p => p);

    if (eyePoints.length < 6) return;

    ctx.moveTo(eyePoints[0][0], eyePoints[0][1]);
    for (let i = 1; i < eyePoints.length; i++) {
        ctx.lineTo(eyePoints[i][0], eyePoints[i][1]);
    }
    ctx.closePath();

    ctx.strokeStyle = isClosed ? '#ef4444' : '#10b981';
    ctx.lineWidth = lineWidth;
    ctx.stroke();
}

function drawMouthContour(ctx, points, lineWidth) {
    const mouthTop = points[LANDMARKS.mouthTop];
    const mouthBottom = points[LANDMARKS.mouthBottom];
    const mouthLeft = points[LANDMARKS.mouthLeft];
    const mouthRight = points[LANDMARKS.mouthRight];

    if (!mouthTop || !mouthBottom || !mouthLeft || !mouthRight) return;

    ctx.beginPath();
    ctx.ellipse(
        (mouthLeft[0] + mouthRight[0]) / 2,
        (mouthTop[1] + mouthBottom[1]) / 2,
        distance(mouthLeft, mouthRight) / 2,
        distance(mouthTop, mouthBottom) / 2,
        0, 0, 2 * Math.PI
    );

    ctx.strokeStyle = state.currentMAR > CONFIG.MAR_THRESHOLD ? '#f59e0b' : '#8b5cf6';
    ctx.lineWidth = lineWidth;
    ctx.stroke();
}

// ============================================
// UI Updates
// ============================================
function updateDisplay() {
    // EAR
    elements.earValue.textContent = state.currentEAR.toFixed(3);
    const earPercent = Math.min((state.currentEAR / 0.4) * 100, 100);
    elements.earBar.style.width = `${earPercent}%`;

    if (state.currentEAR < CONFIG.EAR_THRESHOLD) {
        elements.earBar.classList.add('danger');
    } else {
        elements.earBar.classList.remove('danger');
    }

    // MAR
    elements.marValue.textContent = state.currentMAR.toFixed(3);
    const marPercent = Math.min((state.currentMAR / 1.0) * 100, 100);
    elements.marBar.style.width = `${marPercent}%`;

    // Drowsiness Score
    elements.scoreValue.textContent = Math.round(state.drowsinessScore);

    // Update score circle
    const circumference = 2 * Math.PI * 45;
    const offset = circumference - (state.drowsinessScore / 100) * circumference;
    elements.scoreFill.style.strokeDashoffset = offset;

    // Score color & label
    if (state.drowsinessScore < 30) {
        elements.scoreFill.style.stroke = '#10b981';
        elements.scoreLabel.textContent = 'Alert';
        elements.scoreLabel.className = 'status-label status-good';
    } else if (state.drowsinessScore < 70) {
        elements.scoreFill.style.stroke = '#f59e0b';
        elements.scoreLabel.textContent = 'Moderate';
        elements.scoreLabel.className = 'status-label status-moderate';
    } else {
        elements.scoreFill.style.stroke = '#ef4444';
        elements.scoreLabel.textContent = 'Drowsy!';
        elements.scoreLabel.className = 'status-label status-danger';
    }

    // Counters
    elements.blinkCount.textContent = state.blinkCount;
    elements.yawnCount.textContent = state.yawnCount;
    elements.alertCount.textContent = state.alertCount;
}

function updateFPS(startTime) {
    state.frameCount++;
    const now = performance.now();

    state.fpsHistory.push(now - startTime);
    if (state.fpsHistory.length > CONFIG.FPS_SAMPLE_SIZE) {
        state.fpsHistory.shift();
    }

    if (now - state.lastFpsUpdate > 500) {
        const avgFrameTime = state.fpsHistory.reduce((a, b) => a + b, 0) / state.fpsHistory.length;
        state.fps = Math.round(1000 / avgFrameTime);
        elements.fpsBadge.querySelector('span').textContent = `${state.fps} FPS`;
        state.lastFpsUpdate = now;
    }
}

function updateSessionTime() {
    const elapsed = Math.floor((Date.now() - state.sessionStartTime) / 1000);
    const minutes = Math.floor(elapsed / 60).toString().padStart(2, '0');
    const seconds = (elapsed % 60).toString().padStart(2, '0');
    elements.sessionTime.textContent = `${minutes}:${seconds}`;
}

function startSessionTimer() {
    // Clear any existing interval to prevent multiple timers
    if (state.sessionTimerInterval) {
        clearInterval(state.sessionTimerInterval);
    }
    state.sessionTimerInterval = setInterval(updateSessionTime, 1000);
}

function stopSessionTimer() {
    if (state.sessionTimerInterval) {
        clearInterval(state.sessionTimerInterval);
        state.sessionTimerInterval = null;
    }
}

// ============================================
// Chart
// ============================================
function setupChart() {
    const ctx = elements.metricsChart.getContext('2d');

    state.chart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [
                {
                    label: 'EAR',
                    data: [],
                    borderColor: '#3b82f6',
                    backgroundColor: 'rgba(59, 130, 246, 0.1)',
                    fill: true,
                    tension: 0.4,
                    pointRadius: 0,
                },
                {
                    label: 'MAR',
                    data: [],
                    borderColor: '#ec4899',
                    backgroundColor: 'rgba(236, 72, 153, 0.1)',
                    fill: true,
                    tension: 0.4,
                    pointRadius: 0,
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: false,
            scales: {
                x: {
                    display: false,
                },
                y: {
                    min: 0,
                    max: 1,
                    ticks: {
                        stepSize: 0.25,
                        color: '#94a3b8',
                    },
                    grid: {
                        color: 'rgba(148, 163, 184, 0.1)',
                    }
                }
            },
            plugins: {
                legend: {
                    display: true,
                    position: 'top',
                    labels: {
                        boxWidth: 10,
                        padding: 10,
                        color: '#94a3b8',
                    }
                }
            }
        }
    });
}

function updateChart() {
    if (!state.chart) return;

    state.earHistory.push(state.currentEAR);
    state.marHistory.push(state.currentMAR);

    if (state.earHistory.length > CONFIG.CHART_MAX_POINTS) {
        state.earHistory.shift();
        state.marHistory.shift();
    }

    state.chart.data.labels = state.earHistory.map((_, i) => i);
    state.chart.data.datasets[0].data = state.earHistory;
    state.chart.data.datasets[1].data = state.marHistory;
    state.chart.update('none');
}

// ============================================
// Alerts
// ============================================
function triggerAlert() {
    state.isAlerting = true;
    state.alertCount++;

    // Show alert banner
    elements.alertBanner.classList.remove('hidden');

    // Play sound
    if (state.soundEnabled && elements.alertSound) {
        elements.alertSound.play().catch(() => { });
    }

    // Update status badge
    elements.statusBadge.classList.remove('badge-success');
    elements.statusBadge.classList.add('badge-danger');
    elements.statusBadge.querySelector('span').textContent = 'ALERT!';
}

function stopAlert() {
    state.isAlerting = false;

    // Hide alert banner
    elements.alertBanner.classList.add('hidden');

    // Stop sound
    if (elements.alertSound) {
        elements.alertSound.pause();
        elements.alertSound.currentTime = 0;
    }

    // Update status badge
    elements.statusBadge.classList.remove('badge-danger');
    elements.statusBadge.classList.add('badge-success');
    elements.statusBadge.querySelector('span').textContent = 'Active';
}

function toggleSound() {
    state.soundEnabled = !state.soundEnabled;
    const icon = elements.soundBtn.querySelector('i');

    if (state.soundEnabled) {
        icon.className = 'fas fa-volume-up';
    } else {
        icon.className = 'fas fa-volume-mute';
        // Immediately silence if playing
        if (elements.alertSound) {
            elements.alertSound.pause();
        }
    }
}

// ============================================
// Theme
// ============================================
function toggleTheme() {
    state.isDarkTheme = !state.isDarkTheme;
    applyTheme();
    localStorage.setItem('theme', state.isDarkTheme ? 'dark' : 'light');
}

function loadTheme() {
    const saved = localStorage.getItem('theme');
    state.isDarkTheme = saved !== 'light';
    applyTheme();
}

function applyTheme() {
    const icon = elements.themeToggle.querySelector('i');

    if (state.isDarkTheme) {
        document.documentElement.setAttribute('data-theme', 'dark');
        icon.className = 'fas fa-sun';
    } else {
        document.documentElement.removeAttribute('data-theme');
        icon.className = 'fas fa-moon';
    }
}

// ============================================
// Security
// ============================================
function setupSecurity() {
    // Disable right-click
    document.addEventListener('contextmenu', event => event.preventDefault());

    // Disable keyboard shortcuts
    document.addEventListener('keydown', (e) => {
        // F12
        if (e.key === 'F12') {
            e.preventDefault();
            return false;
        }

        // Ctrl+Shift+I, Ctrl+Shift+J, Ctrl+Shift+C
        if (e.ctrlKey && e.shiftKey && (e.key === 'I' || e.key === 'J' || e.key === 'C')) {
            e.preventDefault();
            return false;
        }

        // Ctrl+U (View Source)
        if (e.ctrlKey && e.key === 'u') {
            e.preventDefault();
            return false;
        }
    });

    console.log("%c⚠️ SECURITY WARNING ⚠️", "color: red; font-size: 20px; font-weight: bold;");
    console.log("%cThis application is protected. Unauthorized inspection is prohibited.", "font-size: 14px;");
}

// ============================================
// Start App
// ============================================
document.addEventListener('DOMContentLoaded', init);
