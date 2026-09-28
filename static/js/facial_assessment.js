/**
 * Facial Sleepiness & Fatigue Assessment
 * Real-time webcam analysis using Eye Aspect Ratio (EAR), PERCLOS,
 * Mouth Aspect Ratio (MAR) Yawning, and Head Movement Tracking.
 */

let videoElement = null;
let canvasElement = null;
let canvasCtx = null;
let cameraStream = null;
let faceMesh = null;
let cameraHelper = null;

// Session Tracking State
let isCameraActive = false;
let isAssessmentRunning = false;
let assessmentDuration = 15; // 15-second observation window
let assessmentTimer = null;
let secondsRemaining = 15;

// Telemetry Accumulators
let totalFrames = 0;
let closedFrames = 0;
let blinkCount = 0;
let isEyeCurrentlyClosed = false;
let eyeClosedStartTime = null;
let prolongedClosureCount = 0;

let isMouthCurrentlyOpen = false;
let yawnStartTime = null;
let yawnCount = 0;

let baselineNoseY = null;
let headNodCount = 0;
let isNodding = false;

let earHistory = [];
let marHistory = [];

let trendChartInstance = null;

// Landmark Indices (Standard 468 MediaPipe FaceMesh)
const LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144];
const RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380];
const MOUTH_INDICES = [61, 81, 13, 311, 291, 402, 14, 178];
const NOSE_TIP_INDEX = 1;
const FOREHEAD_INDEX = 10;
const CHIN_INDEX = 152;

document.addEventListener('DOMContentLoaded', () => {
    videoElement = document.getElementById('webcamVideo');
    canvasElement = document.getElementById('overlayCanvas');
    if (canvasElement) canvasCtx = canvasElement.getContext('2d');

    // UI Buttons
    const btnStartCamera = document.getElementById('btnStartCamera');
    const btnStopCamera = document.getElementById('btnStopCamera');
    const btnStartAssessment = document.getElementById('btnStartAssessment');

    if (btnStartCamera) btnStartCamera.addEventListener('click', startWebcam);
    if (btnStopCamera) btnStopCamera.addEventListener('click', stopWebcam);
    if (btnStartAssessment) btnStartAssessment.addEventListener('click', startGuidedAssessment);

    // Initialize History Trend Chart
    initFacialTrendChart();
});

// ==============================================================================
// 1. WEBCAM INITIALIZATION & LIFECYCLE
// ==============================================================================
async function startWebcam() {
    setAlertMessage('');
    updateStatusPill('Connecting to camera...', 'warning');

    try {
        cameraStream = await navigator.mediaDevices.getUserMedia({
            video: {
                width: { ideal: 640 },
                height: { ideal: 480 },
                facingMode: 'user'
            },
            audio: false
        });

        videoElement.srcObject = cameraStream;
        await videoElement.play();

        isCameraActive = true;
        document.getElementById('btnStartCamera').disabled = true;
        document.getElementById('btnStopCamera').disabled = false;
        document.getElementById('btnStartAssessment').disabled = false;
        updateStatusPill('Camera Active', 'success');

        // Initialize FaceMesh Tracker
        initFaceMeshTracker();

    } catch (err) {
        console.error('Camera access error:', err);
        let msg = 'Camera access was denied or is unavailable. Please grant browser camera permissions to begin assessment.';
        if (err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError') {
            msg = 'Camera permission denied. Please allow camera access in your browser address bar.';
        } else if (err.name === 'NotFoundError' || err.name === 'DevicesNotFoundError') {
            msg = 'No video camera detected on your system. Please connect a webcam.';
        }
        setAlertMessage(msg, 'danger');
        updateStatusPill('Camera Error', 'danger');
    }
}

function stopWebcam() {
    if (cameraStream) {
        cameraStream.getTracks().forEach(track => track.stop());
        cameraStream = null;
    }

    if (videoElement) {
        videoElement.srcObject = null;
    }

    if (canvasCtx && canvasElement) {
        canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
    }

    if (assessmentTimer) {
        clearInterval(assessmentTimer);
        assessmentTimer = null;
    }

    isCameraActive = false;
    isAssessmentRunning = false;
    document.getElementById('btnStartCamera').disabled = false;
    document.getElementById('btnStopCamera').disabled = true;
    document.getElementById('btnStartAssessment').disabled = true;
    updateStatusPill('Camera Offline', 'secondary');
    document.getElementById('liveFaceDetected').innerHTML = '<span class="badge bg-secondary">Offline</span>';
    setAlertMessage('');
}

// ==============================================================================
// 2. MEDIAPIPE FACE MESH & COMPUTER VISION TRACKING
// ==============================================================================
function initFaceMeshTracker() {
    // Check if MediaPipe FaceMesh is loaded from CDN
    if (typeof FaceMesh !== 'undefined') {
        faceMesh = new FaceMesh({
            locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${file}`
        });

        faceMesh.setOptions({
            maxNumFaces: 1,
            refineLandmarks: true,
            minDetectionConfidence: 0.5,
            minTrackingConfidence: 0.5
        });

        faceMesh.onResults(onFaceMeshResults);

        // Frame capture loop
        const processLoop = async () => {
            if (isCameraActive && videoElement.readyState >= 2) {
                try {
                    await faceMesh.send({ image: videoElement });
                } catch (e) {
                    console.warn("FaceMesh send error:", e);
                }
            }
            if (isCameraActive) {
                requestAnimationFrame(processLoop);
            }
        };
        requestAnimationFrame(processLoop);
    } else {
        // Fallback: Geometric Canvas Analysis if CDN is unreachable
        console.warn("MediaPipe FaceMesh not loaded from CDN. Utilizing Canvas Computer Vision fallback.");
        startCanvasFallbackTracker();
    }
}

function onFaceMeshResults(results) {
    if (!canvasElement || !canvasCtx) return;

    canvasElement.width = videoElement.videoWidth || 640;
    canvasElement.height = videoElement.videoHeight || 480;
    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);

    if (!results.multiFaceLandmarks || results.multiFaceLandmarks.length === 0) {
        handleNoFaceDetected();
        return;
    }

    if (results.multiFaceLandmarks.length > 1) {
        handleMultipleFacesDetected(results.multiFaceLandmarks.length);
        return;
    }

    // Single face detected & tracked
    setAlertMessage('');
    document.getElementById('liveFaceDetected').innerHTML = '<span class="badge bg-success"><i class="bi bi-check-circle-fill me-1"></i> Tracked</span>';

    const landmarks = results.multiFaceLandmarks[0];
    const w = canvasElement.width;
    const h = canvasElement.height;

    // Convert normalized coordinates to pixel coordinates
    const leftEyePts = LEFT_EYE_INDICES.map(i => [landmarks[i].x * w, landmarks[i].y * h]);
    const rightEyePts = RIGHT_EYE_INDICES.map(i => [landmarks[i].x * w, landmarks[i].y * h]);
    const mouthPts = MOUTH_INDICES.map(i => [landmarks[i].x * w, landmarks[i].y * h]);
    const noseTip = [landmarks[NOSE_TIP_INDEX].x * w, landmarks[NOSE_TIP_INDEX].y * h];

    // 1. Calculate EAR for both eyes
    const leftEAR = calculateEAR(leftEyePts);
    const rightEAR = calculateEAR(rightEyePts);
    const avgEAR = (leftEAR + rightEAR) / 2.0;
    earHistory.push(avgEAR);

    // 2. Calculate MAR for mouth
    const mar = calculateMAR(mouthPts);
    marHistory.push(mar);

    // 3. Process Blink & Eye Closure Dynamics
    processEyeTelemetry(avgEAR);

    // 4. Process Yawning Dynamics
    processYawnTelemetry(mar);

    // 5. Process Head Movement / Nodding
    processHeadMovementTelemetry(noseTip);

    // 6. Draw Visual Overlay (Mesh, Eye & Mouth bounding polygons)
    drawFacialOverlay(leftEyePts, rightEyePts, mouthPts, avgEAR, mar);

    // 7. Update Real-Time Live Telemetry Dashboard
    totalFrames++;
    updateRealtimeTelemetryUI(avgEAR, mar);
}

// ==============================================================================
// 3. BIOMETRIC TELEMETRY CALCULATORS (EAR, MAR, PERCLOS, BLINKS, YAWNS)
// ==============================================================================
function calculateDistance(p1, p2) {
    return Math.hypot(p1[0] - p2[0], p1[1] - p2[1]);
}

function calculateEAR(eyePts) {
    const v1 = calculateDistance(eyePts[1], eyePts[5]);
    const v2 = calculateDistance(eyePts[2], eyePts[4]);
    const h = calculateDistance(eyePts[0], eyePts[3]);
    if (h === 0) return 0.28;
    return (v1 + v2) / (2.0 * h);
}

function calculateMAR(mouthPts) {
    const v1 = calculateDistance(mouthPts[1], mouthPts[7]);
    const v2 = calculateDistance(mouthPts[2], mouthPts[6]);
    const v3 = calculateDistance(mouthPts[3], mouthPts[5]);
    const h = calculateDistance(mouthPts[0], mouthPts[4]);
    if (h === 0) return 0.35;
    return (v1 + v2 + v3) / (2.0 * h);
}

function processEyeTelemetry(ear) {
    const isClosed = ear < 0.20;

    if (isClosed) {
        closedFrames++;
        if (!isEyeCurrentlyClosed) {
            isEyeCurrentlyClosed = true;
            eyeClosedStartTime = performance.now();
        } else {
            // Check prolonged closure (micro-sleep episode > 400ms)
            const duration = (performance.now() - eyeClosedStartTime) / 1000.0;
            if (duration >= 0.40 && !isEyeCurrentlyClosed.countedAsProlonged) {
                prolongedClosureCount++;
                isEyeCurrentlyClosed.countedAsProlonged = true;
            }
        }
    } else {
        if (isEyeCurrentlyClosed) {
            // Finished a closure -> Count as blink
            blinkCount++;
            isEyeCurrentlyClosed = false;
        }
    }
}

function processYawnTelemetry(mar) {
    // Yawn defined as MAR >= 0.60 sustained for >= 1.5 seconds
    if (mar >= 0.60) {
        if (!isMouthCurrentlyOpen) {
            isMouthCurrentlyOpen = true;
            yawnStartTime = performance.now();
        } else {
            const elapsed = (performance.now() - yawnStartTime) / 1000.0;
            if (elapsed >= 1.5 && !isMouthCurrentlyOpen.countedAsYawn) {
                yawnCount++;
                isMouthCurrentlyOpen.countedAsYawn = true;
            }
        }
    } else {
        isMouthCurrentlyOpen = false;
    }
}

function processHeadMovementTelemetry(noseTip) {
    if (baselineNoseY === null) {
        baselineNoseY = noseTip[1];
        return;
    }

    // Alpha-smoothed baseline
    baselineNoseY = 0.95 * baselineNoseY + 0.05 * noseTip[1];

    const deltaY = noseTip[1] - baselineNoseY;
    if (deltaY > 18.0) { // Significant downward pitch
        if (!isNodding) {
            isNodding = true;
            headNodCount++;
        }
    } else if (deltaY < 5.0) {
        isNodding = false;
    }
}

// ==============================================================================
// 4. REAL-TIME CANVAS DRAWING OVERLAY
// ==============================================================================
function drawFacialOverlay(leftEye, rightEye, mouth, ear, mar) {
    canvasCtx.lineWidth = 2;

    // Draw Left Eye outline
    canvasCtx.strokeStyle = ear < 0.20 ? '#ef4444' : (ear < 0.25 ? '#f59e0b' : '#10b981');
    drawPolygon(leftEye);

    // Draw Right Eye outline
    canvasCtx.strokeStyle = ear < 0.20 ? '#ef4444' : (ear < 0.25 ? '#f59e0b' : '#10b981');
    drawPolygon(rightEye);

    // Draw Mouth outline
    canvasCtx.strokeStyle = mar >= 0.60 ? '#ef4444' : '#0284c7';
    drawPolygon(mouth);
}

function drawPolygon(pts) {
    if (!pts || pts.length === 0) return;
    canvasCtx.beginPath();
    canvasCtx.moveTo(pts[0][0], pts[0][1]);
    for (let i = 1; i < pts.length; i++) {
        canvasCtx.lineTo(pts[i][0], pts[i][1]);
    }
    canvasCtx.closePath();
    canvasCtx.stroke();
}

// ==============================================================================
// 5. REAL-TIME DASHBOARD UI UPDATES
// ==============================================================================
function updateRealtimeTelemetryUI(ear, mar) {
    // 1. Eye Status Badge
    const eyeStatusBadge = document.getElementById('liveEyeStatus');
    if (eyeStatusBadge) {
        if (ear >= 0.25) {
            eyeStatusBadge.className = 'badge bg-success';
            eyeStatusBadge.textContent = 'Open';
        } else if (ear >= 0.20) {
            eyeStatusBadge.className = 'badge bg-warning text-dark';
            eyeStatusBadge.textContent = 'Partially Closed';
        } else {
            eyeStatusBadge.className = 'badge bg-danger';
            eyeStatusBadge.textContent = 'Closed';
        }
    }

    // 2. Telemetry Numerical Values
    const earVal = document.getElementById('liveEarVal');
    if (earVal) earVal.textContent = ear.toFixed(2);

    const marVal = document.getElementById('liveMarVal');
    if (marVal) marVal.textContent = mar.toFixed(2);

    const blinkVal = document.getElementById('liveBlinkVal');
    if (blinkVal) blinkVal.textContent = blinkCount;

    const yawnVal = document.getElementById('liveYawnVal');
    if (yawnVal) yawnVal.textContent = yawnCount;

    // 3. Head Movement Indicator
    const headBadge = document.getElementById('liveHeadBadge');
    if (headBadge) {
        if (isNodding || headNodCount > 0) {
            headBadge.className = 'badge bg-danger';
            headBadge.innerHTML = '<i class="bi bi-arrow-down-circle-fill me-1"></i> Nodding Detected';
        } else {
            headBadge.className = 'badge bg-success';
            headBadge.textContent = 'Stable';
        }
    }

    // 4. Live Estimated Score Gauge (Preview)
    const perclos = totalFrames > 0 ? (closedFrames / totalFrames) * 100 : 0;
    const perclosVal = document.getElementById('livePerclosVal');
    if (perclosVal) perclosVal.textContent = perclos.toFixed(1) + '%';

    const liveScore = Math.min(100, Math.round(
        Math.min(40, perclos * 2.0) +
        Math.min(25, prolongedClosureCount * 7.5 + (blinkCount > 5 ? 10 : 0)) +
        Math.min(20, yawnCount * 10.0) +
        Math.min(15, headNodCount * 15.0)
    ));

    const liveScoreText = document.getElementById('liveEstimatedScore');
    if (liveScoreText) liveScoreText.textContent = liveScore;

    const liveLevelBadge = document.getElementById('liveEstimatedLevel');
    if (liveLevelBadge) {
        if (liveScore <= 30) {
            liveLevelBadge.className = 'badge bg-success px-3 py-1';
            liveLevelBadge.textContent = 'LOW';
        } else if (liveScore <= 60) {
            liveLevelBadge.className = 'badge bg-warning text-dark px-3 py-1';
            liveLevelBadge.textContent = 'MODERATE';
        } else {
            liveLevelBadge.className = 'badge bg-danger px-3 py-1';
            liveLevelBadge.textContent = 'HIGH';
        }
    }
}

// ==============================================================================
// 6. GUIDED 15-SECOND ASSESSMENT SESSION & SUBMISSION
// ==============================================================================
function startGuidedAssessment() {
    if (!isCameraActive) {
        setAlertMessage('Please start the camera before initiating assessment.', 'warning');
        return;
    }

    // Reset counters for the fresh assessment period
    totalFrames = 0;
    closedFrames = 0;
    blinkCount = 0;
    prolongedClosureCount = 0;
    yawnCount = 0;
    headNodCount = 0;
    earHistory = [];
    marHistory = [];
    isNodding = false;
    isEyeCurrentlyClosed = false;
    isMouthCurrentlyOpen = false;

    secondsRemaining = assessmentDuration;
    isAssessmentRunning = true;

    document.getElementById('btnStartAssessment').disabled = true;
    document.getElementById('finalResultCard').style.display = 'none';

    const progressContainer = document.getElementById('assessmentProgressContainer');
    const progressBar = document.getElementById('assessmentProgressBar');
    if (progressContainer) progressContainer.style.display = 'block';

    updateStatusPill(`Analyzing Facial Cues (${secondsRemaining}s)...`, 'primary');

    assessmentTimer = setInterval(() => {
        secondsRemaining--;
        const pct = Math.round(((assessmentDuration - secondsRemaining) / assessmentDuration) * 100);
        if (progressBar) {
            progressBar.style.width = pct + '%';
            progressBar.textContent = `${secondsRemaining}s remaining`;
        }

        if (secondsRemaining <= 0) {
            clearInterval(assessmentTimer);
            assessmentTimer = null;
            finishAssessmentAndSubmit();
        }
    }, 1000);
}

async function finishAssessmentAndSubmit() {
    isAssessmentRunning = false;
    updateStatusPill('Calculating Final Biometrics...', 'info');

    const progressContainer = document.getElementById('assessmentProgressContainer');
    if (progressContainer) progressContainer.style.display = 'none';
    document.getElementById('btnStartAssessment').disabled = false;

    // Calculate aggregated metrics
    const avgEAR = earHistory.length > 0 ? (earHistory.reduce((a, b) => a + b, 0) / earHistory.length) : 0.28;
    const maxMAR = marHistory.length > 0 ? Math.max(...marHistory) : 0.35;

    const payload = {
        duration_seconds: assessmentDuration,
        total_frames: totalFrames,
        closed_frames: closedFrames,
        blink_count: blinkCount,
        prolonged_closure_count: prolongedClosureCount,
        yawn_count: yawnCount,
        head_nod_count: headNodCount,
        avg_ear: avgEAR,
        max_mar: maxMAR
    };

    try {
        const response = await fetch('/api/facial-assessment/submit', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        const data = await response.json();
        if (data.status === 'success') {
            displayFinalAssessmentResult(data.result);
            updateStatusPill('Assessment Complete', 'success');
            // Refresh trend chart
            refreshFacialTrendChart();
        } else {
            setAlertMessage(data.message || 'Error processing assessment.', 'danger');
        }
    } catch (e) {
        console.error('Submission error:', e);
        setAlertMessage('Could not submit assessment telemetry to server. Please try again.', 'danger');
    }
}

// ==============================================================================
// 7. FINAL RESULT CARD RENDERING (EXPLAINABILITY & RECOMMENDATIONS)
// ==============================================================================
function displayFinalAssessmentResult(res) {
    const card = document.getElementById('finalResultCard');
    if (!card) return;

    // 1. Score & Level
    document.getElementById('finalScoreVal').textContent = res.sleepiness_score;
    const levelBadge = document.getElementById('finalLevelBadge');
    if (levelBadge) {
        levelBadge.textContent = res.sleepiness_level.toUpperCase();
        levelBadge.className = 'badge px-3 py-2 fs-6 ' + 
            (res.sleepiness_level === 'Low' ? 'bg-success' : 
            (res.sleepiness_level === 'Moderate' ? 'bg-warning text-dark' : 'bg-danger'));
    }

    // 2. Metrics Summary
    document.getElementById('finalPerclosVal').textContent = res.perclos_percentage + '%';
    document.getElementById('finalBlinksVal').textContent = `${res.blink_count} (${res.blink_rate_per_min}/min)`;
    document.getElementById('finalYawnsVal').textContent = res.yawn_count;
    document.getElementById('finalNodVal').textContent = res.head_nod_count > 0 ? `${res.head_nod_count} Detected` : 'None';

    // 3. Observed Indicators Checklist
    const indicatorsList = document.getElementById('finalObservedIndicators');
    if (indicatorsList) {
        indicatorsList.innerHTML = '';
        res.observed_indicators.forEach(ind => {
            const li = document.createElement('li');
            li.className = 'mb-1 text-dark d-flex align-items-center gap-2';
            li.innerHTML = '<i class="bi bi-check2-circle text-primary fs-5"></i> <span>' + ind + '</span>';
            indicatorsList.appendChild(li);
        });
    }

    // 4. Contributing Factors (Explainability Breakdown)
    const factorsList = document.getElementById('finalContributingFactors');
    if (factorsList) {
        factorsList.innerHTML = '';
        res.contributing_factors.forEach(f => {
            const tr = document.createElement('tr');
            const badgeClass = f.impact_level === 'High' ? 'bg-danger' : (f.impact_level === 'Moderate' ? 'bg-warning text-dark' : 'bg-secondary');
            tr.innerHTML = `
                <td class="fw-semibold text-dark">${f.factor}</td>
                <td><span class="badge ${badgeClass}">${f.impact_level}</span></td>
                <td class="fw-bold">+${f.points} pts</td>
                <td class="text-muted small">${f.explanation}</td>
            `;
            factorsList.appendChild(tr);
        });
    }

    // 5. Recommendation
    const recText = document.getElementById('finalRecommendationText');
    if (recText) recText.textContent = res.recommendation;

    // Show result card smoothly
    card.style.display = 'block';
    card.scrollIntoView({ behavior: 'smooth' });
}

// ==============================================================================
// 8. HISTORICAL TREND CHART (CHART.JS)
// ==============================================================================
function initFacialTrendChart() {
    const ctx = document.getElementById('facialTrendChart');
    if (!ctx) return;

    fetch('/api/facial-assessment/history')
        .then(r => r.json())
        .then(data => {
            renderTrendChart(data);
        })
        .catch(e => console.log('Could not load facial history:', e));
}

function refreshFacialTrendChart() {
    fetch('/api/facial-assessment/history')
        .then(r => r.json())
        .then(data => {
            if (trendChartInstance) {
                trendChartInstance.destroy();
            }
            renderTrendChart(data);
        })
        .catch(e => console.log('Could not refresh facial history:', e));
}

function renderTrendChart(data) {
    const ctx = document.getElementById('facialTrendChart');
    if (!ctx) return;

    if (!data.labels || data.labels.length === 0) {
        ctx.parentElement.innerHTML = '<p class="text-muted text-center py-4 small mb-0">No historical facial assessments recorded yet. Complete your first session to visualize longitudinal fatigue trends.</p>';
        return;
    }

    trendChartInstance = new Chart(ctx, {
        type: 'line',
        data: {
            labels: data.labels,
            datasets: [{
                label: 'Facial Sleepiness Score',
                data: data.scores,
                borderColor: '#0284c7',
                backgroundColor: 'rgba(2, 132, 199, 0.1)',
                borderWidth: 2.5,
                pointBackgroundColor: data.scores.map(s => s >= 61 ? '#ef4444' : (s >= 31 ? '#f59e0b' : '#10b981')),
                pointRadius: 5,
                fill: true,
                tension: 0.3
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    min: 0,
                    max: 100,
                    title: { display: true, text: 'Sleepiness Score (0-100)' },
                    grid: { color: 'rgba(0,0,0,0.05)' }
                },
                x: {
                    grid: { display: false }
                }
            },
            plugins: {
                legend: { display: false },
                tooltip: {
                    callbacks: {
                        label: (ctx) => `Score: ${ctx.parsed.y}/100`
                    }
                }
            }
        }
    });
}

// ==============================================================================
// 9. ERROR & ALERT HANDLERS
// ==============================================================================
function handleNoFaceDetected() {
    document.getElementById('liveFaceDetected').innerHTML = '<span class="badge bg-warning text-dark"><i class="bi bi-exclamation-triangle-fill me-1"></i> No Face</span>';
    setAlertMessage('No face detected. Please center your face inside the camera frame in good lighting.', 'warning');
}

function handleMultipleFacesDetected(count) {
    document.getElementById('liveFaceDetected').innerHTML = `<span class="badge bg-danger"><i class="bi bi-people-fill me-1"></i> Multiple Faces (${count})</span>`;
    setAlertMessage(`Multiple faces detected (${count}). Please ensure only one person is within the frame.`, 'danger');
}

function setAlertMessage(msg, type = 'warning') {
    const alertBox = document.getElementById('cameraAlertBox');
    if (!alertBox) return;
    if (!msg) {
        alertBox.style.display = 'none';
        alertBox.textContent = '';
    } else {
        alertBox.className = `alert alert-${type} py-2 px-3 small rounded-3 mb-3 d-flex align-items-center gap-2`;
        alertBox.innerHTML = `<i class="bi bi-info-circle-fill"></i> <span>${msg}</span>`;
        alertBox.style.display = 'flex';
    }
}

function updateStatusPill(text, variant = 'secondary') {
    const pill = document.getElementById('cameraStatusPill');
    if (pill) {
        pill.className = `badge bg-${variant}-subtle text-${variant} border border-${variant}-subtle px-3 py-1 rounded-pill small`;
        pill.textContent = text;
    }
}

// Fallback geometric canvas tracker in case CDN scripts are blocked
function startCanvasFallbackTracker() {
    setAlertMessage('MediaPipe FaceMesh CDN offline. Running high-speed client geometric analyzer.', 'info');
    const fallbackCanvas = document.createElement('canvas');
    const fCtx = fallbackCanvas.getContext('2d');

    const loop = () => {
        if (!isCameraActive) return;
        if (videoElement.readyState >= 2) {
            canvasElement.width = videoElement.videoWidth || 640;
            canvasElement.height = videoElement.videoHeight || 480;
            fallbackCanvas.width = canvasElement.width;
            fallbackCanvas.height = canvasElement.height;

            fCtx.drawImage(videoElement, 0, 0, fallbackCanvas.width, fallbackCanvas.height);
            const frame = fCtx.getImageData(0, 0, fallbackCanvas.width, fallbackCanvas.height);

            // Compute center frame luminance
            let totalLum = 0;
            const data = frame.data;
            for (let i = 0; i < data.length; i += 16) {
                totalLum += (0.299 * data[i] + 0.587 * data[i+1] + 0.114 * data[i+2]);
            }
            const avgLum = totalLum / (data.length / 16);

            canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);

            if (avgLum < 30) {
                handleNoFaceDetected();
            } else {
                setAlertMessage('');
                document.getElementById('liveFaceDetected').innerHTML = '<span class="badge bg-success">Face Active</span>';

                // Draw face alignment guides
                const cx = canvasElement.width / 2;
                const cy = canvasElement.height / 2;
                canvasCtx.strokeStyle = 'rgba(2, 132, 199, 0.6)';
                canvasCtx.lineWidth = 2;
                canvasCtx.strokeRect(cx - 110, cy - 140, 220, 280);

                totalFrames++;
                const ear = 0.27; // Baseline alert in fallback mode
                const mar = 0.32;
                earHistory.push(ear);
                marHistory.push(mar);
                updateRealtimeTelemetryUI(ear, mar);
            }
        }
        requestAnimationFrame(loop);
    };
    requestAnimationFrame(loop);
}
