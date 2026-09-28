"""
Facial Sleepiness & Fatigue Assessment Engine
================================================
Analyzes observable facial and head-movement indicators associated with
current sleepiness and fatigue:
- Eye Aspect Ratio (EAR) & Eye Closure Duration
- Percentage of Eye Closure (PERCLOS)
- Blink Detection & Frequency Analysis
- Mouth Aspect Ratio (MAR) & Yawning Detection
- Head Nodding & Pitch Droop Detection
- Explainable Composite Sleepiness Score (0–100)

NOTE: This module evaluates observable behavioral sleepiness cues for wellness
awareness and does not diagnose medical sleep disorders.
"""

import math
import numpy as np
import cv2

# ==============================================================================
# PROJECT ASSESSMENT THRESHOLDS (Documented for Academic Transparency)
# ==============================================================================
# Eye Aspect Ratio (EAR) thresholds:
EAR_OPEN_THRESHOLD = 0.25          # EAR >= 0.25 indicates fully open eyes
EAR_PARTIAL_THRESHOLD = 0.20       # 0.20 <= EAR < 0.25 indicates partially closed / drowsy droop
EAR_CLOSED_THRESHOLD = 0.20        # EAR < 0.20 indicates closed eyes

# PERCLOS threshold (Percentage of Eye Closure P80):
PERCLOS_EAR_CUTOFF = 0.21          # Frames with EAR < 0.21 are counted toward closure
PERCLOS_NORMAL_MAX = 8.0           # Under 8% is typical alert baseline
PERCLOS_MODERATE_MAX = 15.0        # 8% to 15% indicates moderate fatigue
                                   # > 15% indicates significant observable sleepiness

# Eye closure duration:
PROLONGED_CLOSURE_SEC = 0.40       # Closures exceeding 400ms indicate micro-sleep / heavy eyelids

# Mouth Aspect Ratio (MAR) & Yawning:
MAR_YAWN_THRESHOLD = 0.60          # MAR >= 0.60 indicates significant mouth opening
YAWN_MIN_DURATION_SEC = 1.5        # Must sustain for >= 1.5 seconds to distinguish from normal speech

# Blink Rate (Blinks per Minute):
NORMAL_BLINK_MIN = 12.0
NORMAL_BLINK_MAX = 22.0
HIGH_BLINK_THRESHOLD = 26.0        # Rapid blinking associated with ocular fatigue/dryness
LOW_BLINK_THRESHOLD = 8.0          # Staring/gaze fixation associated with visual strain

# Head Nodding:
HEAD_NOD_PITCH_THRESHOLD_DEG = 15.0 # Degrees of vertical downward drop from baseline


def euclidean_distance(p1, p2):
    """Calculates 2D Euclidean distance between two landmark points."""
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def calculate_ear(eye_points):
    """
    Computes Eye Aspect Ratio (EAR) from 6 landmark coordinates:
    p1: outer corner, p2: top-outer, p3: top-inner,
    p4: inner corner, p5: bottom-inner, p6: bottom-outer
    
    Formula: EAR = (||p2 - p6|| + ||p3 - p5||) / (2 * ||p1 - p4||)
    """
    if len(eye_points) < 6:
        return 0.0
    
    # Vertical distances
    d_v1 = euclidean_distance(eye_points[1], eye_points[5])
    d_v2 = euclidean_distance(eye_points[2], eye_points[4])
    
    # Horizontal distance
    d_h = euclidean_distance(eye_points[0], eye_points[3])
    
    if d_h == 0:
        return 0.0
    
    ear = (d_v1 + d_v2) / (2.0 * d_h)
    return round(ear, 4)


def calculate_mar(mouth_points):
    """
    Computes Mouth Aspect Ratio (MAR) from 8 landmark coordinates:
    m1, m5: left and right corners of lips
    m2, m3, m4: upper lip points
    m6, m7, m8: lower lip points
    
    Formula: MAR = (||m2 - m8|| + ||m3 - m7|| + ||m4 - m6||) / (2 * ||m1 - m5||)
    """
    if len(mouth_points) < 8:
        return 0.0
    
    d_v1 = euclidean_distance(mouth_points[1], mouth_points[7])
    d_v2 = euclidean_distance(mouth_points[2], mouth_points[6])
    d_v3 = euclidean_distance(mouth_points[3], mouth_points[5])
    d_h = euclidean_distance(mouth_points[0], mouth_points[4])
    
    if d_h == 0:
        return 0.0
    
    mar = (d_v1 + d_v2 + d_v3) / (2.0 * d_h)
    return round(mar, 4)


def compute_facial_sleepiness_score(telemetry):
    """
    Calculates an explainable, transparent Facial Sleepiness Score (0–100)
    combining PERCLOS, blink dynamics, yawning, and head movement.

    Input telemetry dictionary:
    - duration_seconds: float (e.g. 15.0)
    - total_frames: int
    - closed_frames: int (frames where EAR < PERCLOS_EAR_CUTOFF)
    - blink_count: int
    - prolonged_closure_count: int (closures > 400ms)
    - yawn_count: int (mouth openings >= 0.60 sustained for >= 1.5s)
    - head_nod_count: int
    - avg_ear: float
    - max_mar: float
    """
    duration = max(1.0, float(telemetry.get('duration_seconds', 15.0)))
    total_frames = max(1, int(telemetry.get('total_frames', 300)))
    closed_frames = int(telemetry.get('closed_frames', 0))
    blink_count = int(telemetry.get('blink_count', 0))
    prolonged_closures = int(telemetry.get('prolonged_closure_count', 0))
    yawn_count = int(telemetry.get('yawn_count', 0))
    head_nod_count = int(telemetry.get('head_nod_count', 0))
    avg_ear = float(telemetry.get('avg_ear', 0.28))
    max_mar = float(telemetry.get('max_mar', 0.35))

    # 1. PERCLOS (% of time eyes closed)
    perclos = round((closed_frames / total_frames) * 100.0, 1)

    # 2. Blink Rate per minute
    blink_rate = round((blink_count / duration) * 60.0, 1)

    # ==============================================================================
    # WEIGHTED SCORING ENGINE (Max 100 points)
    # ==============================================================================
    contributing_factors = []
    observed_indicators = []

    # Pillar A: PERCLOS Contribution (0 to 40 Points)
    # 0% PERCLOS = 0 pts; 20% PERCLOS = 40 pts
    perclos_points = min(40.0, perclos * 2.0)
    if perclos >= 15.0:
        observed_indicators.append("Elevated PERCLOS (prolonged eyelid closure)")
        contributing_factors.append({
            'factor': 'Eyelid Closure (PERCLOS)',
            'points': round(perclos_points, 1),
            'impact_level': 'High',
            'explanation': f"Eyes were closed for {perclos}% of the session (clinical drowsiness threshold > 15%)."
        })
    elif perclos >= 8.0:
        observed_indicators.append("Moderate eyelid drooping detected")
        contributing_factors.append({
            'factor': 'Eyelid Closure (PERCLOS)',
            'points': round(perclos_points, 1),
            'impact_level': 'Moderate',
            'explanation': f"Eyes were closed for {perclos}% of the observation window, indicating mild-to-moderate drowsiness."
        })
    else:
        contributing_factors.append({
            'factor': 'Eyelid Closure (PERCLOS)',
            'points': round(perclos_points, 1),
            'impact_level': 'Low',
            'explanation': f"Normal eyelid openness ({perclos}% closure time, well within the alert baseline of < 8%)."
        })

    # Pillar B: Blink Pattern & Prolonged Closures (0 to 25 Points)
    blink_points = 0.0
    if prolonged_closures > 0:
        p_pts = min(15.0, prolonged_closures * 7.5)
        blink_points += p_pts
        observed_indicators.append(f"{prolonged_closures} prolonged eye closure{'s' if prolonged_closures > 1 else ''} (>400ms)")
        contributing_factors.append({
            'factor': 'Prolonged Eye Closures',
            'points': round(p_pts, 1),
            'impact_level': 'High',
            'explanation': f"Detected {prolonged_closures} micro-sleep or slow eyelid closure events exceeding 400ms."
        })

    if blink_rate > HIGH_BLINK_THRESHOLD:
        b_pts = min(10.0, (blink_rate - HIGH_BLINK_THRESHOLD) * 1.5)
        blink_points += b_pts
        observed_indicators.append(f"Frequent blinking ({blink_rate} blinks/min)")
        contributing_factors.append({
            'factor': 'Elevated Blink Frequency',
            'points': round(b_pts, 1),
            'impact_level': 'Moderate',
            'explanation': f"Blink rate of {blink_rate} blinks/min exceeds normal resting rate (12–22), reflecting ocular strain."
        })
    elif blink_rate < LOW_BLINK_THRESHOLD and perclos > 8.0:
        b_pts = 6.0
        blink_points += b_pts
        observed_indicators.append("Reduced blink frequency / staring fatigue")
        contributing_factors.append({
            'factor': 'Reduced Blink Frequency',
            'points': round(b_pts, 1),
            'impact_level': 'Moderate',
            'explanation': f"Unusually low blink rate ({blink_rate} blinks/min) paired with eyelid droop indicates fixed-gaze fatigue."
        })
    blink_points = min(25.0, blink_points)

    # Pillar C: Yawning Detection (0 to 20 Points)
    yawn_points = 0.0
    if yawn_count >= 2:
        yawn_points = 20.0
        observed_indicators.append(f"Multiple yawns detected ({yawn_count} events)")
        contributing_factors.append({
            'factor': 'Yawning Events',
            'points': round(yawn_points, 1),
            'impact_level': 'High',
            'explanation': f"Detected {yawn_count} confirmed yawns (mouth aspect ratio >= 0.60 for >= 1.5s)."
        })
    elif yawn_count == 1:
        yawn_points = 10.0
        observed_indicators.append("Yawning detected (1 event)")
        contributing_factors.append({
            'factor': 'Yawning Events',
            'points': round(yawn_points, 1),
            'impact_level': 'Moderate',
            'explanation': "Detected 1 confirmed yawn with extended mouth elongation."
        })
    else:
        contributing_factors.append({
            'factor': 'Yawning Events',
            'points': 0.0,
            'impact_level': 'None',
            'explanation': "No yawning events observed during the assessment window."
        })

    # Pillar D: Head Nodding & Drooping (0 to 15 Points)
    head_points = 0.0
    if head_nod_count > 0:
        head_points = min(15.0, head_nod_count * 15.0)
        observed_indicators.append(f"Head nodding/dropping detected ({head_nod_count} event{'s' if head_nod_count > 1 else ''})")
        contributing_factors.append({
            'factor': 'Head Nodding / Drooping',
            'points': round(head_points, 1),
            'impact_level': 'High',
            'explanation': f"Observed {head_nod_count} involuntary head drops followed by recovery, characteristic of acute drowsiness."
        })
    else:
        contributing_factors.append({
            'factor': 'Head Nodding / Drooping',
            'points': 0.0,
            'impact_level': 'None',
            'explanation': "Head orientation remained stable throughout the session."
        })

    # Total Score
    total_score = round(min(100.0, max(0.0, perclos_points + blink_points + yawn_points + head_points)), 1)

    # Classify Sleepiness Level (Project Defined Ranges)
    if total_score <= 30.0:
        sleepiness_level = "Low"
        level_badge = "bg-success"
        recommendation = (
            "Your facial assessment shows active alertness with minimal signs of eyelid drooping or fatigue. "
            "Continue maintaining regular circadian sleep timing and hydration."
        )
    elif total_score <= 60.0:
        sleepiness_level = "Moderate"
        level_badge = "bg-warning text-dark"
        recommendation = (
            "Your facial assessment shows mild-to-moderate indicators of eye strain and fatigue. "
            "Consider stepping away from screens for a 10-15 minute break or taking a short rest before mentally demanding tasks."
        )
    else:
        sleepiness_level = "High"
        level_badge = "bg-danger"
        recommendation = (
            "Your current facial cues indicate elevated sleepiness and observable fatigue. "
            "Avoid driving or operating hazardous machinery, and prioritize getting adequate restorative sleep tonight."
        )

    if not observed_indicators:
        observed_indicators = ["Stable ocular engagement", "Normal resting blink frequency", "No drowsiness cues detected"]

    return {
        'sleepiness_score': total_score,
        'sleepiness_level': sleepiness_level,
        'level_badge': level_badge,
        'perclos_percentage': perclos,
        'blink_count': blink_count,
        'blink_rate_per_min': blink_rate,
        'prolonged_closure_count': prolonged_closures,
        'yawn_count': yawn_count,
        'head_nod_count': head_nod_count,
        'head_movement_detected': head_nod_count > 0,
        'avg_ear': round(avg_ear, 3),
        'max_mar': round(max_mar, 3),
        'duration_seconds': duration,
        'observed_indicators': observed_indicators,
        'contributing_factors': contributing_factors,
        'recommendation': recommendation,
        'disclaimer': "Facial assessment provides an estimate of observable sleepiness and fatigue cues. It is not a medical diagnostic tool."
    }


class FrameQualityValidator:
    """
    Validates frame conditions (face detection, face count, lighting, scale)
    using lightweight OpenCV computer vision.
    """
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

    def analyze_frame_bytes(self, image_bytes):
        """
        Validates frame bytes and returns face detection status, bounding box,
        and basic lighting metrics.
        """
        np_arr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img is None:
            return {'status': 'error', 'message': 'Invalid image format or corrupted frame.'}

        return self.analyze_cv2_image(img)

    def analyze_cv2_image(self, img):
        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 1. Lighting Check
        mean_brightness = float(np.mean(gray))
        if mean_brightness < 40.0:
            return {
                'status': 'warning',
                'code': 'POOR_LIGHTING',
                'message': 'Lighting is too dark. Please illuminate your face for an accurate assessment.',
                'brightness': mean_brightness
            }
        elif mean_brightness > 225.0:
            return {
                'status': 'warning',
                'code': 'OVEREXPOSED',
                'message': 'Lighting is overexposed. Please adjust position to reduce glare.',
                'brightness': mean_brightness
            }

        # 2. Face Detection
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=5, minSize=(90, 90))

        if len(faces) == 0:
            return {
                'status': 'warning',
                'code': 'NO_FACE',
                'message': 'No face detected. Please position your face inside the camera frame.',
                'face_count': 0
            }
        elif len(faces) > 1:
            return {
                'status': 'warning',
                'code': 'MULTIPLE_FACES',
                'message': f'Multiple faces detected ({len(faces)}). Please ensure only one person is in view.',
                'face_count': len(faces)
            }

        # 3. Scale and Centering Check
        fx, fy, fw, fh = faces[0]
        face_area_ratio = (fw * fh) / (w * h)
        if face_area_ratio < 0.06:
            return {
                'status': 'warning',
                'code': 'FACE_TOO_FAR',
                'message': 'Face is too far from camera. Please move closer.',
                'face_box': [int(fx), int(fy), int(fw), int(fh)]
            }

        # 4. Eye region detection
        roi_gray = gray[fy:fy + int(fh * 0.65), fx:fx + fw]
        eyes = self.eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=3, minSize=(20, 20))

        return {
            'status': 'ok',
            'code': 'FACE_TRACKED',
            'message': 'Face tracked successfully.',
            'face_count': 1,
            'face_box': [int(fx), int(fy), int(fw), int(fh)],
            'eyes_detected': len(eyes),
            'brightness': round(mean_brightness, 1)
        }
