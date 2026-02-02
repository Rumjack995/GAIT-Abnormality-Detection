"""
Gait Analysis Web Application (Pose Estimation Version)

A Flask-based web interface for testing gait classification with video uploads.
Now uses MediaPipe Pose Estimation for advanced biomechanical analysis.
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import pickle
import time
import mediapipe as mp

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

app = Flask(__name__, 
            template_folder=str(project_root / 'web' / 'templates'),
            static_folder=str(project_root / 'web' / 'static'))

app.config['UPLOAD_FOLDER'] = str(project_root / 'web' / 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max
app.config['ALLOWED_EXTENSIONS'] = {'mp4', 'avi', 'mov', 'mkv', 'webm'}

# Create upload folder
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global model variables
model = None
label_encoder = None
scaler = None
USE_POSE_MODEL = False

# Initialize MediaPipe
try:
    mp_pose = mp.solutions.pose
except AttributeError:
    import mediapipe.python.solutions.pose as mp_pose

pose_extractor = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def load_model():
    """Load the trained model and preprocessors."""
    global model, label_encoder, scaler, USE_POSE_MODEL
    
    models_dir = project_root / 'models'
    
    # Try to load the Pose model first (Best)
    pose_model_path = models_dir / 'pose_gait_model.keras'
    lstm_model_path = models_dir / 'lstm_gait_model.keras'
    
    if pose_model_path.exists():
        print(f"[OK] Found Pose Estimation Model: {pose_model_path.name}")
        model_path = pose_model_path
        USE_POSE_MODEL = True
    elif lstm_model_path.exists():
        print(f"[OK] Found Standard Model: {lstm_model_path.name}")
        model_path = lstm_model_path
        USE_POSE_MODEL = False
    else:
        print("[WARNING] No trained model found")
        model = None
        return
    
    try:
        import tensorflow as tf
        model = tf.keras.models.load_model(str(model_path))
        print(f"[OK] Loaded model successfully")
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        model = None
    
    # Load label encoder
    encoder_filename = 'pose_label_encoder.pkl' if USE_POSE_MODEL else 'label_encoder.pkl'
    encoder_path = models_dir / encoder_filename
    
    if encoder_path.exists():
        with open(encoder_path, 'rb') as f:
            label_encoder = pickle.load(f)
        print(f"[OK] Loaded label encoder: {list(label_encoder.classes_)}")
    else:
        print("[WARNING] No label encoder found")
        label_encoder = None
        
    # Load scaler (only for pose model)
    if USE_POSE_MODEL:
        scaler_path = models_dir / 'pose_scaler.pkl'
        if scaler_path.exists():
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
            print("[OK] Loaded feature scaler")
        else:
            print("[WARNING] No scaler found for pose model!")
            scaler = None


def calculate_angle(a, b, c):
    """Calculate angle between three points."""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
    return angle


def extract_pose_features(video_path, max_frames=150):
    """Extract 33-point landmarks + biomechanical features using MediaPipe."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, total_frames // max_frames) if total_frames > max_frames else 1
    
    landmarks_sequence = []
    frame_idx = 0
    count = 0
    
    while cap.isOpened() and count < max_frames:
        success, image = cap.read()
        if not success:
            break
            
        if frame_idx % step == 0:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose_extractor.process(image_rgb)
            
            if results.pose_landmarks:
                frame_landmarks = []
                for lm in results.pose_landmarks.landmark:
                    frame_landmarks.extend([lm.x, lm.y, lm.z, lm.visibility])
                landmarks_sequence.append(frame_landmarks)
            elif landmarks_sequence:
                landmarks_sequence.append(landmarks_sequence[-1]) # Pad with previous
            else:
                landmarks_sequence.append([0.0] * (33 * 4)) # Pad with zero
            
            count += 1
        frame_idx += 1
        
    cap.release()
    
    if len(landmarks_sequence) < 10:
        return None
        
    # Pad to max_frames
    landmarks_array = np.array(landmarks_sequence, dtype=np.float32)
    if len(landmarks_array) < max_frames:
        padding = np.zeros((max_frames - len(landmarks_array), 33 * 4), dtype=np.float32)
        landmarks_array = np.vstack([landmarks_array, padding])
    else:
        landmarks_array = landmarks_array[:max_frames]
        
    # Calculate Biomechanical Features
    features_sequence = []
    
    for frame in landmarks_array:
        lm = frame.reshape(33, 4)
        p = lambda i: lm[i, :2]
        
        # Calculate Leg Length (for normalization)
        l_leg_len = np.linalg.norm(p(23) - p(27))
        r_leg_len = np.linalg.norm(p(24) - p(28))
        scale_factor = (l_leg_len + r_leg_len) / 2.0
        if scale_factor == 0: scale_factor = 1.0

        # Angles
        l_knee = calculate_angle(p(23), p(25), p(27))
        r_knee = calculate_angle(p(24), p(26), p(28))
        l_hip = calculate_angle(p(11), p(23), p(25))
        r_hip = calculate_angle(p(12), p(24), p(26))
        
        mid_shoulder = (p(11) + p(12)) / 2
        mid_hip = (p(23) + p(24)) / 2
        trunk_vec = mid_shoulder - mid_hip
        trunk = np.arctan2(trunk_vec[0], -trunk_vec[1]) * 180 / np.pi
        
        # Normalized Distances
        step_w = abs(lm[27, 0] - lm[28, 0]) / scale_factor
        
        l_arm = calculate_angle(p(11), p(13), p(15))
        r_arm = calculate_angle(p(12), p(14), p(16))
        
        l_foot = abs(lm[23, 1] - lm[27, 1]) / scale_factor
        r_foot = abs(lm[24, 1] - lm[28, 1]) / scale_factor
        
        features_sequence.append([
            l_knee, r_knee, l_hip, r_hip, trunk,
            step_w, l_arm, r_arm, l_foot, r_foot
        ])
        
    # Convert to array
    features_array = np.array(features_sequence)
    
    # SMOOTHING (Critical for Velocity)
    from scipy.signal import savgol_filter
    
    window = min(7, len(features_array))
    if window % 2 == 0: window -= 1
    
    if window > 3:
        try:
            features_smoothed = savgol_filter(features_array, window_length=window, polyorder=2, axis=0)
        except:
            features_smoothed = features_array
    else:
        features_smoothed = features_array

    # Calculate Velocities from SMOOTHED features
    velocities = np.diff(features_smoothed, axis=0, prepend=features_smoothed[0:1])
    
    # Combine Position (Smoothed) + Velocity
    combined_features = np.concatenate([features_smoothed, velocities], axis=1)
        
    return combined_features


def calculate_clinical_metrics(landmarks_array, fps=30):
    """
    Calculate clinical gait parameters from landmark time-series.
    """
    # 1. Cadence (Steps per minute)
    # Detect heel strikes using Z-coordinate minima of indices 29/30 (Heels) or 27/28 (Ankles)
    # Let's use ankles (27=Left, 28=Right) y-coord (vertical movement)
    
    metrics = {}
    
    # Calculate step events (simple peak detection on vertical velocity)
    l_ankle_y = landmarks_array[:, 27, 1]
    r_ankle_y = landmarks_array[:, 28, 1]
    
    # Find peaks (heel strikes approx)
    from scipy.signal import find_peaks
    # Invert because y increases downwards in image coords
    # But usually we want local maxima of height (which is local minima of y)
    l_peaks, _ = find_peaks(-l_ankle_y, distance=fps/2) 
    r_peaks, _ = find_peaks(-r_ankle_y, distance=fps/2)
    
    total_steps = len(l_peaks) + len(r_peaks)
    duration_min = len(landmarks_array) / fps / 60
    
    cadence = int(total_steps / duration_min) if duration_min > 0 else 0
    metrics['cadence'] = cadence
    
    # 2. Symmetry Index (Step time difference)
    if len(l_peaks) > 1 and len(r_peaks) > 1:
        l_step_time = np.mean(np.diff(l_peaks))
        r_step_time = np.mean(np.diff(r_peaks))
        # Symmetry index: 1.0 = perfect. 
        symmetry = min(l_step_time, r_step_time) / max(l_step_time, r_step_time)
        metrics['symmetry'] = round(symmetry, 2)
    else:
        metrics['symmetry'] = 1.0 # Fallback
        
    # 3. Step Width (Average lateral distance)
    # x-coords difference
    step_width_avg = np.mean(np.abs(landmarks_array[:, 27, 0] - landmarks_array[:, 28, 0]))
    metrics['step_width'] = round(float(step_width_avg), 3)
    
    # 4. Gait Velocity (Estimated)
    # Hard to estimate absolute velocity without calibration.
    # We can estimate "movement intensity"
    metrics['velocity'] = "N/A" # Requires calibration
    
    return metrics


def extract_pose_data_for_ui(video_path, max_frames=300):
    """
    Extract data specifically for UI visualization (Graphs & Overlay).
    Returns:
    - time_series: { 'frames', 'l_knee', 'r_knee', 'l_hip', 'r_hip' }
    - landmarks: raw 3D coords for overlay
    - metrics: dictionary of calculated values
    """
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    
    landmarks_sequence = []
    frames = []
    
    count = 0
    while cap.isOpened() and count < max_frames:
        success, image = cap.read()
        if not success: break
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose_extractor.process(image_rgb)
        
        if results.pose_landmarks:
            lm_list = []
            for lm in results.pose_landmarks.landmark:
                lm_list.append([lm.x, lm.y, lm.z])
            landmarks_sequence.append(lm_list)
            frames.append(count)
        
        count += 1
    
    cap.release()
    
    if not landmarks_sequence:
        return None
        
    landmarks_array = np.array(landmarks_sequence) # (T, 33, 3)
    
    # Calculate Time Series Angles for Graphs
    l_knee_angles = []
    r_knee_angles = []
    l_hip_angles = []
    r_hip_angles = []
    
    for frame in landmarks_array:
        # Indices: 11/12=Shoulders, 23/24=Hips, 25/26=Knees, 27/28=Ankles
        p = lambda i: frame[i, :2]
        
        l_knee_angles.append(calculate_angle(p(23), p(25), p(27)))
        r_knee_angles.append(calculate_angle(p(24), p(26), p(28)))
        l_hip_angles.append(calculate_angle(p(11), p(23), p(25)))
        r_hip_angles.append(calculate_angle(p(12), p(24), p(26)))
        
    metrics = calculate_clinical_metrics(landmarks_array, fps)
    
    return {
        'timeseries': {
            'frames': frames,
            'l_knee': l_knee_angles,
            'r_knee': r_knee_angles,
            'l_hip': l_hip_angles,
            'r_hip': r_hip_angles
        },
        'metrics': metrics,
        'landmarks': landmarks_sequence[::5] # Downsample for overlay to save bandwidth
    }


def classify_video(video_path):
    """Classify video using the loaded model."""
    global model, label_encoder, scaler, USE_POSE_MODEL
    
    if model is None:
        return {'success': False, 'error': 'Model not loaded.'}
    
    start_time = time.time()
    
    if USE_POSE_MODEL:
        # 1. Extract Classification Features (Velocity+Pos) for Model
        features = extract_pose_features(video_path)
        if features is None:
            return {'success': False, 'error': 'Could not extract skeleton from video.'}
            
        # 2. Extract UI Features (for Graphs/Overlay)
        ui_data = extract_pose_data_for_ui(video_path)
        
        # 3. Normalize Features for Prediction
        if scaler:
            b, t, f = 1, features.shape[0], features.shape[1]
            features_flat = features.reshape(-1, f)
            features_scaled = scaler.transform(features_flat)
            features = features_scaled.reshape(b, t, f)
        else:
            features = np.expand_dims(features, axis=0) # (1, T, F)
            
    else:
        # Legacy Optical Flow path
        return {'success': False, 'error': 'Legacy model not supported in this update. Please train pose model.'}
    
    # Predict
    predictions = model.predict(features, verbose=0)[0]
    
    # Process Results
    predicted_idx = np.argmax(predictions)
    confidence = float(predictions[predicted_idx])
    
    if label_encoder:
        predicted_class = label_encoder.inverse_transform([predicted_idx])[0]
        all_classes = list(label_encoder.classes_)
    else:
        predicted_class = str(predicted_idx)
        all_classes = [str(i) for i in range(len(predictions))]
        
    class_probs = {cls: float(prob) for cls, prob in zip(all_classes, predictions)}
    processing_time = time.time() - start_time
    
    response = {
        'success': True,
        'prediction': predicted_class,
        'confidence': confidence,
        'processing_time': round(processing_time, 2),
        'all_probabilities': class_probs
    }
    
    # Merge UI Data
    if USE_POSE_MODEL and ui_data:
        response.update(ui_data)
        
    return response


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'video' not in request.files:
        return jsonify({'success': False, 'error': 'No video provided'})
    file = request.files['video']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No file selected'})
    
    if allowed_file(file.filename):
        filename = secure_filename(file.filename)
        timestamp = int(time.time())
        save_name = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], save_name)
        file.save(filepath)
        
        result = classify_video(filepath)
        result['filename'] = save_name
        return jsonify(result)
        
    return jsonify({'success': False, 'error': 'Invalid file type'})


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/api/model-status')
def model_status():
    return jsonify({
        'model_loaded': model is not None,
        'type': 'Pose Estimation (MediaPipe)' if USE_POSE_MODEL else 'Optical Flow',
        'classes': list(label_encoder.classes_) if label_encoder else []
    })


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("GAIT ANALYSIS WEB APPLICATION (POSE EDITION)")
    print("=" * 60)
    load_model()
    app.run(debug=True, host='0.0.0.0', port=5000)
