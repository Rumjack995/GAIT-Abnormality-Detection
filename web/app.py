"""
Gait Analysis Web Application — Multi-Model Edition

Supports 3 models trained on the same (120, 15) biomechanical pose feature pipeline:
  1. LSTM + Attention    (lstm_model.keras)
  2. Conv1D-LSTM Hybrid  (cnn_lstm_model.keras)
  3. Deep 1D-CNN         (cnn_model.keras)

The user can select the model from the frontend before uploading.
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import warnings
warnings.filterwarnings('ignore')

import sys
import cv2
import numpy as np
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import pickle
import time
import traceback
import mediapipe as mp
from scipy.signal import savgol_filter

# ── Setup ──────────────────────────────────────────────────────────────────────
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

app = Flask(__name__,
            template_folder=str(project_root / 'web' / 'templates'),
            static_folder=str(project_root / 'web' / 'static'))

app.config['UPLOAD_FOLDER']       = str(project_root / 'web' / 'uploads')
app.config['MAX_CONTENT_LENGTH']  = 100 * 1024 * 1024   # 100 MB
app.config['ALLOWED_EXTENSIONS']  = {'mp4', 'avi', 'mov', 'mkv', 'webm'}
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ── Model registry ─────────────────────────────────────────────────────────────
MODEL_REGISTRY = {
    'lstm': {
        'label':    'LSTM + Attention',
        'file':     'lstm_model.keras',
        'desc':     'Bidirectional LSTM with self-attention. Best for capturing long-range temporal dependencies.',
        'instance': None,
    },
    'cnn_lstm': {
        'label':    'Conv1D-LSTM Hybrid',
        'file':     'cnn_lstm_model.keras',
        'desc':     'Convolutional feature extraction + LSTM temporal modelling. Best overall balance.',
        'instance': None,
    },
    'cnn': {
        'label':    'Deep 1D-CNN',
        'file':     'cnn_model.keras',
        'desc':     'Deep convolutional network. Fastest inference, great for local motion patterns.',
        'instance': None,
    },
}

# Shared preprocessors (all models use same features)
label_encoder  = None
scaler         = None
active_model_key = 'lstm'   # default

# ── MediaPipe pose ─────────────────────────────────────────────────────────────
try:
    _mp_pose = mp.solutions.pose
except AttributeError:
    import mediapipe.python.solutions.pose as _mp_pose

pose_extractor = _mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


# ══════════════════════════════════════════════════════════════════════════════
# Utilities
# ══════════════════════════════════════════════════════════════════════════════

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def calculate_angle(a, b, c):
    """Angle at point b (degrees)."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    rad = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(rad * 180.0 / np.pi)
    return 360 - angle if angle > 180.0 else angle


# ══════════════════════════════════════════════════════════════════════════════
# Model Loading
# ══════════════════════════════════════════════════════════════════════════════

def load_all_models():
    """Load all available models and shared preprocessors at startup."""
    global label_encoder, scaler

    import tensorflow as tf
    models_dir = project_root / 'models'

    print("\n" + "=" * 60)
    print("  GAIT ANALYSIS — MULTI-MODEL EDITION")
    print("=" * 60)

    # Load shared preprocessors
    enc_path = models_dir / 'pose_label_encoder.pkl'
    scl_path = models_dir / 'pose_scaler.pkl'

    if enc_path.exists():
        with open(enc_path, 'rb') as f:
            label_encoder = pickle.load(f)
        print(f"[OK] Label encoder: {list(label_encoder.classes_)}")
    else:
        print("[WARN] pose_label_encoder.pkl not found — retrain models first")

    if scl_path.exists():
        with open(scl_path, 'rb') as f:
            scaler = pickle.load(f)
        print("[OK] Feature scaler loaded")
    else:
        print("[WARN] pose_scaler.pkl not found")

    # Load each model
    loaded_count = 0
    for key, info in MODEL_REGISTRY.items():
        mpath = models_dir / info['file']
        if mpath.exists():
            try:
                m = tf.keras.models.load_model(str(mpath), compile=False, safe_mode=False)
                info['instance'] = m
                print(f"[OK] {info['label']:25s}  {m.input_shape} -> {m.output_shape}  ({m.count_params():,} params)")
                loaded_count += 1
            except Exception as e:
                print(f"[FAIL] {info['label']}: {e}")
        else:
            print(f"[MISS] {info['label']:25s}  ({info['file']} not found — run train_all_models.py)")

    if loaded_count == 0:
        print("\n[WARNING] No models loaded. Run: python scripts/train_all_models.py")
    else:
        print(f"\n{loaded_count}/{len(MODEL_REGISTRY)} models ready.")
    print("=" * 60 + "\n")


# ══════════════════════════════════════════════════════════════════════════════
# Feature Extraction  (120 frames × 15 biomechanical features)
# ══════════════════════════════════════════════════════════════════════════════

def extract_pose_features(video_path, max_frames=120):
    """Extract (120, 15) biomechanical feature matrix from a video."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step  = max(1, total // max_frames) if total > max_frames else 1

    seq, count, idx = [], 0, 0
    while cap.isOpened() and count < max_frames:
        ok, img = cap.read()
        if not ok:
            break
        if idx % step == 0:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            res = pose_extractor.process(rgb)
            if res.pose_landmarks:
                lm = []
                for p in res.pose_landmarks.landmark:
                    lm.extend([p.x, p.y, p.z, p.visibility])
                seq.append(lm)
            elif seq:
                seq.append(seq[-1])
            else:
                seq.append([0.0] * 132)
            count += 1
        idx += 1
    cap.release()

    if len(seq) < 10:
        return None

    arr = np.array(seq, dtype=np.float32)
    if len(arr) < max_frames:
        pad = np.zeros((max_frames - len(arr), 132), dtype=np.float32)
        arr = np.vstack([arr, pad])
    else:
        arr = arr[:max_frames]

    # ── Compute 15 biomechanical features per frame ────────────────────────
    features_seq = []
    for frame in arr:
        lm = frame.reshape(33, 4)
        p  = lambda i: lm[i, :2]

        l_leg = np.linalg.norm(p(23) - p(27))
        r_leg = np.linalg.norm(p(24) - p(28))
        sf    = max((l_leg + r_leg) / 2.0, 1e-6)

        l_knee = calculate_angle(p(23), p(25), p(27))
        r_knee = calculate_angle(p(24), p(26), p(28))
        l_hip  = calculate_angle(p(11), p(23), p(25))
        r_hip  = calculate_angle(p(12), p(24), p(26))

        ms    = (p(11) + p(12)) / 2
        mh    = (p(23) + p(24)) / 2
        tv    = ms - mh
        trunk = np.arctan2(tv[0], -tv[1]) * 180 / np.pi

        step_w = abs(lm[27, 0] - lm[28, 0]) / sf
        l_arm  = calculate_angle(p(11), p(13), p(15))
        r_arm  = calculate_angle(p(12), p(14), p(16))
        l_foot = abs(lm[23, 1] - lm[27, 1]) / sf
        r_foot = abs(lm[24, 1] - lm[28, 1]) / sf

        knee_asym      = l_knee - r_knee
        hip_asym       = l_hip  - r_hip
        arm_asym       = l_arm  - r_arm
        foot_asym      = l_foot - r_foot
        hip_knee_ratio = (l_hip + r_hip) / (l_knee + r_knee + 1e-6)

        features_seq.append([
            l_knee, r_knee, l_hip, r_hip, trunk,
            step_w, l_arm, r_arm, l_foot, r_foot,
            knee_asym, hip_asym, arm_asym, foot_asym, hip_knee_ratio
        ])

    feat = np.array(features_seq, dtype=np.float32)

    # Smooth
    w = min(7, len(feat))
    if w % 2 == 0: w -= 1
    if w > 3:
        try:
            feat = savgol_filter(feat, window_length=w, polyorder=2, axis=0).astype(np.float32)
        except Exception:
            pass

    return feat   # (120, 15)


def extract_pose_data_for_ui(video_path, max_frames=300):
    """Extract raw landmark data for UI graphs and overlay."""
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30

    landmarks_seq, frames = [], []
    count = 0
    while cap.isOpened() and count < max_frames:
        ok, img = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        res = pose_extractor.process(rgb)
        if res.pose_landmarks:
            lm_list = [[lm.x, lm.y, lm.z] for lm in res.pose_landmarks.landmark]
            landmarks_seq.append(lm_list)
            frames.append(count)
        count += 1
    cap.release()

    if not landmarks_seq:
        return None

    lm_arr = np.array(landmarks_seq)

    l_knee_angles, r_knee_angles = [], []
    l_hip_angles,  r_hip_angles  = [], []
    for frame in lm_arr:
        p = lambda i: frame[i, :2]
        l_knee_angles.append(calculate_angle(p(23), p(25), p(27)))
        r_knee_angles.append(calculate_angle(p(24), p(26), p(28)))
        l_hip_angles.append(calculate_angle(p(11), p(23), p(25)))
        r_hip_angles.append(calculate_angle(p(12), p(24), p(26)))

    metrics = _calculate_clinical_metrics(lm_arr, fps)

    return {
        'timeseries': {
            'frames': frames,
            'l_knee': l_knee_angles,
            'r_knee': r_knee_angles,
            'l_hip':  l_hip_angles,
            'r_hip':  r_hip_angles,
        },
        'metrics':   metrics,
        'landmarks': landmarks_seq[::5],
    }


def _calculate_clinical_metrics(lm_arr, fps=30):
    from scipy.signal import find_peaks
    metrics = {}
    l_ankle_y = lm_arr[:, 27, 1]
    r_ankle_y = lm_arr[:, 28, 1]
    l_peaks, _ = find_peaks(-l_ankle_y, distance=fps/2)
    r_peaks, _ = find_peaks(-r_ankle_y, distance=fps/2)
    total_steps   = len(l_peaks) + len(r_peaks)
    duration_min  = len(lm_arr) / fps / 60
    metrics['cadence']  = int(total_steps / duration_min) if duration_min > 0 else 0
    if len(l_peaks) > 1 and len(r_peaks) > 1:
        metrics['symmetry'] = round(
            min(np.mean(np.diff(l_peaks)), np.mean(np.diff(r_peaks))) /
            max(np.mean(np.diff(l_peaks)), np.mean(np.diff(r_peaks))), 2
        )
    else:
        metrics['symmetry'] = 1.0
    metrics['step_width'] = round(float(np.mean(np.abs(lm_arr[:, 27, 0] - lm_arr[:, 28, 0]))), 3)
    metrics['velocity']   = 'N/A'
    return metrics


# ══════════════════════════════════════════════════════════════════════════════
# Inference
# ══════════════════════════════════════════════════════════════════════════════

def classify_video(video_path, model_key='lstm'):
    """Classify video using the specified model."""
    info = MODEL_REGISTRY.get(model_key)
    if info is None or info['instance'] is None:
        return {
            'success': False,
            'error':   f"Model '{model_key}' is not loaded. Run train_all_models.py first."
        }

    model = info['instance']
    start = time.time()

    # Extract features
    features = extract_pose_features(video_path)
    if features is None:
        return {'success': False, 'error': 'Could not detect a human skeleton in the video.'}

    ui_data = extract_pose_data_for_ui(video_path)

    # Scale
    if scaler is not None:
        T, F = features.shape
        features = scaler.transform(features.reshape(-1, F)).reshape(1, T, F).astype(np.float32)
    else:
        features = np.expand_dims(features, axis=0)

    # Predict
    predictions   = model.predict(features, verbose=0)[0]
    predicted_idx = int(np.argmax(predictions))
    confidence    = float(predictions[predicted_idx])

    if label_encoder is not None:
        predicted_class = label_encoder.inverse_transform([predicted_idx])[0]
        all_classes     = list(label_encoder.classes_)
    else:
        predicted_class = str(predicted_idx)
        all_classes     = [str(i) for i in range(len(predictions))]

    class_probs      = {cls: float(prob) for cls, prob in zip(all_classes, predictions)}
    processing_time  = round(time.time() - start, 2)

    response = {
        'success':          True,
        'prediction':       predicted_class,
        'confidence':       confidence,
        'processing_time':  processing_time,
        'all_probabilities': class_probs,
        'model_used':       info['label'],
        'model_key':        model_key,
    }
    if ui_data:
        response.update(ui_data)
    return response


# ══════════════════════════════════════════════════════════════════════════════
# Flask Routes
# ══════════════════════════════════════════════════════════════════════════════

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

    # Which model did the user choose?
    model_key = request.form.get('model', active_model_key)

    if allowed_file(file.filename):
        filename  = secure_filename(file.filename)
        save_name = f"{int(time.time())}_{filename}"
        filepath  = os.path.join(app.config['UPLOAD_FOLDER'], save_name)
        file.save(filepath)

        try:
            result = classify_video(filepath, model_key=model_key)
            result['filename'] = save_name
        except Exception as e:
            print(f"[ERROR] classify_video: {traceback.format_exc()}")
            return jsonify({'success': False, 'error': f'Analysis failed: {str(e)}'})
        finally:
            # Delete video to save storage space
            if os.path.exists(filepath):
                try:
                    os.remove(filepath)
                except Exception as e:
                    print(f"[WARN] Failed to delete video: {e}")

        return jsonify(result)

    return jsonify({'success': False, 'error': 'Invalid file type'})


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/api/model-status')
def model_status():
    """Return status of all 3 models."""
    models_info = {}
    for key, info in MODEL_REGISTRY.items():
        models_info[key] = {
            'label':    info['label'],
            'desc':     info['desc'],
            'loaded':   info['instance'] is not None,
            'file':     info['file'],
        }
    return jsonify({
        'models':        models_info,
        'active_model':  active_model_key,
        'classes':       list(label_encoder.classes_) if label_encoder else [],
        'preprocessors': {
            'label_encoder': label_encoder is not None,
            'scaler':        scaler is not None,
        }
    })


@app.route('/api/set-model', methods=['POST'])
def set_model():
    """Switch the active model."""
    global active_model_key
    data = request.get_json()
    key  = data.get('model_key', 'lstm')

    if key not in MODEL_REGISTRY:
        return jsonify({'success': False, 'error': f"Unknown model: {key}"})
    if MODEL_REGISTRY[key]['instance'] is None:
        return jsonify({'success': False, 'error': f"Model '{key}' is not loaded yet. Run train_all_models.py"})

    active_model_key = key
    return jsonify({
        'success':   True,
        'model_key': key,
        'label':     MODEL_REGISTRY[key]['label'],
    })


if __name__ == '__main__':
    load_all_models()
    app.run(debug=True, host='0.0.0.0', port=5000)
