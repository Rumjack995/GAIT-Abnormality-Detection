"""
train_all_models.py - Trains LSTM, Conv1D-LSTM, and 1D-CNN on pose features.
Run: python scripts/train_all_models.py
"""
import os, sys, pickle, warnings
import numpy as np
from pathlib import Path

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings('ignore')

import cv2
import mediapipe as mp
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from scipy.signal import savgol_filter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils import class_weight as sk_class_weight
from sklearn.metrics import classification_report

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

MAX_FRAMES    = 120
N_FEATURES    = 15
EPOCHS        = 80
BATCH_SIZE    = 16
TEST_SPLIT    = 0.20
VAL_SPLIT     = 0.15
CATEGORIES    = ['normal', 'parkinsonian', 'hemiplegic', 'ataxic', 'other_abnormal']
MODELS_DIR    = project_root / 'models'
DATA_DIR      = project_root / 'data' / 'raw'
DATASET_CACHE = project_root / 'data' / 'gait_features_120x15.npz'
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    rad = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = abs(rad * 180.0 / np.pi)
    return 360 - angle if angle > 180.0 else angle


def landmarks_to_features(arr):
    """Convert (T, 132) landmark array to (T, 15) biomechanical features."""
    seq = []
    for frame in arr:
        lm = frame.reshape(33, 4)
        p  = lambda i: lm[i, :2]
        l_leg = np.linalg.norm(p(23)-p(27)); r_leg = np.linalg.norm(p(24)-p(28))
        sf = max((l_leg + r_leg) / 2.0, 1e-6)
        l_k = calculate_angle(p(23), p(25), p(27)); r_k = calculate_angle(p(24), p(26), p(28))
        l_h = calculate_angle(p(11), p(23), p(25)); r_h = calculate_angle(p(12), p(24), p(26))
        tv  = (p(11)+p(12))/2 - (p(23)+p(24))/2
        trunk = np.arctan2(tv[0], -tv[1]) * 180 / np.pi
        sw  = abs(lm[27,0]-lm[28,0]) / sf
        l_a = calculate_angle(p(11), p(13), p(15)); r_a = calculate_angle(p(12), p(14), p(16))
        l_f = abs(lm[23,1]-lm[27,1]) / sf; r_f = abs(lm[24,1]-lm[28,1]) / sf
        seq.append([l_k, r_k, l_h, r_h, trunk, sw, l_a, r_a, l_f, r_f,
                    l_k-r_k, l_h-r_h, l_a-r_a, l_f-r_f, (l_h+r_h)/(l_k+r_k+1e-6)])
    feat = np.array(seq, dtype=np.float32)
    w = min(7, len(feat)); w = w-1 if w%2==0 else w
    if w > 3:
        try: feat = savgol_filter(feat, w, 2, axis=0).astype(np.float32)
        except: pass
    return feat


def process_video(path, pose_model):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened(): return None
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step  = max(1, total // MAX_FRAMES) if total > MAX_FRAMES else 1
    seq, count, idx = [], 0, 0
    while cap.isOpened() and count < MAX_FRAMES:
        ok, img = cap.read()
        if not ok: break
        if idx % step == 0:
            res = pose_model.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            if res.pose_landmarks:
                lm = []
                for p in res.pose_landmarks.landmark: lm.extend([p.x, p.y, p.z, p.visibility])
                seq.append(lm)
            elif seq: seq.append(seq[-1])
            else: seq.append([0.0]*132)
            count += 1
        idx += 1
    cap.release()
    if len(seq) < 20: return None
    arr = np.array(seq, dtype=np.float32)
    if len(arr) < MAX_FRAMES:
        arr = np.vstack([arr, np.zeros((MAX_FRAMES-len(arr), 132), np.float32)])
    else:
        arr = arr[:MAX_FRAMES]
    return landmarks_to_features(arr)


def build_dataset():
    if DATASET_CACHE.exists():
        print(f"\n[CACHE] Loading {DATASET_CACHE.name}")
        d = np.load(DATASET_CACHE, allow_pickle=True)
        return d['X'], d['y']
    print("\n[EXTRACTION] Processing videos...")
    try: mp_pose = mp.solutions.pose
    except AttributeError:
        import mediapipe.python.solutions.pose as mp_pose
    pose = mp_pose.Pose(static_image_mode=False, model_complexity=1,
                        smooth_landmarks=True, enable_segmentation=False,
                        min_detection_confidence=0.5, min_tracking_confidence=0.5)
    X_data, y_data = [], []
    for cat in CATEGORIES:
        cat_dir = DATA_DIR / cat
        if not cat_dir.exists():
            print(f"  [SKIP] {cat}: {cat_dir}"); continue
        vids = list(cat_dir.glob('*.mp4')) + list(cat_dir.glob('*.avi')) + \
               list(cat_dir.glob('*.mov')) + list(cat_dir.glob('*.webm'))
        print(f"  {cat}: {len(vids)} videos")
        for vp in vids:
            f = process_video(vp, pose)
            if f is not None: X_data.append(f); y_data.append(cat)
    pose.close()
    if not X_data:
        raise RuntimeError(
            f"No videos found in {DATA_DIR}\n"
            "Ensure videos exist under data/raw/<category>/ folders.\n"
            "Categories: " + str(CATEGORIES))
    X = np.array(X_data, dtype=np.float32); y = np.array(y_data)
    DATASET_CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(DATASET_CACHE, X=X, y=y)
    print(f"  Saved dataset: {X.shape} → {DATASET_CACHE.name}")
    return X, y


def build_lstm_model(T, F, C):
    inp = keras.Input((T, F), name='pose_input')
    x = layers.LayerNormalization()(inp)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.1))(x)
    x = layers.BatchNormalization()(x)
    att = layers.Softmax(axis=1)(layers.Dense(1, activation='tanh')(x))
    x = layers.Multiply()([x, att])
    x = layers.Lambda(lambda t: tf.reduce_sum(t, axis=1))(x)
    x = layers.Dense(64, activation='relu')(x); x = layers.Dropout(0.3)(x)
    x = layers.Dense(32, activation='relu')(x); x = layers.Dropout(0.2)(x)
    out = layers.Dense(C, activation='softmax', name='predictions')(x)
    m = keras.Model(inp, out, name='lstm_attention')
    m.compile(optimizer=keras.optimizers.Adam(1e-3),
              loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return m


def build_cnn_lstm_model(T, F, C):
    inp = keras.Input((T, F), name='pose_input')
    x = layers.Conv1D(64, 5, activation='relu', padding='same')(inp)
    x = layers.BatchNormalization()(x); x = layers.MaxPooling1D(2)(x); x = layers.Dropout(0.2)(x)
    x = layers.Conv1D(128, 3, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x); x = layers.MaxPooling1D(2)(x); x = layers.Dropout(0.2)(x)
    x = layers.LSTM(64, dropout=0.2)(x); x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation='relu')(x); x = layers.BatchNormalization()(x)
    out = layers.Dense(C, activation='softmax', name='predictions')(x)
    m = keras.Model(inp, out, name='cnn_lstm_hybrid')
    m.compile(optimizer=keras.optimizers.Adam(1e-3),
              loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return m


def build_cnn_model(T, F, C):
    inp = keras.Input((T, F), name='pose_input')
    x = layers.Conv1D(32, 5, activation='relu', padding='same')(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(32, 5, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x); x = layers.MaxPooling1D(2)(x); x = layers.Dropout(0.2)(x)
    x = layers.Conv1D(64, 3, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(64, 3, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x); x = layers.MaxPooling1D(2)(x); x = layers.Dropout(0.2)(x)
    x = layers.Conv1D(128, 3, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x); x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(64, activation='relu')(x); x = layers.Dropout(0.3)(x)
    out = layers.Dense(C, activation='softmax', name='predictions')(x)
    m = keras.Model(inp, out, name='deep_cnn_1d')
    m.compile(optimizer=keras.optimizers.Adam(1e-3),
              loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return m


def get_callbacks(name, patience=15):
    return [
        keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=patience,
                                       restore_best_weights=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                           patience=7, min_lr=1e-6, verbose=1),
    ]


def main():
    print("=" * 60)
    print("  GAIT ANALYSIS — TRAINING 3 MODELS FROM SCRATCH")
    print("=" * 60)

    X, y_raw = build_dataset()
    print(f"\nDataset: {X.shape}  Classes: {np.unique(y_raw)}")

    le = LabelEncoder(); y = le.fit_transform(y_raw); C = len(le.classes_)
    print(f"Encoded {C} classes: {list(le.classes_)}")

    S, T, F = X.shape
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X.reshape(-1,F)).reshape(S,T,F).astype(np.float32)

    Xtv, Xte, ytv, yte = train_test_split(Xs, y, test_size=TEST_SPLIT, random_state=42, stratify=y)
    Xtr, Xv, ytr, yv   = train_test_split(Xtv, ytv,
                          test_size=VAL_SPLIT/(1-TEST_SPLIT), random_state=42, stratify=ytv)
    print(f"Train:{len(Xtr)}  Val:{len(Xv)}  Test:{len(Xte)}")

    cw = dict(enumerate(sk_class_weight.compute_class_weight('balanced',
                         classes=np.unique(ytr), y=ytr)))

    with open(MODELS_DIR / 'pose_label_encoder.pkl', 'wb') as f: pickle.dump(le, f)
    with open(MODELS_DIR / 'pose_scaler.pkl', 'wb') as f: pickle.dump(scaler, f)
    print("Saved preprocessors.")

    models_cfg = [
        ('lstm_model',     build_lstm_model(T, F, C)),
        ('cnn_lstm_model', build_cnn_lstm_model(T, F, C)),
        ('cnn_model',      build_cnn_model(T, F, C)),
    ]

    results = {}
    for mname, model in models_cfg:
        print(f"\n{'='*60}\n  {mname.upper()}\n{'='*60}")
        print(f"  Params: {model.count_params():,}  Input: {model.input_shape}")
        model.fit(Xtr, ytr, validation_data=(Xv, yv), epochs=EPOCHS,
                  batch_size=BATCH_SIZE, class_weight=cw,
                  callbacks=get_callbacks(mname), verbose=1)
        loss, acc = model.evaluate(Xte, yte, verbose=0)
        yp = np.argmax(model.predict(Xte, verbose=0), axis=1)
        print(f"\n  Test Accuracy: {acc*100:.2f}%  Loss: {loss:.4f}")
        print(classification_report(yte, yp, target_names=le.classes_))
        model.save(str(MODELS_DIR / f'{mname}.keras'))
        print(f"  Saved: {mname}.keras")
        results[mname] = acc

    print(f"\n{'='*60}\n  SUMMARY\n{'='*60}")
    for n, a in results.items(): print(f"  {n:20s}  {a*100:.2f}%")
    best = max(results, key=results.get)
    print(f"\n  Best: {best} ({results[best]*100:.2f}%)")
    print("\nAll models saved to models/")


if __name__ == '__main__':
    main()
