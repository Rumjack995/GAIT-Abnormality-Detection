"""
Train a Biomechanical Gait Classifier using MediaPipe Pose Landmarks.

This model uses 33-point 3D skeletons to calculate:
- Joint angles (Knee, Hip, Ankle)
- Step length and asymmetry
- Trunk posture (Lean)
- Arm swing analysis

This approach is far more robust than pixel-based methods for detecting
Hemiplegic and Parkinsonian gait.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Input
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils import class_weight
import pickle
from pathlib import Path
import matplotlib.pyplot as plt

project_root = Path(__file__).parent.parent


def calculate_angle(a, b, c):
    """Calculate angle between three points (in degrees)."""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
        
    return angle


def extract_biomechanical_features(landmarks_sequence):
    """
    Convert raw landmarks (frames, 33*4) into biomechanical features.
    Features per frame:
    - Knee Flexion (L/R)
    - Hip Extension (L/R)
    - Trunk Lean
    - Arm Swing (L/R)
    - Step Width
    """
    features_sequence = []
    
    # MediaPipe Landmark Indices
    # 11=L_Shoulder, 12=R_Shoulder
    # 23=L_Hip, 24=R_Hip
    # 25=L_Knee, 26=R_Knee
    # 27=L_Ankle, 28=R_Ankle
    
    for frame in landmarks_sequence:
        # Reshape flat array to (33, 4) -> (x, y, z, vis)
        lm = frame.reshape(33, 4)
        
        # Helper to get coords (x, y)
        p = lambda i: lm[i, :2]
        
        # 1. Knee Angles (Hip-Knee-Ankle)
        l_knee_angle = calculate_angle(p(23), p(25), p(27))
        r_knee_angle = calculate_angle(p(24), p(26), p(28))
        
        # 2. Hip Angles (Shoulder-Hip-Knee)
        l_hip_angle = calculate_angle(p(11), p(23), p(25))
        r_hip_angle = calculate_angle(p(12), p(24), p(26))
        
        # 3. Trunk Lean (MidShoulder to MidHip angle vs Vertical)
        mid_shoulder = (p(11) + p(12)) / 2
        mid_hip = (p(23) + p(24)) / 2
        trunk_vec = mid_shoulder - mid_hip
        trunk_angle = np.arctan2(trunk_vec[0], -trunk_vec[1]) * 180 / np.pi
        
        # Calculate Leg Length (for normalization)
        # Distance from Hip(23/24) to Ankle(27/28)
        # We use Avg of L and R leg length as the "Scale Factor" for this frame
        l_leg_len = np.linalg.norm(p(23) - p(27))
        r_leg_len = np.linalg.norm(p(24) - p(28))
        scale_factor = (l_leg_len + r_leg_len) / 2.0
        if scale_factor == 0: scale_factor = 1.0 # Prevent div/0

        # ... (Angles are fine, they are scale invariant) ...
        
        # 4. Step Width (Horizontal distance between ankles) - NORMALIZED
        step_width = abs(lm[27, 0] - lm[28, 0]) / scale_factor
        
        # 5. Arm Swing (Shoulder-Elbow-Wrist is not swing, we want Shoulder-Hip-Wrist angle maybe?)
        # Let's use simple arm angle: Shoulder-Elbow-Wrist
        l_arm_angle = calculate_angle(p(11), p(13), p(15))
        r_arm_angle = calculate_angle(p(12), p(14), p(16))
        
        # 6. Foot Height (z-coord or y-coord diff relative to hip) - NORMALIZED
        # Actually, Foot Height typically means distance from ankle to ground?
        # But we don't know ground.
        # Original: abs(lm[23, 1] - lm[27, 1]) is vertical distance Hip-to-Ankle.
        # This is just "Leg Compression" during stance/swing. 
        # Normalized by leg length, this becomes "Leg Straightness Ratio".
        l_foot_lift = abs(lm[23, 1] - lm[27, 1]) / scale_factor
        r_foot_lift = abs(lm[24, 1] - lm[28, 1]) / scale_factor
        
        features = [
            l_knee_angle, r_knee_angle,
            l_hip_angle, r_hip_angle,
            trunk_angle,
            step_width,
            l_arm_angle, r_arm_angle,
            l_foot_lift, r_foot_lift,
        ]
        
        features_sequence.append(features)
        
    # Convert to array
    features_array = np.array(features_sequence)
    
    # SMOOTHING (Critical for Velocity)
    # Raw features are jittery. Velocity (diff) amplifies jitter.
    # We must smooth the signal first.
    from scipy.signal import savgol_filter
    
    # window_length must be odd and <= length of signal
    window = min(7, len(features_array))
    if window % 2 == 0: window -= 1
    
    if window > 3:
        try:
            # Smooth features along time axis (axis 0)
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


def build_model(input_shape, num_classes):
    """LSTM model for biomechanical features."""
    model = Sequential([
        Input(shape=input_shape),
        
        # LSTM layers to capture temporal gait cycle
        LSTM(64, return_sequences=True),
        BatchNormalization(),
        Dropout(0.3),
        
        LSTM(32, return_sequences=False),
        BatchNormalization(),
        Dropout(0.3),
        
        # Dense classifier
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def main():
    print("Loading extracted pose data...")
    dataset_path = project_root / "data" / "gait_pose_dataset.npz"
    if not dataset_path.exists():
        print("Dataset not found! Run extract_poses.py first.")
        return
        
    data = np.load(dataset_path, allow_pickle=True)
    X_raw = data['X'] # (samples, frames, 132)
    y_raw = data['y']
    
    print(f"Raw data shape: {X_raw.shape}")
    print(f"Calculating biomechanical features...")
    
    X_features = []
    for sample in X_raw:
        feats = extract_biomechanical_features(sample)
        X_features.append(feats)
    
    X = np.array(X_features)
    print(f"Feature shape: {X.shape}") # (samples, frames, 10)
    
    # Normalize features
    # Reshape, fit scaler, reshape back
    samples, time_steps, features = X.shape
    X_flat = X.reshape(-1, features)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_flat).reshape(samples, time_steps, features)
    
    # Encode labels
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Calculate class weights for imbalance
    class_weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weight_dict = dict(enumerate(class_weights))
    print(f"Class weights: {class_weight_dict}")
    
    # Build Model
    model = build_model((time_steps, features), len(le.classes_))
    model.summary()
    
    # Train
    history = model.fit(
        X_train, y_train,
        validation_split=0.15,
        epochs=100,
        batch_size=16,
        class_weight=class_weight_dict,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(patience=7)
        ]
    )
    
    # Evaluate
    loss, acc = model.evaluate(X_test, y_test)
    print(f"\nTest Accuracy: {acc*100:.2f}%")
    
    # Detailed Report
    from sklearn.metrics import classification_report
    y_pred = np.argmax(model.predict(X_test), axis=1)
    
    print("\nClassification Report (Check F1-Scores for Bias):")
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    
    # Save
    models_dir = project_root / "models"
    models_dir.mkdir(exist_ok=True)
    
    model.save(models_dir / "pose_gait_model.keras")
    
    with open(models_dir / "pose_label_encoder.pkl", 'wb') as f:
        pickle.dump(le, f)
        
    with open(models_dir / "pose_scaler.pkl", 'wb') as f:
        pickle.dump(scaler, f)
        
    print("Model and preprocessors saved.")

if __name__ == "__main__":
    main()
