"""
Process ALL GAVD videos from real downloaded data and retrain with correct categories.

Categories from GAVD + Rochester:
- normal (139 videos)
- parkinsonian (10 videos)
- hemiplegic (18 videos)
- other_abnormal (160 videos)
- ataxic (152 videos from Rochester)
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import pickle

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def extract_motion_features(video_path, target_fps=10, max_frames=120):
    """Extract motion features from video."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    
    original_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_interval = max(1, int(original_fps / target_fps))
    
    features_list = []
    prev_gray = None
    frame_idx = 0
    
    while len(features_list) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_idx % frame_interval == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (224, 224))
            
            if prev_gray is not None:
                flow = cv2.calcOpticalFlowFarneback(
                    prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
                )
                
                mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                
                features = [
                    np.mean(mag), np.std(mag), np.max(mag),
                    np.mean(ang), np.std(ang),
                    np.mean(mag[:, :112]) - np.mean(mag[:, 112:]),
                    np.mean(mag[:112, :]) - np.mean(mag[112:, :]),
                    np.mean(mag[:112, :]), np.mean(mag[112:, :]),
                    np.mean(np.abs(np.diff(mag.flatten()))),
                    np.mean(mag[:112, :112]), np.mean(mag[:112, 112:]),
                    np.mean(mag[112:, :112]), np.mean(mag[112:, 112:]),
                    len(features_list) / max_frames
                ]
                features_list.append(features)
            
            prev_gray = gray
        frame_idx += 1
    
    cap.release()
    
    if len(features_list) < 10:
        return None
    
    features_array = np.array(features_list, dtype=np.float32)
    
    if len(features_array) < max_frames:
        padding = np.tile(features_array[-1:], (max_frames - len(features_array), 1))
        features_array = np.vstack([features_array, padding])
    else:
        features_array = features_array[:max_frames]
    
    return features_array


def process_all_videos():
    """Process all video categories."""
    data_dir = project_root / "data" / "raw"
    
    # Define categories we want
    categories = {
        'normal': data_dir / 'normal',
        'parkinsonian': data_dir / 'parkinsonian',
        'hemiplegic': data_dir / 'hemiplegic',
        'ataxic': data_dir / 'ataxic',
        'other_abnormal': data_dir / 'other_abnormal',
    }
    
    all_features = []
    all_labels = []
    
    print("=" * 60)
    print("Processing All Video Categories (Balanced: ~75 per Class)")
    print("=" * 60)
    
    LIMIT_PER_CLASS = 75
    
    for category, cat_dir in categories.items():
        if not cat_dir.exists():
            print(f"  [SKIP] {category}: directory not found")
            continue
        
        # Get all videos
        videos = list(cat_dir.glob("*.mp4")) + list(cat_dir.glob("*.webm")) + list(cat_dir.glob("*.mkv"))
        
        # Randomly sample if we have more than limit
        if len(videos) > LIMIT_PER_CLASS:
            np.random.shuffle(videos)
            videos = videos[:LIMIT_PER_CLASS]
            print(f"\n  [{category.upper()}] Processing {len(videos)} videos (Downsampled from total)")
        else:
            print(f"\n  [{category.upper()}] Processing {len(videos)} videos (All available)")
        
        for video_path in tqdm(videos, desc=f"  {category}"):
            features = extract_motion_features(video_path)
            if features is not None:
                all_features.append(features)
                all_labels.append(category)
    
    # Convert to arrays
    X = np.array(all_features, dtype=np.float32)
    y = np.array(all_labels)
    
    # Normalize features
    mean = np.mean(X, axis=(0, 1), keepdims=True)
    std = np.std(X, axis=(0, 1), keepdims=True) + 1e-7
    X = (X - mean) / std
    
    print(f"\n  Total samples: {len(X)}")
    print(f"  Shape: {X.shape}")
    
    # Class distribution
    print("\n  Class distribution:")
    unique, counts = np.unique(y, return_counts=True)
    for label, count in zip(unique, counts):
        print(f"    {label}: {count}")
    
    return X, y, mean, std


def build_lstm_model(input_shape, num_classes):
    """Build LSTM model."""
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
    
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=input_shape),
        Dropout(0.3),
        LSTM(64, return_sequences=False),
        Dropout(0.3),
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def train_model(X, y):
    """Train the model."""
    import tensorflow as tf
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    
    print("\n" + "=" * 60)
    print("Training Model")
    print("=" * 60)
    
    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    print(f"  Classes: {list(le.classes_)}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.15, random_state=42, stratify=y_train
    )
    
    print(f"  Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # Build model
    input_shape = (X.shape[1], X.shape[2])
    num_classes = len(le.classes_)
    
    model = build_lstm_model(input_shape, num_classes)
    model.summary()
    
    # Train
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=10, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=5
        )
    ]
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=16,  # Smaller batch for small dataset
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate
    print("\n  Evaluation on Test Set:")
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"  Test Accuracy: {accuracy:.4f}")
    
    # Classification report
    from sklearn.metrics import classification_report
    y_pred = model.predict(X_test, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    print("\n  Classification Report:")
    print(classification_report(y_test, y_pred_classes, target_names=le.classes_))
    
    return model, le


def main():
    print("\n" + "=" * 60)
    print("FULL VIDEO PROCESSING AND RETRAINING")
    print("=" * 60)
    
    # Process all videos
    X, y, mean, std = process_all_videos()
    
    # Save processed data
    processed_path = project_root / "data" / "real_video_dataset.npz"
    np.savez(processed_path, X=X, y=y, mean=mean, std=std)
    print(f"\n  Saved: {processed_path}")
    
    # Train model
    model, label_encoder = train_model(X, y)
    
    # Save model
    models_dir = project_root / "models"
    models_dir.mkdir(exist_ok=True)
    
    model.save(models_dir / "lstm_gait_model.keras")
    print(f"\n  Model saved: {models_dir / 'lstm_gait_model.keras'}")
    
    with open(models_dir / "label_encoder.pkl", 'wb') as f:
        pickle.dump(label_encoder, f)
    print(f"  Label encoder saved: {models_dir / 'label_encoder.pkl'}")
    
    print("\n" + "=" * 60)
    print("[SUCCESS] Retraining Complete!")
    print("=" * 60)
    print("\nRestart the web app to use the new model:")
    print("  python web\\app.py")


if __name__ == "__main__":
    main()
