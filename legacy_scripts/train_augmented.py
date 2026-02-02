"""
Train gait classification models on the augmented dataset.

Uses the augmented_gait_dataset.npz with 7 gait categories:
- normal, ataxic, hemiplegic, circumduction, limping, no_arm_swing, slouch

Estimated training time:
- LSTM model: ~3-5 minutes (CPU)
- CNN-1D model: ~2-4 minutes (CPU)
- Hybrid model: ~5-8 minutes (CPU)
- Total: ~10-20 minutes on CPU

With GPU: ~3-5 minutes total
"""

import os
import sys
import time
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def load_dataset(data_path):
    """Load the augmented gait dataset."""
    print("=" * 60)
    print("Loading Dataset")
    print("=" * 60)
    
    data = np.load(data_path, allow_pickle=True)
    X = data['X'].astype(np.float32)
    y = data['y']
    
    print(f"  Samples: {X.shape[0]}")
    print(f"  Sequence length: {X.shape[1]}")
    print(f"  Features: {X.shape[2]}")
    
    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    print(f"\n  Classes: {list(le.classes_)}")
    print(f"  Num classes: {len(le.classes_)}")
    
    return X, y_encoded, le


def create_data_splits(X, y, test_size=0.2, val_size=0.15):
    """Create train/val/test splits."""
    print("\n" + "=" * 60)
    print("Creating Data Splits")
    print("=" * 60)
    
    # First split: train+val vs test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    # Second split: train vs val
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio, random_state=42, stratify=y_temp
    )
    
    print(f"  Train: {len(X_train)} samples")
    print(f"  Val: {len(X_val)} samples")
    print(f"  Test: {len(X_test)} samples")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def build_lstm_model(input_shape, num_classes):
    """Build LSTM model for sequence classification."""
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


def build_cnn1d_model(input_shape, num_classes):
    """Build 1D CNN model for sequence classification."""
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization
    
    model = Sequential([
        Conv1D(64, kernel_size=5, activation='relu', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.2),
        
        Conv1D(128, kernel_size=3, activation='relu'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.2),
        
        Conv1D(64, kernel_size=3, activation='relu'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.2),
        
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def build_hybrid_model(input_shape, num_classes):
    """Build hybrid CNN-LSTM model."""
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout, BatchNormalization
    
    model = Sequential([
        Conv1D(64, kernel_size=5, activation='relu', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.2),
        
        Conv1D(128, kernel_size=3, activation='relu'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.2),
        
        LSTM(64, return_sequences=False),
        Dropout(0.3),
        
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def train_model(model, model_name, X_train, y_train, X_val, y_val, epochs=30, batch_size=32):
    """Train a model and return results."""
    import tensorflow as tf
    
    print(f"\n[TRAINING] {model_name}")
    print("-" * 40)
    
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3
        )
    ]
    
    start_time = time.time()
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    train_time = time.time() - start_time
    
    print(f"\n  Training time: {train_time:.1f}s")
    print(f"  Best val accuracy: {max(history.history['val_accuracy']):.4f}")
    
    return model, history, train_time


def evaluate_model(model, model_name, X_test, y_test, label_encoder):
    """Evaluate model on test set."""
    print(f"\n[EVALUATION] {model_name}")
    print("-" * 40)
    
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"  Test Loss: {loss:.4f}")
    print(f"  Test Accuracy: {accuracy:.4f}")
    
    # Get predictions
    y_pred = model.predict(X_test, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    # Classification report
    print("\n  Classification Report:")
    report = classification_report(
        y_test, y_pred_classes,
        target_names=label_encoder.classes_,
        digits=3
    )
    print(report)
    
    return accuracy, y_pred_classes


def main():
    """Main training script."""
    print("\n" + "=" * 60)
    print("GAIT CLASSIFICATION MODEL TRAINING")
    print("=" * 60)
    
    # Check for GPU
    import tensorflow as tf
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"[GPU] Found {len(gpus)} GPU(s)")
        print("  Estimated total time: 3-5 minutes")
    else:
        print("[CPU] Running on CPU")
        print("  Estimated total time: 10-20 minutes")
    
    # Load data - use final dataset with real ataxia data
    data_path = project_root / "data" / "final_gait_dataset.npz"
    if not data_path.exists():
        data_path = project_root / "data" / "augmented_gait_dataset.npz"
    X, y, label_encoder = load_dataset(data_path)
    
    # Create splits
    X_train, X_val, X_test, y_train, y_val, y_test = create_data_splits(X, y)
    
    # Model parameters
    input_shape = (X.shape[1], X.shape[2])  # (120, 15)
    num_classes = len(label_encoder.classes_)
    
    # Store results
    results = {}
    models_dir = project_root / "models"
    models_dir.mkdir(exist_ok=True)
    
    total_start = time.time()
    
    # Train LSTM model
    print("\n" + "=" * 60)
    print("Model 1: LSTM")
    print("=" * 60)
    lstm_model = build_lstm_model(input_shape, num_classes)
    lstm_model.summary()
    lstm_model, lstm_history, lstm_time = train_model(
        lstm_model, "LSTM", X_train, y_train, X_val, y_val
    )
    lstm_acc, _ = evaluate_model(lstm_model, "LSTM", X_test, y_test, label_encoder)
    results['LSTM'] = {'accuracy': lstm_acc, 'time': lstm_time}
    lstm_model.save(models_dir / "lstm_gait_model.keras")
    
    # Train CNN-1D model
    print("\n" + "=" * 60)
    print("Model 2: CNN-1D")
    print("=" * 60)
    cnn_model = build_cnn1d_model(input_shape, num_classes)
    cnn_model.summary()
    cnn_model, cnn_history, cnn_time = train_model(
        cnn_model, "CNN-1D", X_train, y_train, X_val, y_val
    )
    cnn_acc, _ = evaluate_model(cnn_model, "CNN-1D", X_test, y_test, label_encoder)
    results['CNN-1D'] = {'accuracy': cnn_acc, 'time': cnn_time}
    cnn_model.save(models_dir / "cnn1d_gait_model.keras")
    
    # Train Hybrid model
    print("\n" + "=" * 60)
    print("Model 3: Hybrid (CNN + LSTM)")
    print("=" * 60)
    hybrid_model = build_hybrid_model(input_shape, num_classes)
    hybrid_model.summary()
    hybrid_model, hybrid_history, hybrid_time = train_model(
        hybrid_model, "Hybrid", X_train, y_train, X_val, y_val
    )
    hybrid_acc, _ = evaluate_model(hybrid_model, "Hybrid", X_test, y_test, label_encoder)
    results['Hybrid'] = {'accuracy': hybrid_acc, 'time': hybrid_time}
    hybrid_model.save(models_dir / "hybrid_gait_model.keras")
    
    total_time = time.time() - total_start
    
    # Save label encoder
    import pickle
    with open(models_dir / "label_encoder.pkl", 'wb') as f:
        pickle.dump(label_encoder, f)
    
    # Print summary
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    print(f"\nTotal training time: {total_time/60:.1f} minutes")
    
    print("\n  Model Comparison:")
    print("-" * 40)
    best_model = None
    best_acc = 0
    for name, data in results.items():
        print(f"  {name:12} | Accuracy: {data['accuracy']:.4f} | Time: {data['time']:.1f}s")
        if data['accuracy'] > best_acc:
            best_acc = data['accuracy']
            best_model = name
    
    print(f"\n  Best Model: {best_model} (Accuracy: {best_acc:.4f})")
    
    print(f"\n  Models saved to: {models_dir}")
    print("    - lstm_gait_model.keras")
    print("    - cnn1d_gait_model.keras")
    print("    - hybrid_gait_model.keras")
    print("    - label_encoder.pkl")


if __name__ == "__main__":
    main()
