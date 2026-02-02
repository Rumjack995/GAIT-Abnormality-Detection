"""
Generate Confusion Matrix for the trained Gait Analysis model.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import tensorflow as tf
import pickle

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def plot_confusion_matrix():
    print("Loading data and model...")
    
    # Load dataset
    data_path = project_root / "data" / "real_video_dataset.npz"
    if not data_path.exists():
        print("Dataset not found!")
        return

    data = np.load(data_path, allow_pickle=True)
    X = data['X']
    y = data['y']
    
    # Load label encoder
    with open(project_root / "models" / "label_encoder.pkl", 'rb') as f:
        le = pickle.load(f)
        
    y_encoded = le.transform(y)
    
    # Split exactly as training did
    # Note: This is an approximation since we don't have the exact split indices saved
    # Ideally, we should have saved the test set separately.
    # We will execute the same random split with same seed.
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    # Load model
    model = tf.keras.models.load_model(project_root / "models" / "lstm_gait_model.keras")
    
    # Predict
    print("Running predictions on test set...")
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    # Generate Matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=le.classes_,
                yticklabels=le.classes_)
    plt.title('Confusion Matrix - Gait Classification')
    plt.ylabel('True Class')
    plt.xlabel('Predicted Class')
    
    # Save plot
    output_path = project_root / "confusion_matrix.png"
    plt.savefig(output_path)
    print(f"Confusion matrix saved to: {output_path}")
    
    # Print numerical report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    
    # Calculate per-class accuracy
    print("\nPer-class Recall (Accuracy):")
    cm_diag = cm.diagonal()
    cm_sum = cm.sum(axis=1)
    for i, label in enumerate(le.classes_):
        acc = cm_diag[i] / cm_sum[i] if cm_sum[i] > 0 else 0
        print(f"  {label}: {acc*100:.1f}%")

if __name__ == "__main__":
    plot_confusion_matrix()
