"""
Train gait abnormality detection models

This script trains all three model architectures (3D-CNN, LSTM, Hybrid)
on the preprocessed GAVD dataset and compares their performance.
"""

import os
import sys
import json
import pickle
import numpy as np
from pathlib import Path
from datetime import datetime
import tensorflow as tf

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from gait_analysis.training import ModelTrainer
from gait_analysis.models import (
    create_lightweight_3dcnn,
    create_efficient_lstm,
    create_hybrid_cnn_lstm
)
from gait_analysis.feature_extraction import FeatureExtractor


def setup_gpu():
    """Configure GPU settings for optimal performance."""
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Enable memory growth
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"✅ GPU available: {len(gpus)} device(s)")
            print(f"   {gpus[0].name}")
        except RuntimeError as e:
            print(f"⚠️  GPU configuration error: {e}")
    else:
        print("⚠️  No GPU detected - training will use CPU (slower)")


def load_dataset_splits(processed_dir):
    """Load dataset splits."""
    splits_file = processed_dir / 'dataset_splits.json'
    
    if not splits_file.exists():
        raise FileNotFoundError(
            f"Dataset splits not found: {splits_file}\n"
            "Please run preprocess_dataset.py first."
        )
    
    with open(splits_file, 'r') as f:
        splits = json.load(f)
    
    return splits


def load_processed_data(file_path):
    """Load a single processed data file."""
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data


def prepare_data_for_training(splits, processed_dir, architecture_type):
    """
    Prepare data for training based on architecture type.
    
    Args:
        splits: Dataset splits
        processed_dir: Directory with processed data
        architecture_type: '3dcnn', 'lstm', or 'hybrid'
        
    Returns:
        Tuple of (X_train, y_train, X_val, y_val, class_labels)
    """
    print(f"\n📊 Preparing data for {architecture_type.upper()} architecture...")
    
    feature_extractor = FeatureExtractor()
    
    # Get unique categories
    all_categories = set()
    for split_data in splits.values():
        for item in split_data:
            all_categories.add(item['category'])
    
    class_labels = sorted(list(all_categories))
    label_to_idx = {label: idx for idx, label in enumerate(class_labels)}
    
    print(f"Classes: {class_labels}")
    
    def load_split_data(split_name):
        X_data = []
        y_data = []
        
        print(f"  Loading {split_name} data...")
        for item in splits[split_name]:
            try:
                # Load processed data
                data = load_processed_data(item['path'])
                pose_sequence = data['pose_sequence']
                
                # Extract features based on architecture
                if architecture_type == '3dcnn':
                    # For 3D-CNN: use frame sequences with spatial structure
                    features = feature_extractor.extract_spatiotemporal_features(
                        pose_sequence
                    )
                elif architecture_type == 'lstm':
                    # For LSTM: use temporal features
                    features = feature_extractor.extract_temporal_features(
                        pose_sequence
                    )
                else:  # hybrid
                    # For Hybrid: use both spatial and temporal
                    features = feature_extractor.extract_spatiotemporal_features(
                        pose_sequence
                    )
                
                X_data.append(features)
                y_data.append(label_to_idx[item['category']])
                
            except Exception as e:
                print(f"    ⚠️  Error loading {item['path']}: {e}")
                continue
        
        return np.array(X_data), np.array(y_data)
    
    X_train, y_train = load_split_data('train')
    X_val, y_val = load_split_data('val')
    
    print(f"\n✅ Data prepared:")
    print(f"   Train: {X_train.shape}, {y_train.shape}")
    print(f"   Val: {X_val.shape}, {y_val.shape}")
    
    return X_train, y_train, X_val, y_val, class_labels


def train_architecture(architecture_type, X_train, y_train, X_val, y_val, 
                       class_labels, models_dir):
    """
    Train a specific architecture.
    
    Args:
        architecture_type: '3dcnn', 'lstm', or 'hybrid'
        X_train, y_train: Training data
        X_val, y_val: Validation data
        class_labels: List of class labels
        models_dir: Directory to save models
        
    Returns:
        Training history
    """
    print("\n" + "=" * 60)
    print(f"Training {architecture_type.upper()} Model")
    print("=" * 60)
    
    # Create model
    input_shape = X_train.shape[1:]
    num_classes = len(class_labels)
    
    if architecture_type == '3dcnn':
        model = create_lightweight_3dcnn(
            input_shape=input_shape,
            num_classes=num_classes
        )
    elif architecture_type == 'lstm':
        model = create_efficient_lstm(
            input_shape=input_shape,
            num_classes=num_classes
        )
    else:  # hybrid
        model = create_hybrid_cnn_lstm(
            input_shape=input_shape,
            num_classes=num_classes
        )
    
    print(f"\n📊 Model Summary:")
    model.summary()
    
    # Create trainer
    trainer = ModelTrainer(
        model=model,
        model_name=f"{architecture_type}_gait_classifier"
    )
    
    # Train model
    history = trainer.train(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        epochs=50,
        batch_size=8,  # Adjust based on GPU memory
        use_data_augmentation=True
    )
    
    # Save model
    model_path = models_dir / f"{architecture_type}_model.h5"
    model.save(model_path)
    print(f"\n✅ Model saved to: {model_path}")
    
    # Save training history
    history_path = models_dir / f"{architecture_type}_history.json"
    with open(history_path, 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        history_dict = {k: [float(v) for v in vals] 
                       for k, vals in history.history.items()}
        json.dump(history_dict, f, indent=2)
    
    return history


def compare_models(models_dir):
    """Compare performance of all trained models."""
    print("\n" + "=" * 60)
    print("Model Comparison")
    print("=" * 60)
    
    architectures = ['3dcnn', 'lstm', 'hybrid']
    results = {}
    
    for arch in architectures:
        history_path = models_dir / f"{arch}_history.json"
        
        if history_path.exists():
            with open(history_path, 'r') as f:
                history = json.load(f)
            
            # Get best validation accuracy
            best_val_acc = max(history.get('val_accuracy', [0]))
            final_val_acc = history.get('val_accuracy', [0])[-1]
            
            results[arch] = {
                'best_val_accuracy': best_val_acc,
                'final_val_accuracy': final_val_acc
            }
            
            print(f"\n{arch.upper()}:")
            print(f"  Best Val Accuracy: {best_val_acc:.4f}")
            print(f"  Final Val Accuracy: {final_val_acc:.4f}")
    
    # Save comparison
    comparison_path = models_dir / "model_comparison.json"
    with open(comparison_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Determine best model
    if results:
        best_arch = max(results.items(), 
                       key=lambda x: x[1]['best_val_accuracy'])
        print(f"\n🏆 Best Model: {best_arch[0].upper()}")
        print(f"   Accuracy: {best_arch[1]['best_val_accuracy']:.4f}")


def main():
    """Main training function."""
    print("\n🚀 Model Training Pipeline")
    print("=" * 60)
    
    # Setup paths
    project_root = Path(__file__).parent.parent
    processed_dir = project_root / "data" / "processed"
    models_dir = project_root / "models"
    models_dir.mkdir(exist_ok=True)
    
    # Setup GPU
    setup_gpu()
    
    # Load dataset splits
    try:
        splits = load_dataset_splits(processed_dir)
    except FileNotFoundError as e:
        print(f"\n❌ {e}")
        return
    
    # Train each architecture
    architectures = ['lstm', '3dcnn', 'hybrid']  # LSTM first (fastest)
    
    for arch in architectures:
        try:
            # Prepare data
            X_train, y_train, X_val, y_val, class_labels = \
                prepare_data_for_training(splits, processed_dir, arch)
            
            # Train model
            history = train_architecture(
                arch, X_train, y_train, X_val, y_val, 
                class_labels, models_dir
            )
            
        except Exception as e:
            print(f"\n❌ Error training {arch}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Compare models
    compare_models(models_dir)
    
    print("\n" + "=" * 60)
    print("✅ Training Complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Evaluate models on test set")
    print("  2. Run inference on new videos")
    print("  3. Deploy best model")


if __name__ == "__main__":
    main()
