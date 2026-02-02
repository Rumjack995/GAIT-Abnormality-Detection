"""
Hybrid CNN-LSTM architecture combining MobileNetV2 backbone with LSTM layers.

This module implements the most balanced approach for spatial-temporal feature learning,
optimized for RTX 4050 with efficient inference pipeline for real-time processing.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import MobileNetV2
from typing import Tuple, Optional, Dict, Any, List
import numpy as np


class SpatialFeatureExtractor(layers.Layer):
    """
    Spatial feature extractor using MobileNetV2 backbone.
    
    This layer extracts spatial features from individual frames
    using a lightweight MobileNetV2 architecture.
    """
    
    def __init__(self, 
                 input_shape: Tuple[int, int, int] = (224, 224, 3),
                 feature_dim: int = 128,
                 trainable_backbone: bool = False,
                 **kwargs):
        """
        Initialize spatial feature extractor.
        
        Args:
            input_shape: Input frame shape (height, width, channels)
            feature_dim: Output feature dimension
            trainable_backbone: Whether to fine-tune MobileNetV2 weights
        """
        super(SpatialFeatureExtractor, self).__init__(**kwargs)
        self.input_shape = input_shape
        self.feature_dim = feature_dim
        self.trainable_backbone = trainable_backbone
        
        # MobileNetV2 backbone (lightweight and efficient)
        self.backbone = MobileNetV2(
            input_shape=input_shape,
            include_top=False,
            weights='imagenet',
            pooling='avg'
        )
        self.backbone.trainable = trainable_backbone
        
        # Feature projection layer
        self.feature_projection = layers.Dense(
            feature_dim,
            activation='relu',
            name='spatial_features'
        )
        
    def build(self, input_shape):
        """Build the layer."""
        super(SpatialFeatureExtractor, self).build(input_shape)
        
    def compute_output_shape(self, input_shape):
        """
        Compute the output shape of the layer.
        
        Args:
            input_shape: Input shape tuple
            
        Returns:
            Output shape tuple
        """
        batch_size = input_shape[0]
        return (batch_size, self.feature_dim)
    
    def call(self, inputs, training=None):
        """
        Extract spatial features from input frames.
        
        Args:
            inputs: Input frames (batch_size, height, width, channels)
            training: Training mode flag
            
        Returns:
            Spatial features (batch_size, feature_dim)
        """
        # Extract features using MobileNetV2
        backbone_features = self.backbone(inputs, training=training)
        
        # Project to desired feature dimension
        spatial_features = self.feature_projection(backbone_features)
        
        return spatial_features
    
    def get_config(self):
        """Get layer configuration."""
        config = super(SpatialFeatureExtractor, self).get_config()
        config.update({
            'input_shape': self.input_shape,
            'feature_dim': self.feature_dim,
            'trainable_backbone': self.trainable_backbone
        })
        return config


class TemporalFeatureExtractor(layers.Layer):
    """
    Temporal feature extractor using LSTM layers.
    
    This layer processes sequences of spatial features to capture
    temporal dynamics in gait patterns.
    """
    
    def __init__(self,
                 lstm_units: int = 64,
                 return_sequences: bool = False,
                 **kwargs):
        """
        Initialize temporal feature extractor.
        
        Args:
            lstm_units: Number of LSTM units
            return_sequences: Whether to return full sequence or last output
        """
        super(TemporalFeatureExtractor, self).__init__(**kwargs)
        self.lstm_units = lstm_units
        self.return_sequences = return_sequences
        
        # LSTM layer for temporal modeling
        self.lstm = layers.LSTM(
            lstm_units,
            return_sequences=return_sequences,
            dropout=0.2,
            recurrent_dropout=0.2,
            name='temporal_lstm'
        )
        
    def build(self, input_shape):
        """Build the layer."""
        super(TemporalFeatureExtractor, self).build(input_shape)
        
    def compute_output_shape(self, input_shape):
        """
        Compute the output shape of the layer.
        
        Args:
            input_shape: Input shape tuple (batch_size, sequence_length, feature_dim)
            
        Returns:
            Output shape tuple
        """
        batch_size = input_shape[0]
        if self.return_sequences:
            return (batch_size, input_shape[1], self.lstm_units)
        else:
            return (batch_size, self.lstm_units)
        
    def call(self, inputs, training=None):
        """
        Extract temporal features from spatial feature sequences.
        
        Args:
            inputs: Spatial feature sequences (batch_size, sequence_length, feature_dim)
            training: Training mode flag
            
        Returns:
            Temporal features
        """
        temporal_features = self.lstm(inputs, training=training)
        return temporal_features
    
    def get_config(self):
        """Get layer configuration."""
        config = super(TemporalFeatureExtractor, self).get_config()
        config.update({
            'lstm_units': self.lstm_units,
            'return_sequences': self.return_sequences
        })
        return config


class FusionLayer(layers.Layer):
    """
    Fusion layer for combining spatial and temporal features.
    
    This layer implements an attention-based fusion mechanism
    to combine spatial and temporal representations effectively.
    """
    
    def __init__(self, fusion_dim: int = 128, **kwargs):
        """
        Initialize fusion layer.
        
        Args:
            fusion_dim: Dimension of fused features
        """
        super(FusionLayer, self).__init__(**kwargs)
        self.fusion_dim = fusion_dim
        
        # Fusion components
        self.spatial_projection = layers.Dense(fusion_dim, name='spatial_proj')
        self.temporal_projection = layers.Dense(fusion_dim, name='temporal_proj')
        self.attention_weights = layers.Dense(2, activation='softmax', name='fusion_attention')
        self.fusion_output = layers.Dense(fusion_dim, activation='relu', name='fused_features')
        
    def build(self, input_shape):
        """Build the layer."""
        super(FusionLayer, self).build(input_shape)
        
    def compute_output_shape(self, input_shape):
        """
        Compute the output shape of the layer.
        
        Args:
            input_shape: Tuple of input shapes (spatial_shape, temporal_shape)
            
        Returns:
            Output shape tuple
        """
        spatial_shape, temporal_shape = input_shape
        batch_size = spatial_shape[0]
        return (batch_size, self.fusion_dim)
        
    def call(self, inputs, training=None):
        """
        Fuse spatial and temporal features.
        
        Args:
            inputs: Tuple of (spatial_features, temporal_features)
            training: Training mode flag
            
        Returns:
            Fused features
        """
        spatial_features, temporal_features = inputs
        
        # Project features to same dimension
        spatial_proj = self.spatial_projection(spatial_features)
        temporal_proj = self.temporal_projection(temporal_features)
        
        # Calculate attention weights
        combined = layers.concatenate([spatial_proj, temporal_proj])
        attention = self.attention_weights(combined)
        
        # Apply attention weights
        weighted_spatial = spatial_proj * attention[:, 0:1]
        weighted_temporal = temporal_proj * attention[:, 1:2]
        
        # Fuse features
        fused = weighted_spatial + weighted_temporal
        output = self.fusion_output(fused)
        
        return output
    
    def get_config(self):
        """Get layer configuration."""
        config = super(FusionLayer, self).get_config()
        config.update({'fusion_dim': self.fusion_dim})
        return config


class HybridCNNLSTM:
    """
    Hybrid CNN-LSTM architecture for gait abnormality detection.
    
    Features:
    - MobileNetV2 backbone for spatial feature extraction
    - LSTM layers for temporal modeling
    - Attention-based fusion of spatial-temporal features
    - Optimized for real-time inference on RTX 4050
    """
    
    def __init__(self,
                 frame_shape: Tuple[int, int, int] = (224, 224, 3),
                 sequence_length: int = 16,
                 num_classes: int = 5,
                 spatial_feature_dim: int = 128,
                 lstm_units: int = 64,
                 fusion_dim: int = 128,
                 trainable_backbone: bool = False,
                 use_mixed_precision: bool = True):
        """
        Initialize the hybrid CNN-LSTM model.
        
        Args:
            frame_shape: Input frame shape (height, width, channels)
            sequence_length: Number of frames in sequence
            num_classes: Number of gait abnormality classes
            spatial_feature_dim: Dimension of spatial features
            lstm_units: Number of LSTM units
            fusion_dim: Dimension of fused features
            trainable_backbone: Whether to fine-tune MobileNetV2
            use_mixed_precision: Enable FP16 mixed precision training
        """
        self.frame_shape = frame_shape
        self.sequence_length = sequence_length
        self.input_shape = (sequence_length,) + frame_shape
        self.num_classes = num_classes
        self.spatial_feature_dim = spatial_feature_dim
        self.lstm_units = lstm_units
        self.fusion_dim = fusion_dim
        self.trainable_backbone = trainable_backbone
        self.use_mixed_precision = use_mixed_precision
        self.model = None
        
        if use_mixed_precision:
            # Enable mixed precision for memory efficiency
            policy = keras.mixed_precision.Policy('mixed_float16')
            keras.mixed_precision.set_global_policy(policy)
    
    def build_model(self) -> Model:
        """
        Build the hybrid CNN-LSTM architecture.
        
        Architecture optimized for RTX 4050:
        - CNN Feature Extractor: MobileNetV2 backbone (lightweight)
        - TimeDistributed wrapper for sequence processing
        - LSTM layers: 64 units for temporal modeling
        - Fusion: Attention-based combination of features
        - Memory usage: ~3GB VRAM
        
        Returns:
            Compiled Keras model
        """
        # Input layer for video sequences
        inputs = keras.Input(shape=self.input_shape, name='video_sequence_input')
        
        # Spatial feature extraction using TimeDistributed MobileNetV2
        spatial_extractor = SpatialFeatureExtractor(
            input_shape=self.frame_shape,
            feature_dim=self.spatial_feature_dim,
            trainable_backbone=self.trainable_backbone,
            name='spatial_extractor'
        )
        
        # Apply spatial feature extraction to each frame
        spatial_features = layers.TimeDistributed(
            spatial_extractor,
            name='time_distributed_spatial'
        )(inputs)
        
        # Temporal feature extraction using LSTM
        temporal_extractor = TemporalFeatureExtractor(
            lstm_units=self.lstm_units,
            return_sequences=False,
            name='temporal_extractor'
        )
        
        temporal_features = temporal_extractor(spatial_features)
        
        # Global spatial features (average pooling over time)
        global_spatial_features = layers.GlobalAveragePooling1D(
            name='global_spatial_pooling'
        )(spatial_features)
        
        # Fusion of spatial and temporal features
        fusion_layer = FusionLayer(
            fusion_dim=self.fusion_dim,
            name='feature_fusion'
        )
        
        fused_features = fusion_layer([global_spatial_features, temporal_features])
        
        # Classification head
        x = layers.BatchNormalization(name='fusion_bn')(fused_features)
        x = layers.Dropout(0.3, name='fusion_dropout')(x)
        
        x = layers.Dense(64, activation='relu', name='classifier_dense')(x)
        x = layers.Dropout(0.3, name='classifier_dropout')(x)
        
        # Output layer with float32 for numerical stability
        if self.use_mixed_precision:
            outputs = layers.Dense(
                self.num_classes,
                activation='softmax',
                dtype='float32',
                name='predictions'
            )(x)
        else:
            outputs = layers.Dense(
                self.num_classes,
                activation='softmax',
                name='predictions'
            )(x)
        
        self.model = Model(
            inputs=inputs,
            outputs=outputs,
            name='hybrid_cnn_lstm'
        )
        
        return self.model
    
    def compile_model(self,
                     learning_rate: float = 0.001,
                     optimizer: str = 'adam') -> None:
        """
        Compile the model with appropriate optimizer and loss function.
        
        Args:
            learning_rate: Learning rate for optimizer
            optimizer: Optimizer type ('adam', 'sgd', 'rmsprop')
        """
        if self.model is None:
            raise ValueError("Model must be built before compilation")
        
        # Configure optimizer
        if optimizer == 'adam':
            opt = keras.optimizers.Adam(learning_rate=learning_rate)
        elif optimizer == 'sgd':
            opt = keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
        elif optimizer == 'rmsprop':
            opt = keras.optimizers.RMSprop(learning_rate=learning_rate)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer}")
        
        # Mixed precision optimizer wrapper
        if self.use_mixed_precision:
            opt = keras.mixed_precision.LossScaleOptimizer(opt)
        
        self.model.compile(
            optimizer=opt,
            loss='categorical_crossentropy',
            metrics=['accuracy', 'top_k_categorical_accuracy']
        )
    
    def create_inference_pipeline(self) -> Model:
        """
        Create optimized inference pipeline for real-time processing.
        
        Returns:
            Optimized model for inference
        """
        if self.model is None:
            raise ValueError("Model must be built before creating inference pipeline")
        
        # Create inference-optimized model
        inference_model = keras.models.clone_model(self.model)
        inference_model.set_weights(self.model.get_weights())
        
        # Compile with inference-optimized settings
        inference_model.compile(
            optimizer='adam',  # Optimizer not used in inference
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return inference_model
    
    def get_memory_efficient_callbacks(self,
                                     model_checkpoint_path: str = 'best_hybrid_model.h5',
                                     patience: int = 12) -> list:
        """
        Get callbacks optimized for memory-efficient training.
        
        Args:
            model_checkpoint_path: Path to save best model
            patience: Early stopping patience
            
        Returns:
            List of Keras callbacks
        """
        callbacks = [
            # Save best model
            keras.callbacks.ModelCheckpoint(
                filepath=model_checkpoint_path,
                monitor='val_accuracy',
                save_best_only=True,
                save_weights_only=False,
                verbose=1
            ),
            
            # Early stopping
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=patience,
                restore_best_weights=True,
                verbose=1
            ),
            
            # Reduce learning rate on plateau
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=6,
                min_lr=1e-7,
                verbose=1
            ),
            
            # Memory cleanup
            keras.callbacks.LambdaCallback(
                on_epoch_end=lambda epoch, logs: tf.keras.backend.clear_session()
            )
        ]
        
        return callbacks
    
    def train_model(self,
                   train_dataset: tf.data.Dataset,
                   val_dataset: tf.data.Dataset,
                   epochs: int = 80,
                   callbacks: Optional[list] = None) -> keras.callbacks.History:
        """
        Train the hybrid model with memory-efficient settings.
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            epochs: Number of training epochs
            callbacks: List of Keras callbacks
            
        Returns:
            Training history
        """
        if self.model is None:
            raise ValueError("Model must be built and compiled before training")
        
        if callbacks is None:
            callbacks = self.get_memory_efficient_callbacks()
        
        history = self.model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def predict_realtime(self, video_sequence: np.ndarray) -> Dict[str, Any]:
        """
        Make real-time predictions with performance metrics.
        
        Args:
            video_sequence: Input video sequence (frames, height, width, channels)
            
        Returns:
            Dictionary with predictions and performance metrics
        """
        if self.model is None:
            raise ValueError("Model must be built before prediction")
        
        import time
        
        # Ensure correct input shape
        if len(video_sequence.shape) == 4:
            video_sequence = np.expand_dims(video_sequence, axis=0)
        
        # Measure inference time
        start_time = time.time()
        predictions = self.model.predict(video_sequence, verbose=0)
        inference_time = time.time() - start_time
        
        # Process predictions
        predicted_class = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class])
        
        return {
            'predicted_class': int(predicted_class),
            'confidence': confidence,
            'class_probabilities': predictions[0].tolist(),
            'inference_time_ms': inference_time * 1000,
            'fps_capability': 1.0 / inference_time if inference_time > 0 else float('inf')
        }
    
    def get_model_summary(self) -> str:
        """
        Get detailed model summary including parameter count and memory usage.
        
        Returns:
            Model summary string
        """
        if self.model is None:
            return "Model not built yet"
        
        # Calculate approximate memory usage
        total_params = self.model.count_params()
        trainable_params = sum([tf.keras.backend.count_params(w) for w in self.model.trainable_weights])
        
        # Estimate memory usage (rough calculation)
        param_memory = total_params * 4  # 4 bytes per float32 parameter
        if self.use_mixed_precision:
            param_memory = total_params * 2  # 2 bytes per float16 parameter
        
        summary = f"""
Hybrid CNN-LSTM Model Summary:
=============================
Total Parameters: {total_params:,}
Trainable Parameters: {trainable_params:,}
Estimated Memory Usage: ~{param_memory / (1024**2):.1f} MB
Mixed Precision: {self.use_mixed_precision}
Spatial Feature Dim: {self.spatial_feature_dim}
LSTM Units: {self.lstm_units}
Fusion Dim: {self.fusion_dim}
Trainable Backbone: {self.trainable_backbone}
Input Shape: {self.input_shape}
Output Classes: {self.num_classes}

Architecture Details:
{self.model.summary()}
        """
        
        return summary
    
    def save_model(self, filepath: str) -> None:
        """Save the complete model to file."""
        if self.model is None:
            raise ValueError("Model must be built before saving")
        self.model.save(filepath)
    
    def load_model(self, filepath: str) -> None:
        """Load a complete model from file."""
        # Register custom layers
        custom_objects = {
            'SpatialFeatureExtractor': SpatialFeatureExtractor,
            'TemporalFeatureExtractor': TemporalFeatureExtractor,
            'FusionLayer': FusionLayer
        }
        self.model = keras.models.load_model(filepath, custom_objects=custom_objects)


def create_hybrid_cnn_lstm(frame_shape: Tuple[int, int, int] = (224, 224, 3),
                          sequence_length: int = 16,
                          num_classes: int = 5,
                          spatial_feature_dim: int = 128,
                          lstm_units: int = 64,
                          fusion_dim: int = 128,
                          trainable_backbone: bool = False,
                          use_mixed_precision: bool = True) -> HybridCNNLSTM:
    """
    Factory function to create and build a hybrid CNN-LSTM model.
    
    Args:
        frame_shape: Input frame shape (height, width, channels)
        sequence_length: Number of frames in sequence
        num_classes: Number of gait abnormality classes
        spatial_feature_dim: Dimension of spatial features
        lstm_units: Number of LSTM units
        fusion_dim: Dimension of fused features
        trainable_backbone: Whether to fine-tune MobileNetV2
        use_mixed_precision: Enable FP16 mixed precision training
        
    Returns:
        Built and compiled HybridCNNLSTM model
    """
    model = HybridCNNLSTM(
        frame_shape=frame_shape,
        sequence_length=sequence_length,
        num_classes=num_classes,
        spatial_feature_dim=spatial_feature_dim,
        lstm_units=lstm_units,
        fusion_dim=fusion_dim,
        trainable_backbone=trainable_backbone,
        use_mixed_precision=use_mixed_precision
    )
    
    model.build_model()
    model.compile_model()
    
    return model