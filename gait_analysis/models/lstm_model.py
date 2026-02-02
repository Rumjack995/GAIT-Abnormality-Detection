"""
Efficient LSTM architecture optimized for sequential gait pattern recognition.

This module implements a memory-optimized bidirectional LSTM with attention mechanism
for gait abnormality detection, designed for RTX 4050 constraints.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from typing import Tuple, Optional, Dict, Any
import numpy as np


class AttentionLayer(layers.Layer):
    """
    Lightweight self-attention mechanism for LSTM outputs.
    
    This attention layer helps the model focus on the most important
    time steps in the gait sequence for classification.
    """
    
    def __init__(self, attention_dim: int = 64, **kwargs):
        """
        Initialize attention layer.
        
        Args:
            attention_dim: Dimension of attention mechanism
        """
        super(AttentionLayer, self).__init__(**kwargs)
        self.attention_dim = attention_dim
        
    def build(self, input_shape):
        """Build attention weights."""
        # Attention weights
        self.W = self.add_weight(
            name='attention_weight',
            shape=(input_shape[-1], self.attention_dim),
            initializer='glorot_uniform',
            trainable=True
        )
        self.b = self.add_weight(
            name='attention_bias',
            shape=(self.attention_dim,),
            initializer='zeros',
            trainable=True
        )
        self.u = self.add_weight(
            name='attention_context',
            shape=(self.attention_dim,),
            initializer='glorot_uniform',
            trainable=True
        )
        super(AttentionLayer, self).build(input_shape)
    
    def call(self, inputs, mask=None):
        """
        Apply attention mechanism.
        
        Args:
            inputs: LSTM outputs (batch_size, sequence_length, features)
            mask: Optional mask for padded sequences
            
        Returns:
            Attention-weighted output and attention weights
        """
        # Calculate attention scores
        uit = tf.tanh(tf.tensordot(inputs, self.W, axes=1) + self.b)
        ait = tf.tensordot(uit, self.u, axes=1)
        
        # Apply mask if provided
        if mask is not None:
            ait = tf.where(mask, ait, tf.float32.min)
        
        # Softmax attention weights
        attention_weights = tf.nn.softmax(ait, axis=1)
        
        # Apply attention weights
        attention_weights = tf.expand_dims(attention_weights, axis=-1)
        weighted_input = inputs * attention_weights
        output = tf.reduce_sum(weighted_input, axis=1)
        
        return output, attention_weights
    
    def get_config(self):
        """Get layer configuration."""
        config = super(AttentionLayer, self).get_config()
        config.update({'attention_dim': self.attention_dim})
        return config


class EfficientLSTM:
    """
    Efficient LSTM architecture optimized for gait pattern recognition.
    
    Features:
    - Bidirectional LSTM for temporal modeling
    - Lightweight attention mechanism
    - Memory-efficient batch processing
    - Optimized for 6GB VRAM constraint
    """
    
    def __init__(self,
                 input_shape: Tuple[int, int] = (30, 66),  # (sequence_length, features)
                 num_classes: int = 5,
                 lstm_units: int = 64,
                 attention_dim: int = 64,
                 use_mixed_precision: bool = True):
        """
        Initialize the LSTM model.
        
        Args:
            input_shape: Input shape (sequence_length, feature_dim)
                        feature_dim = 33 keypoints * 2D coordinates = 66
            num_classes: Number of gait abnormality classes
            lstm_units: Number of LSTM units (reduced for memory efficiency)
            attention_dim: Dimension of attention mechanism
            use_mixed_precision: Enable FP16 mixed precision training
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.lstm_units = lstm_units
        self.attention_dim = attention_dim
        self.use_mixed_precision = use_mixed_precision
        self.model = None
        
        if use_mixed_precision:
            # Enable mixed precision for memory efficiency
            policy = keras.mixed_precision.Policy('mixed_float16')
            keras.mixed_precision.set_global_policy(policy)
    
    def build_model(self) -> Model:
        """
        Build the efficient LSTM architecture.
        
        Architecture optimized for RTX 4050:
        - Input: (batch_size=4, sequence_length=30, feature_dim=66)
        - Bidirectional LSTM: 64 units (reduced from 128)
        - Attention mechanism: Lightweight self-attention
        - Memory usage: ~2GB VRAM
        
        Returns:
            Compiled Keras model
        """
        inputs = keras.Input(shape=self.input_shape, name='pose_sequence_input')
        
        # Input normalization
        x = layers.LayerNormalization(name='input_norm')(inputs)
        
        # First Bidirectional LSTM layer
        lstm_out = layers.Bidirectional(
            layers.LSTM(
                self.lstm_units,
                return_sequences=True,
                dropout=0.2,
                recurrent_dropout=0.2,
                name='lstm_1'
            ),
            name='bidirectional_lstm'
        )(x)
        
        # Batch normalization for stability
        lstm_out = layers.BatchNormalization(name='lstm_bn')(lstm_out)
        
        # Attention mechanism
        attention_layer = AttentionLayer(
            attention_dim=self.attention_dim,
            name='attention'
        )
        attended_output, attention_weights = attention_layer(lstm_out)
        
        # Dense layers for classification
        x = layers.Dense(128, activation='relu', name='dense_1')(attended_output)
        x = layers.Dropout(0.3, name='dropout_1')(x)
        
        x = layers.Dense(64, activation='relu', name='dense_2')(x)
        x = layers.Dropout(0.3, name='dropout_2')(x)
        
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
            name='efficient_lstm'
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
    
    def create_memory_efficient_dataset(self,
                                      X: np.ndarray,
                                      y: np.ndarray,
                                      batch_size: int = 4,
                                      shuffle: bool = True) -> tf.data.Dataset:
        """
        Create memory-efficient dataset with optimized batching.
        
        Args:
            X: Input sequences (samples, sequence_length, features)
            y: Labels (samples, num_classes)
            batch_size: Batch size optimized for memory
            shuffle: Whether to shuffle the dataset
            
        Returns:
            TensorFlow dataset optimized for memory usage
        """
        dataset = tf.data.Dataset.from_tensor_slices((X, y))
        
        if shuffle:
            # Use buffer size smaller than dataset for memory efficiency
            buffer_size = min(1000, len(X))
            dataset = dataset.shuffle(buffer_size)
        
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset
    
    def get_memory_efficient_callbacks(self,
                                     model_checkpoint_path: str = 'best_lstm_model.h5',
                                     patience: int = 15) -> list:
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
            
            # Early stopping with longer patience for LSTM
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
                patience=7,
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
                   epochs: int = 100,
                   callbacks: Optional[list] = None) -> keras.callbacks.History:
        """
        Train the LSTM model with memory-efficient settings.
        
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
    
    def predict_with_attention(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions and return attention weights for interpretability.
        
        Args:
            X: Input sequences (samples, sequence_length, features)
            
        Returns:
            Tuple of (predictions, attention_weights)
        """
        if self.model is None:
            raise ValueError("Model must be built before prediction")
        
        # Create a model that outputs both predictions and attention weights
        attention_layer = None
        for layer in self.model.layers:
            if isinstance(layer, AttentionLayer):
                attention_layer = layer
                break
        
        if attention_layer is None:
            # Fallback to regular prediction
            predictions = self.model.predict(X)
            return predictions, None
        
        # Get intermediate outputs including attention
        intermediate_model = Model(
            inputs=self.model.input,
            outputs=[self.model.output, attention_layer.output[1]]
        )
        
        predictions, attention_weights = intermediate_model.predict(X)
        return predictions, attention_weights
    
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
Efficient LSTM Model Summary:
============================
Total Parameters: {total_params:,}
Trainable Parameters: {trainable_params:,}
Estimated Memory Usage: ~{param_memory / (1024**2):.1f} MB
Mixed Precision: {self.use_mixed_precision}
LSTM Units: {self.lstm_units}
Attention Dimension: {self.attention_dim}
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
        # Register custom attention layer
        custom_objects = {'AttentionLayer': AttentionLayer}
        self.model = keras.models.load_model(filepath, custom_objects=custom_objects)


def create_efficient_lstm(input_shape: Tuple[int, int] = (30, 66),
                         num_classes: int = 5,
                         lstm_units: int = 64,
                         attention_dim: int = 64,
                         use_mixed_precision: bool = True) -> EfficientLSTM:
    """
    Factory function to create and build an efficient LSTM model.
    
    Args:
        input_shape: Input shape (sequence_length, feature_dim)
        num_classes: Number of gait abnormality classes
        lstm_units: Number of LSTM units
        attention_dim: Dimension of attention mechanism
        use_mixed_precision: Enable FP16 mixed precision training
        
    Returns:
        Built and compiled EfficientLSTM model
    """
    model = EfficientLSTM(
        input_shape=input_shape,
        num_classes=num_classes,
        lstm_units=lstm_units,
        attention_dim=attention_dim,
        use_mixed_precision=use_mixed_precision
    )
    
    model.build_model()
    model.compile_model()
    
    return model