"""Video processing and validation for gait analysis."""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
import logging

from ..utils.data_structures import ValidationResult

logger = logging.getLogger(__name__)


class VideoProcessor:
    """Handles video input validation, preprocessing, and frame extraction."""
    
    # Supported video formats
    SUPPORTED_FORMATS = {'.mp4', '.avi', '.mov'}
    
    # Quality thresholds
    MIN_RESOLUTION = (480, 270)  # 270p minimum (lowered for YouTube videos)
    MIN_DURATION = 5.0  # seconds
    MAX_DURATION = 120.0  # 2 minutes
    TARGET_FPS = 30
    
    def __init__(self):
        """Initialize VideoProcessor."""
        self.logger = logging.getLogger(__name__)
    
    def validate_video(self, video_path: str) -> ValidationResult:
        """
        Validate video file format, resolution, and duration.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            ValidationResult with validation status and details
        """
        try:
            video_path = Path(video_path)
            
            # Check if file exists
            if not video_path.exists():
                return ValidationResult(
                    is_valid=False,
                    resolution=(0, 0),
                    duration=0.0,
                    format="",
                    error_message=f"Video file not found: {video_path}"
                )
            
            # Check file format
            file_extension = video_path.suffix.lower()
            if file_extension not in self.SUPPORTED_FORMATS:
                return ValidationResult(
                    is_valid=False,
                    resolution=(0, 0),
                    duration=0.0,
                    format=file_extension,
                    error_message=f"Unsupported video format: {file_extension}. "
                                f"Supported formats: {', '.join(self.SUPPORTED_FORMATS)}"
                )
            
            # Open video and get properties
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                return ValidationResult(
                    is_valid=False,
                    resolution=(0, 0),
                    duration=0.0,
                    format=file_extension,
                    error_message="Failed to open video file. File may be corrupted."
                )
            
            try:
                # Get video properties
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                
                resolution = (width, height)
                duration = frame_count / fps if fps > 0 else 0.0
                
                # Validate resolution
                if width < self.MIN_RESOLUTION[0] or height < self.MIN_RESOLUTION[1]:
                    return ValidationResult(
                        is_valid=False,
                        resolution=resolution,
                        duration=duration,
                        format=file_extension,
                        error_message=f"Resolution too low: {width}x{height}. "
                                    f"Minimum required: {self.MIN_RESOLUTION[0]}x{self.MIN_RESOLUTION[1]} (480p)"
                    )
                
                # Validate duration
                if duration < self.MIN_DURATION:
                    return ValidationResult(
                        is_valid=False,
                        resolution=resolution,
                        duration=duration,
                        format=file_extension,
                        error_message=f"Video too short: {duration:.1f}s. "
                                    f"Minimum duration: {self.MIN_DURATION}s"
                    )
                
                if duration > self.MAX_DURATION:
                    return ValidationResult(
                        is_valid=False,
                        resolution=resolution,
                        duration=duration,
                        format=file_extension,
                        error_message=f"Video too long: {duration:.1f}s. "
                                    f"Maximum duration: {self.MAX_DURATION}s (2 minutes)"
                    )
                
                # All validations passed
                return ValidationResult(
                    is_valid=True,
                    resolution=resolution,
                    duration=duration,
                    format=file_extension,
                    error_message=None
                )
                
            finally:
                cap.release()
                
        except Exception as e:
            self.logger.error(f"Error validating video {video_path}: {str(e)}")
            return ValidationResult(
                is_valid=False,
                resolution=(0, 0),
                duration=0.0,
                format="",
                error_message=f"Validation error: {str(e)}"
            )
    
    def extract_frames(self, video_path: str, target_fps: Optional[float] = None) -> List[np.ndarray]:
        """
        Extract frames from video at consistent intervals.
        
        Args:
            video_path: Path to the video file
            target_fps: Target FPS for frame extraction (default: TARGET_FPS)
            
        Returns:
            List of extracted frames as numpy arrays
            
        Raises:
            ValueError: If video validation fails
            RuntimeError: If frame extraction fails
        """
        if target_fps is None:
            target_fps = self.TARGET_FPS
        
        # Validate video first
        validation_result = self.validate_video(video_path)
        if not validation_result.is_valid:
            raise ValueError(f"Video validation failed: {validation_result.error_message}")
        
        frames = []
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video file: {video_path}")
        
        try:
            original_fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Calculate frame interval for consistent extraction
            if original_fps <= target_fps:
                # Extract all frames if original FPS is lower than target
                frame_interval = 1
            else:
                # Skip frames to achieve target FPS
                frame_interval = int(original_fps / target_fps)
            
            frame_idx = 0
            extracted_count = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Extract frame at consistent intervals
                if frame_idx % frame_interval == 0:
                    # Convert BGR to RGB (OpenCV uses BGR by default)
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame_rgb)
                    extracted_count += 1
                
                frame_idx += 1
            
            self.logger.info(f"Extracted {extracted_count} frames from {total_frames} total frames "
                           f"(interval: {frame_interval}, target FPS: {target_fps})")
            
            if not frames:
                raise RuntimeError("No frames could be extracted from the video")
            
            return frames
            
        except Exception as e:
            self.logger.error(f"Error extracting frames from {video_path}: {str(e)}")
            raise RuntimeError(f"Frame extraction failed: {str(e)}")
        
        finally:
            cap.release()
    
    def enhance_quality(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """
        Enhance video frames for low-quality inputs.
        
        Applies preprocessing including:
        - Frame normalization
        - Lighting correction
        - Noise reduction
        - Contrast enhancement
        
        Args:
            frames: List of frames as numpy arrays
            
        Returns:
            List of enhanced frames
        """
        if not frames:
            return frames
        
        enhanced_frames = []
        
        for frame in frames:
            try:
                enhanced_frame = self._enhance_single_frame(frame)
                enhanced_frames.append(enhanced_frame)
            except Exception as e:
                self.logger.warning(f"Failed to enhance frame, using original: {str(e)}")
                enhanced_frames.append(frame)
        
        self.logger.info(f"Enhanced {len(enhanced_frames)} frames")
        return enhanced_frames
    
    def _enhance_single_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Enhance a single frame.
        
        Args:
            frame: Input frame as numpy array (RGB format)
            
        Returns:
            Enhanced frame
        """
        # Convert to float for processing
        frame_float = frame.astype(np.float32) / 255.0
        
        # 1. Normalize frame dimensions if needed
        frame_normalized = self._normalize_frame_dimensions(frame_float)
        
        # 2. Apply lighting correction
        frame_corrected = self._correct_lighting(frame_normalized)
        
        # 3. Enhance contrast
        frame_enhanced = self._enhance_contrast(frame_corrected)
        
        # 4. Reduce noise
        frame_denoised = self._reduce_noise(frame_enhanced)
        
        # Convert back to uint8
        frame_final = np.clip(frame_denoised * 255.0, 0, 255).astype(np.uint8)
        
        return frame_final
    
    def _normalize_frame_dimensions(self, frame: np.ndarray) -> np.ndarray:
        """
        Normalize frame dimensions and aspect ratio.
        
        Args:
            frame: Input frame
            
        Returns:
            Normalized frame
        """
        height, width = frame.shape[:2]
        
        # If frame is below minimum resolution, upscale it
        if width < self.MIN_RESOLUTION[0] or height < self.MIN_RESOLUTION[1]:
            # Calculate scale factor to meet minimum resolution
            scale_w = self.MIN_RESOLUTION[0] / width
            scale_h = self.MIN_RESOLUTION[1] / height
            scale = max(scale_w, scale_h)
            
            new_width = int(width * scale)
            new_height = int(height * scale)
            
            # Convert to uint8 for resize, then back to float
            frame_uint8 = (frame * 255).astype(np.uint8)
            frame_resized = cv2.resize(frame_uint8, (new_width, new_height), 
                                     interpolation=cv2.INTER_CUBIC)
            frame = frame_resized.astype(np.float32) / 255.0
            
            self.logger.info(f"Upscaled frame from {width}x{height} to {new_width}x{new_height}")
        
        return frame
    
    def _correct_lighting(self, frame: np.ndarray) -> np.ndarray:
        """
        Correct lighting conditions using histogram equalization.
        
        Args:
            frame: Input frame in RGB format
            
        Returns:
            Lighting-corrected frame
        """
        # Convert RGB to LAB color space for better lighting correction
        frame_uint8 = (frame * 255).astype(np.uint8)
        lab = cv2.cvtColor(frame_uint8, cv2.COLOR_RGB2LAB)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        
        # Convert back to RGB
        corrected = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        return corrected.astype(np.float32) / 255.0
    
    def _enhance_contrast(self, frame: np.ndarray) -> np.ndarray:
        """
        Enhance frame contrast.
        
        Args:
            frame: Input frame
            
        Returns:
            Contrast-enhanced frame
        """
        # Apply gamma correction for contrast enhancement
        gamma = 1.2  # Slightly increase contrast
        frame_gamma = np.power(frame, 1.0 / gamma)
        
        # Apply adaptive contrast stretching
        frame_stretched = self._adaptive_contrast_stretch(frame_gamma)
        
        return frame_stretched
    
    def _adaptive_contrast_stretch(self, frame: np.ndarray) -> np.ndarray:
        """
        Apply adaptive contrast stretching.
        
        Args:
            frame: Input frame
            
        Returns:
            Contrast-stretched frame
        """
        # Calculate percentiles for robust contrast stretching
        p2, p98 = np.percentile(frame, (2, 98))
        
        # Avoid division by zero
        if p98 - p2 > 0.01:
            frame_stretched = (frame - p2) / (p98 - p2)
            frame_stretched = np.clip(frame_stretched, 0, 1)
        else:
            frame_stretched = frame
        
        return frame_stretched
    
    def _reduce_noise(self, frame: np.ndarray) -> np.ndarray:
        """
        Reduce noise in the frame.
        
        Args:
            frame: Input frame
            
        Returns:
            Denoised frame
        """
        # Convert to uint8 for noise reduction
        frame_uint8 = (frame * 255).astype(np.uint8)
        
        # Apply bilateral filter for noise reduction while preserving edges
        denoised = cv2.bilateralFilter(frame_uint8, 9, 75, 75)
        
        return denoised.astype(np.float32) / 255.0
    
    def extract_frames_with_enhancement(self, video_path: str, 
                                      target_fps: Optional[float] = None,
                                      enhance_quality: bool = True) -> List[np.ndarray]:
        """
        Extract frames with optional quality enhancement.
        
        Args:
            video_path: Path to the video file
            target_fps: Target FPS for frame extraction
            enhance_quality: Whether to apply quality enhancement
            
        Returns:
            List of extracted (and optionally enhanced) frames
        """
        # Extract frames normally
        frames = self.extract_frames(video_path, target_fps)
        
        if enhance_quality:
            # Check if enhancement is needed based on video quality
            validation_result = self.validate_video(video_path)
            
            # Apply enhancement if resolution is close to minimum or other quality issues
            width, height = validation_result.resolution
            if (width <= self.MIN_RESOLUTION[0] * 1.2 or 
                height <= self.MIN_RESOLUTION[1] * 1.2):
                
                self.logger.info("Applying quality enhancement due to low resolution")
                frames = self.enhance_quality(frames)
            else:
                self.logger.info("Video quality sufficient, skipping enhancement")
        
        return frames
    
    def get_video_info(self, video_path: str) -> dict:
        """
        Get detailed video information.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Dictionary with video properties
        """
        validation_result = self.validate_video(video_path)
        
        if not validation_result.is_valid:
            return {
                'valid': False,
                'error': validation_result.error_message,
                'resolution': validation_result.resolution,
                'duration': validation_result.duration,
                'format': validation_result.format
            }
        
        cap = cv2.VideoCapture(str(video_path))
        
        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            return {
                'valid': True,
                'resolution': validation_result.resolution,
                'duration': validation_result.duration,
                'format': validation_result.format,
                'fps': fps,
                'frame_count': frame_count,
                'estimated_extracted_frames': int(frame_count * min(1.0, self.TARGET_FPS / fps))
            }
        
        finally:
            cap.release()