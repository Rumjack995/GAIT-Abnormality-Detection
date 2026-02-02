"""Tests for VideoProcessor class."""

import pytest
import numpy as np
import cv2
import tempfile
import os
from pathlib import Path

from gait_analysis.video_processing import VideoProcessor
from gait_analysis.utils.data_structures import ValidationResult


class TestVideoProcessor:
    """Test cases for VideoProcessor."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.processor = VideoProcessor()
    
    def create_test_video(self, width=640, height=480, fps=30, duration=10, format_ext='.mp4'):
        """Create a test video file."""
        temp_file = tempfile.NamedTemporaryFile(suffix=format_ext, delete=False)
        temp_file.close()
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_file.name, fourcc, fps, (width, height))
        
        # Generate frames
        total_frames = int(fps * duration)
        for i in range(total_frames):
            # Create a simple test frame (gradient)
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            frame[:, :, 0] = (i * 255 // total_frames) % 256  # Red channel varies
            frame[:, :, 1] = 128  # Green constant
            frame[:, :, 2] = 255 - (i * 255 // total_frames) % 256  # Blue varies inversely
            out.write(frame)
        
        out.release()
        return temp_file.name
    
    def teardown_method(self):
        """Clean up after tests."""
        # Clean up any temporary files
        pass
    
    def test_validate_video_valid_mp4(self):
        """Test validation of valid MP4 video."""
        video_path = self.create_test_video(width=640, height=480, duration=10, format_ext='.mp4')
        
        try:
            result = self.processor.validate_video(video_path)
            
            assert result.is_valid
            assert result.resolution == (640, 480)
            assert 9.5 <= result.duration <= 10.5  # Allow some tolerance
            assert result.format == '.mp4'
            assert result.error_message is None
        
        finally:
            os.unlink(video_path)
    
    def test_validate_video_invalid_format(self):
        """Test validation with unsupported format."""
        # Create a file with unsupported extension
        temp_file = tempfile.NamedTemporaryFile(suffix='.txt', delete=False)
        temp_file.write(b'not a video')
        temp_file.close()
        
        try:
            result = self.processor.validate_video(temp_file.name)
            
            assert not result.is_valid
            assert '.txt' in result.error_message
            assert 'Unsupported video format' in result.error_message
        
        finally:
            os.unlink(temp_file.name)
    
    def test_validate_video_low_resolution(self):
        """Test validation with resolution below minimum."""
        video_path = self.create_test_video(width=320, height=240, duration=10)
        
        try:
            result = self.processor.validate_video(video_path)
            
            assert not result.is_valid
            assert result.resolution == (320, 240)
            assert 'Resolution too low' in result.error_message
            assert '480p' in result.error_message
        
        finally:
            os.unlink(video_path)
    
    def test_validate_video_short_duration(self):
        """Test validation with duration below minimum."""
        video_path = self.create_test_video(duration=3)  # 3 seconds, below 5s minimum
        
        try:
            result = self.processor.validate_video(video_path)
            
            assert not result.is_valid
            assert result.duration < 5.0
            assert 'Video too short' in result.error_message
        
        finally:
            os.unlink(video_path)
    
    def test_validate_video_long_duration(self):
        """Test validation with duration above maximum."""
        video_path = self.create_test_video(duration=150)  # 150 seconds, above 120s maximum
        
        try:
            result = self.processor.validate_video(video_path)
            
            assert not result.is_valid
            assert result.duration > 120.0
            assert 'Video too long' in result.error_message
        
        finally:
            os.unlink(video_path)
    
    def test_validate_video_nonexistent_file(self):
        """Test validation with non-existent file."""
        result = self.processor.validate_video('/nonexistent/path/video.mp4')
        
        assert not result.is_valid
        assert 'not found' in result.error_message
    
    def test_extract_frames_valid_video(self):
        """Test frame extraction from valid video."""
        video_path = self.create_test_video(width=640, height=480, fps=30, duration=6)
        
        try:
            frames = self.processor.extract_frames(video_path)
            
            assert len(frames) > 0
            assert isinstance(frames[0], np.ndarray)
            assert frames[0].shape == (480, 640, 3)  # Height, Width, Channels
            assert frames[0].dtype == np.uint8
            
            # Should extract approximately target_fps * duration frames
            expected_frames = 30 * 6  # 30 fps * 6 seconds
            assert len(frames) >= expected_frames * 0.8  # Allow some tolerance
        
        finally:
            os.unlink(video_path)
    
    def test_extract_frames_invalid_video(self):
        """Test frame extraction from invalid video."""
        with pytest.raises(ValueError, match="Video validation failed"):
            self.processor.extract_frames('/nonexistent/video.mp4')
    
    def test_extract_frames_with_target_fps(self):
        """Test frame extraction with custom target FPS."""
        video_path = self.create_test_video(width=640, height=480, fps=60, duration=6)
        
        try:
            # Extract at 15 FPS (lower than original 60 FPS)
            frames = self.processor.extract_frames(video_path, target_fps=15)
            
            assert len(frames) > 0
            # Should extract approximately 15 * 6 = 90 frames
            expected_frames = 15 * 6
            assert len(frames) >= expected_frames * 0.8  # Allow tolerance
            assert len(frames) <= expected_frames * 1.2
        
        finally:
            os.unlink(video_path)
    
    def test_enhance_quality_basic(self):
        """Test basic quality enhancement."""
        # Create some test frames
        frames = []
        for i in range(5):
            frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
            frames.append(frame)
        
        enhanced_frames = self.processor.enhance_quality(frames)
        
        assert len(enhanced_frames) == len(frames)
        assert all(isinstance(frame, np.ndarray) for frame in enhanced_frames)
        assert all(frame.shape == (480, 640, 3) for frame in enhanced_frames)
        assert all(frame.dtype == np.uint8 for frame in enhanced_frames)
    
    def test_enhance_quality_empty_list(self):
        """Test enhancement with empty frame list."""
        enhanced_frames = self.processor.enhance_quality([])
        assert enhanced_frames == []
    
    def test_get_video_info_valid(self):
        """Test getting video information for valid video."""
        video_path = self.create_test_video(width=640, height=480, fps=25, duration=8)
        
        try:
            info = self.processor.get_video_info(video_path)
            
            assert info['valid'] is True
            assert info['resolution'] == (640, 480)
            assert 7.5 <= info['duration'] <= 8.5
            assert info['format'] == '.mp4'
            assert info['fps'] == 25.0
            assert 'frame_count' in info
            assert 'estimated_extracted_frames' in info
        
        finally:
            os.unlink(video_path)
    
    def test_get_video_info_invalid(self):
        """Test getting video information for invalid video."""
        info = self.processor.get_video_info('/nonexistent/video.mp4')
        
        assert info['valid'] is False
        assert 'error' in info
        assert 'not found' in info['error']
    
    def test_extract_frames_with_enhancement_low_quality(self):
        """Test frame extraction with enhancement for low-quality video."""
        # Create a low-resolution video that should trigger enhancement
        video_path = self.create_test_video(width=650, height=490, duration=6)  # Just above minimum
        
        try:
            frames = self.processor.extract_frames_with_enhancement(video_path, enhance_quality=True)
            
            assert len(frames) > 0
            assert isinstance(frames[0], np.ndarray)
            assert frames[0].dtype == np.uint8
        
        finally:
            os.unlink(video_path)
    
    def test_extract_frames_with_enhancement_disabled(self):
        """Test frame extraction with enhancement disabled."""
        video_path = self.create_test_video(width=640, height=480, duration=6)
        
        try:
            frames = self.processor.extract_frames_with_enhancement(video_path, enhance_quality=False)
            
            assert len(frames) > 0
            assert isinstance(frames[0], np.ndarray)
        
        finally:
            os.unlink(video_path)