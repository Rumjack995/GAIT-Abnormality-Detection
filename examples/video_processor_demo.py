"""Demo script for VideoProcessor functionality."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from gait_analysis.video_processing import VideoProcessor
import numpy as np
import cv2
import tempfile


def create_demo_video():
    """Create a demo video for testing."""
    temp_file = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
    temp_file.close()
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_file.name, fourcc, 30, (640, 480))
    
    # Generate 180 frames (6 seconds at 30 FPS)
    for i in range(180):
        # Create a simple animated frame
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Moving circle to simulate motion
        center_x = int(320 + 200 * np.sin(i * 0.1))
        center_y = 240
        cv2.circle(frame, (center_x, center_y), 30, (0, 255, 0), -1)
        
        # Add some noise to simulate low quality
        noise = np.random.randint(0, 50, frame.shape, dtype=np.uint8)
        frame = cv2.add(frame, noise)
        
        out.write(frame)
    
    out.release()
    return temp_file.name


def main():
    """Run the demo."""
    print("VideoProcessor Demo")
    print("==================")
    
    # Create processor
    processor = VideoProcessor()
    
    # Create demo video
    print("\n1. Creating demo video...")
    video_path = create_demo_video()
    print(f"Demo video created: {video_path}")
    
    try:
        # Validate video
        print("\n2. Validating video...")
        validation_result = processor.validate_video(video_path)
        print(f"Valid: {validation_result.is_valid}")
        print(f"Resolution: {validation_result.resolution}")
        print(f"Duration: {validation_result.duration:.2f}s")
        print(f"Format: {validation_result.format}")
        
        if validation_result.is_valid:
            # Get detailed info
            print("\n3. Getting video info...")
            info = processor.get_video_info(video_path)
            print(f"FPS: {info['fps']}")
            print(f"Frame count: {info['frame_count']}")
            print(f"Estimated extracted frames: {info['estimated_extracted_frames']}")
            
            # Extract frames
            print("\n4. Extracting frames...")
            frames = processor.extract_frames(video_path, target_fps=15)
            print(f"Extracted {len(frames)} frames")
            print(f"Frame shape: {frames[0].shape}")
            print(f"Frame dtype: {frames[0].dtype}")
            
            # Test enhancement
            print("\n5. Testing quality enhancement...")
            enhanced_frames = processor.enhance_quality(frames[:5])  # Just first 5 frames
            print(f"Enhanced {len(enhanced_frames)} frames")
            
            # Test extraction with enhancement
            print("\n6. Testing extraction with enhancement...")
            enhanced_extracted = processor.extract_frames_with_enhancement(
                video_path, target_fps=10, enhance_quality=True
            )
            print(f"Extracted and enhanced {len(enhanced_extracted)} frames")
            
        else:
            print(f"Video validation failed: {validation_result.error_message}")
    
    finally:
        # Clean up
        os.unlink(video_path)
        print(f"\nDemo video cleaned up: {video_path}")
    
    print("\nDemo completed successfully!")


if __name__ == "__main__":
    main()