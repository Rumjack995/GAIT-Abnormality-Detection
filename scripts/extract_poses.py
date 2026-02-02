"""
Extract Pose Landmarks using MediaPipe for Gait Analysis.

This script processes all videos in the dataset and extracts 33-point 3D skeletons.
It saves the raw landmarks which will be used for biomechanical feature calculation.
"""

import os
import sys
import cv2
import numpy as np
import mediapipe as mp
import json
from pathlib import Path
from tqdm import tqdm
import warnings

# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class PoseExtractor:
    def __init__(self):
        try:
            self.mp_pose = mp.solutions.pose
        except AttributeError:
            import mediapipe.python.solutions.pose as mp_pose
            self.mp_pose = mp_pose

        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,  # 0=Lite, 1=Full, 2=Heavy. 1 is good balance.
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
    
    def process_video(self, video_path, max_frames=150):
        """Extract landmarks from video."""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return None
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # We want to sample frames to get a consistent sequence length
        # For gait, 3-5 seconds is usually enough (90-150 frames)
        # Scan every Nth frame to cover the whole video if it's long
        if total_frames > max_frames:
            step = total_frames // max_frames
        else:
            step = 1
            
        landmarks_sequence = []
        frame_idx = 0
        processed_count = 0
        
        while cap.isOpened() and processed_count < max_frames:
            success, image = cap.read()
            if not success:
                break
            
            if frame_idx % step == 0:
                # Convert BGR to RGB
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Process
                try:
                    results = self.pose.process(image_rgb)
                    
                    if results.pose_landmarks:
                        # Extract 33 landmarks (x, y, z, visibility)
                        frame_landmarks = []
                        for lm in results.pose_landmarks.landmark:
                            frame_landmarks.extend([lm.x, lm.y, lm.z, lm.visibility])
                        landmarks_sequence.append(frame_landmarks)
                    else:
                        # No pose detected in this frame
                        # If we have previous frames, repeat the last one (fill gap)
                        if landmarks_sequence:
                            landmarks_sequence.append(landmarks_sequence[-1])
                        else:
                            # Start with zero padding if no pose yet
                            landmarks_sequence.append([0.0] * (33 * 4))
                            
                    processed_count += 1
                except Exception as e:
                    pass
            
            frame_idx += 1
            
        cap.release()
        
        if len(landmarks_sequence) < 30: # Need at least ~1 second of data
            return None
            
        # Pad sequence to max_frames
        landmarks_array = np.array(landmarks_sequence, dtype=np.float32)
        
        if len(landmarks_array) < max_frames:
            padding = np.zeros((max_frames - len(landmarks_array), 33 * 4), dtype=np.float32)
            landmarks_array = np.vstack([landmarks_array, padding])
        
        return landmarks_array


def main():
    print("\n" + "=" * 60)
    print("MEDIAPIPE POSE EXTRACTION")
    print("=" * 60)
    
    data_dir = project_root / "data" / "raw"
    output_dir = project_root / "data" / "pose_data"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    categories = ['normal', 'parkinsonian', 'hemiplegic', 'ataxic', 'other_abnormal']
    
    extractor = PoseExtractor()
    
    video_count = 0
    extracted_count = 0
    
    X_data = []
    y_data = []
    video_names = []
    
    for category in categories:
        cat_dir = data_dir / category
        if not cat_dir.exists():
            continue
        
        videos = list(cat_dir.glob("*.mp4")) + list(cat_dir.glob("*.webm"))
        
        # Processing all available videos
        # Imbalance will be handled by Class Weights in training
        videos = videos
        
        print(f"\nProcessing {category}: {len(videos)} videos")
        
        for video_path in tqdm(videos):
            video_count += 1
            
            # Check if we already processed this specific video (optional caching)
            # But here we'll process all to build the combined dataset
            
            landmarks = extractor.process_video(video_path)
            
            if landmarks is not None:
                X_data.append(landmarks)
                y_data.append(category)
                video_names.append(video_path.name)
                extracted_count += 1
    
    # Save dataset
    if extracted_count > 0:
        X = np.array(X_data, dtype=np.float32)
        y = np.array(y_data)
        
        output_file = project_root / "data" / "gait_pose_dataset.npz"
        np.savez(output_file, X=X, y=y, video_names=video_names)
        
        print("\n" + "=" * 60)
        print("EXTRACTION COMPLETE")
        print("=" * 60)
        print(f"Processed: {video_count} videos")
        print(f"Successfully extracted: {extracted_count} samples")
        print(f"Dataset saved to: {output_file}")
        print(f"Shape: {X.shape}")
        
    else:
        print("\n[ERROR] No pose data extracted!")

if __name__ == "__main__":
    main()
