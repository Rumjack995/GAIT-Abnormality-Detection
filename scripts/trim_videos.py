"""
trim_videos.py - Trims GAVD dataset videos based on CSV annotations.
Output will be saved to data/trimmed/
"""

import os
import sys
import cv2
import pandas as pd
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

def main():
    root_dir = Path(__file__).parent.parent
    raw_dir = root_dir / 'data' / 'raw'
    csv_path = raw_dir / 'GAVD_Clinical_Annotations_1.csv'
    trimmed_dir = root_dir / 'data' / 'trimmed'
    
    if not csv_path.exists():
        print(f"Error: {csv_path} not found.")
        sys.exit(1)
        
    print(f"Loading CSV annotations from {csv_path.name} ...")
    # low_memory=False to stop pandas from complaining about mixed dtypes in unused cols
    df = pd.read_csv(csv_path, low_memory=False)
    
    # Calculate min and max frame bounds for every video ID
    print("Calculating trim bounds...")
    bounds = df.groupby('id')['frame_num'].agg(['min', 'max']).to_dict('index')
    
    # Iterate through Categories
    categories = ['ataxic', 'hemiplegic', 'normal', 'other_abnormal', 'parkinsonian']
    
    total_videos = 0
    trimmed_count = 0
    skipped_count = 0
    
    for cat in categories:
        cat_path = raw_dir / cat
        out_cat_path = trimmed_dir / cat
        out_cat_path.mkdir(parents=True, exist_ok=True)
        
        if not cat_path.exists():
            continue
            
        videos = list(cat_path.glob("*.mp4")) + list(cat_path.glob("*.avi")) + list(cat_path.glob("*.mkv"))
        
        for v in videos:
            total_videos += 1
            vid_id = v.stem  # e.g. "B5hrxKe2nP8.mp4" -> "B5hrxKe2nP8"
            
            if vid_id not in bounds:
                print(f"  [SKIP] No annotations found for {v.name}")
                skipped_count += 1
                continue
                
            min_frame = int(bounds[vid_id]['min'])
            max_frame = int(bounds[vid_id]['max'])
            
            out_file = out_cat_path / v.name
            
            # Check if already trimmed
            if out_file.exists():
                print(f"  [CACHE] Already trimmed {v.name}")
                continue
                
            print(f"🎬 Trimming {cat}/{v.name} (Frames {min_frame} -> {max_frame})")
            
            cap = cv2.VideoCapture(str(v))
            if not cap.isOpened():
                print(f"  [ERROR] Cannot open {v.name}")
                skipped_count += 1
                continue
                
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = float(cap.get(cv2.CAP_PROP_FPS))
            if fps <= 0: fps = 30.0
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(out_file), fourcc, fps, (w, h))
            
            current_frame = 0
            frames_saved = 0
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Assume 0-indexed or 1-indexed? Usually frame numbers in CSVs are 1-indexed.
                # However, if min_frame is e.g. 500, we skip everything before it.
                if current_frame >= min_frame and current_frame <= max_frame:
                    out.write(frame)
                    frames_saved += 1
                    
                if current_frame > max_frame:
                    break
                    
                current_frame += 1
                
            cap.release()
            out.release()
            
            if frames_saved > 0:
                trimmed_count += 1
                print(f"      -> Saved {frames_saved} frames to {out_file.parent.name}/{out_file.name}")
            else:
                print(f"      -> [WARN] 0 frames saved, video might be shorter than max_frame.")
                if out_file.exists():
                    out_file.unlink() # delete empty file

    print("\n" + "="*50)
    print("TRIMMING COMPLETE")
    print("="*50)
    print(f"Total videos processed: {total_videos}")
    print(f"Successfully trimmed:   {trimmed_count}")
    print(f"Skipped/Empty:          {skipped_count}")
    print("\nOutput saved to: data/trimmed/")
    print("If you are happy with the trimmed versions, you can rename 'trimmed' to 'raw'.")

if __name__ == "__main__":
    main()
