from facenet_pytorch import MTCNN
from PIL import Image
import cv2
import os
from pathlib import Path
from tqdm import tqdm
import torch

# Paths
video_dir = Path("videos4")
output_dir = Path("datasets_faces_4")
output_dir.mkdir(parents=True, exist_ok=True)

# Face detector
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(image_size=160, margin=20, keep_all=False, device=device)

# Loop through videos2
video_files = list(video_dir.glob("*.*"))
for video_path in tqdm(video_files, desc="Processing videos"):
    person_name = video_path.stem
    person_dir = output_dir / person_name
    person_dir.mkdir(exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_idx = 0
    saved_count = 0

    with tqdm(total=total_frames, desc=f"{person_name}", leave=False) as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % 5 == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(rgb)

                # Use MTCNN to directly save cropped image (no dark image problem)
                save_path = str(person_dir / f"{saved_count:04d}.jpg")
                try:
                    mtcnn(img, save_path=save_path)
                    saved_count += 1
                except Exception:
                    pass

            frame_idx += 1
            pbar.update(1)

    cap.release()
    print(f"✅ Finished {person_name}: {saved_count} faces saved.")
