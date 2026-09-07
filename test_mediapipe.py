#!/usr/bin/env python3
"""Test MediaPipe face detection on photos"""
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core import base_options as bo
import numpy as np
from pathlib import Path
import time

# Model paths
MODEL_DIR = 'D:/TinyGiant'
face_model = f'{MODEL_DIR}/face_landmarker.task'
pose_model = f'{MODEL_DIR}/pose_landmarker_lite.task'

# Create detectors
face_landmarker = vision.FaceLandmarker.create_from_options(
    vision.FaceLandmarkerOptions(
        base_options=bo.BaseOptions(model_asset_path=face_model),
        running_mode=vision.RunningMode.IMAGE,
        output_face_blendshapes=True,
        num_faces=5))

pose_landmarker = vision.PoseLandmarker.create_from_options(
    vision.PoseLandmarkerOptions(
        base_options=bo.BaseOptions(model_asset_path=pose_model),
        running_mode=vision.RunningMode.IMAGE))

# Test photos
photos = sorted(Path('D:/0_camera/26.7.25上海zzzkfc/10160725').glob('*.HIF'))[:10]
print(f'Testing {len(photos)} photos\n')

times = []
for f in photos:
    t0 = time.time()
    image = mp.Image.create_from_file(str(f))

    # Face detection + landmarks
    face_result = face_landmarker.detect(image)
    n_faces = len(face_result.face_landmarks)

    eye_closed = None
    face_score = 0.0
    if n_faces > 0:
        landmarks = face_result.face_landmarks[0]
        # Left eye EAR
        le = [landmarks[i] for i in [33, 160, 158, 133, 153, 144]]
        ear_left = (abs(le[1].y - le[5].y) + abs(le[2].y - le[4].y)) / (2 * abs(le[0].x - le[3].x)) if abs(le[0].x - le[3].x) > 0 else 0
        # Right eye EAR
        re = [landmarks[i] for i in [362, 385, 387, 263, 373, 380]]
        ear_right = (abs(re[1].y - re[5].y) + abs(re[2].y - re[4].y)) / (2 * abs(re[0].x - re[3].x)) if abs(re[0].x - re[3].x) > 0 else 0
        ear = (ear_left + ear_right) / 2
        eye_closed = ear < 0.2
        # Score: 1 face = 1.0, 0 faces = 0.0
        face_score = min(n_faces, 3) / 3.0
        # Bonus for open eyes
        if not eye_closed:
            face_score += 0.3

    # Pose detection
    pose_result = pose_landmarker.detect(image)
    n_poses = len(pose_result.pose_landmarks)

    t = time.time() - t0
    times.append(t)
    status = 'OPEN' if eye_closed == False else ('CLOSED' if eye_closed else 'NO_FACE')
    print(f'{f.name}: faces={n_faces} eyes={status} poses={n_poses} face_score={face_score:.2f} ({t:.2f}s)')

avg = sum(times) / len(times)
print(f'\nAverage: {avg:.2f}s per photo')
print('Done')