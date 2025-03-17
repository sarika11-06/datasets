import os
import pickle
import mediapipe as mp
import cv2
import numpy as np

# Initialize MediaPipe Pose Model
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.3)

DATA_DIR = './data'  # Path to dataset directory

data = []  # Stores posture landmarks
labels = []  # Stores class labels (folder names as integers)

for dir_ in os.listdir(DATA_DIR):
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
        data_aux = []
        x_, y_ = [], []

        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
        
        if img is None:
            print(f"Error loading image: {img_path}")
            continue
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = pose.process(img_rgb)
        
        if results.pose_landmarks:
            print(f"✅ Full-body detected in {img_path}")
            
            for i, landmark in enumerate(results.pose_landmarks.landmark):
                x = landmark.x
                y = landmark.y
                x_.append(x)
                y_.append(y)

            # Normalize keypoints relative to the bounding box
            for i, landmark in enumerate(results.pose_landmarks.landmark):
                x = landmark.x - min(x_)
                y = landmark.y - min(y_)
                data_aux.append(x)
                data_aux.append(y)

            data.append(data_aux)
            labels.append(int(dir_))  # Convert folder names to class labels

        else:
            print(f"❌ No full-body detected in {img_path}")

# Save dataset
with open('bharatanatyam_data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)

print("✅ Bharatanatyam posture dataset creation complete!")
