import pickle
import cv2
import mediapipe as mp
import numpy as np

# Load trained model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

cap = cv2.VideoCapture(0)

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(min_detection_confidence=0.3)

#hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Bharatanatyam Posture Labels
labels_dict = {0: 'Araimandi', 1: 'Muzhumandi', 2: 'Samapadam', 3: 'Natyarambham', 4: 'Alidha', 5: 'Pratyalidha'}

while True:
    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()
    if not ret:
        break
    
    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame,  # Image to draw
            results.pose_landmarks,  # Model output
            mp_pose.POSE_CONNECTIONS  # Connections between keypoints
            #mp_drawing_styles.get_default_hand_landmarks_style(),
            #mp_drawing_styles.get_default_hand_connections_style())
        )

        for landmark in results.pose_landmarks.landmark:
            x = landmark.x
            y = landmark.y
            x_.append(x)
            y_.append(y)

        # Normalize keypoints relative to bounding box
        for landmark in results.pose_landmarks.landmark:
            x = landmark.x - min(x_)
            y = landmark.y - min(y_)
            data_aux.append(x)
            data_aux.append(y)

        x1 = int(min(x_) * W) - 10
        y1 = int(min(y_) * H) - 10
        x2 = int(max(x_) * W) + 10
        y2 = int(max(y_) * H) + 10

        prediction = model.predict([np.asarray(data_aux)])
        predicted_posture = labels_dict[int(prediction[0])]

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
        cv2.putText(frame, predicted_posture, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)

    cv2.imshow('Bharatanatyam Posture Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


cap.release()
cv2.destroyAllWindows()