from fastapi import FastAPI, File, UploadFile
import pickle
import cv2
import mediapipe as mp
import numpy as np
import uvicorn
import io
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

# Load trained model
model_dict = pickle.load(open('model.p', 'rb'))
model = model_dict['model']

# Setup FastAPI
app = FastAPI()

# Fix CORS issues
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mediapipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.3)

labels_dict = {0: 'Pataka', 1: 'Tripataka', 2: 'Ardhapataka', 3: 'Katarimukha', 4: 'Mayura', 5: 'Ardhachandra', 6: 'Arala', 7: 'Sukatunda', 8: 'Musti', 9: 'Sikhara', 10: 'Kapittha', 11: 'Katamukha', 12: 'Suchi', 13: 'Chandrakala', 14: 'Padmakosa', 15: 'Sarpashirsa', 16: 'Mrgasirsa', 17: 'Simhamukha', 18: 'Kangula', 19: 'Alapadma', 20: 'Catura', 21: 'Bhramara', 22: 'Hamsasya', 23: 'Hamsapaksa', 24: 'Sandamsa', 25: 'Mukula', 26: 'Tamracuda',27:'Trisula', 28:'Anjali', 29:'Kapota', 30:'Karkata', 31:'Svastika', 32:'Dola', 33:'Puspaputa', 34:'Utsanga', 35:'Sivalinga', 36:'Katakavardhana', 37:'Kartarisvastika', 38:'Sakata', 39:'Sankha', 40:'Chakra', 41:'Pasa', 42:'Kilaka',43:'Matsya', 44:'Kurma', 45:'Varaha', 46:'Garuda', 47:'Nagabandha', 48:'Katva', 49:'Bherunda'}

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read()))
    image = np.array(image)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    data_aux = []
    x_, y_ = [], []

    results = hands.process(image_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y

                x_.append(x)
                y_.append(y)

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

        prediction = model.predict([np.asarray(data_aux)])
        predicted_character = labels_dict[int(prediction[0])]

        return {"gesture": predicted_character}

    return {"error": "No hand detected"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
