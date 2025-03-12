# Hand Gesture Recognition using Mediapipe & Machine Learning

This project captures hand gestures using OpenCV, extracts features using Mediapipe, and classifies gestures using a Machine Learning model. The trained model can recognize different hand gestures in real-time.

## Features
✅ Capture hand gesture images
✅ Process images to extract hand landmarks
✅ Train a gesture classification model using Random Forest
✅ Perform real-time gesture recognition via a FastAPI backend
✅ Integrate with a React frontend for live predictions

---

## 1️⃣ Installation
### **Clone the Repository**
```bash
git clone https://github.com/sarika11-06/datasets.git
cd datasets
```

### **Install Dependencies**
Make sure you have Python installed (preferably Python 3.8+).
```bash
pip install fastapi uvicorn opencv-python numpy mediapipe scikit-learn pillow
```

---

## 2️⃣ Data Collection
### **Run the Image Collection Script**
This script captures images from your webcam and saves them in a `data/` directory.
```bash
python collect_imgs.py
```
Press `Q` to start collecting images for each class.

---

## 3️⃣ Prepare the Dataset
Once images are collected, run the following script to process them and create a dataset.
```bash
python create_dataset.py
```
This will generate a `data.pickle` file containing the extracted hand landmarks and labels.

---

## 4️⃣ Train the Classifier
Train a Random Forest classifier on the collected dataset.
```bash
python train_classifier.py
```
After training, a model file `model.p` will be created.

---

## 5️⃣ Run the FastAPI Server
Start the FastAPI server to perform real-time hand gesture recognition.
```bash
python app.py
```
The server will start at `http://localhost:8000`. You can test it via:
```bash
http://localhost:8000/docs
```

---

## 6️⃣ Real-Time Inference (Live Gesture Detection)
To perform live inference, run the script below:
```bash
python inference_classifier.py
```
Your webcam will open, and the system will classify gestures in real time.

---

## 7️⃣ Folder Structure
```
├── data/                  # Collected images
├── app.py                 # FastAPI backend
├── collect_imgs.py        # Image collection script
├── create_dataset.py      # Convert images to dataset
├── train_classifier.py    # Train the ML model
├── inference_classifier.py # Live gesture recognition
├── model.p                # Trained ML model
├── data.pickle            # Processed dataset
├── README.md              # Project documentation
```

---

## 🔥 Future Enhancements
- ✅ Train a deep learning model (CNN) for better accuracy
- ✅ Add more gestures
- ✅ Improve React UI with real-time visualization
  
---

🎉 **Enjoy building your hand gesture recognition system! 🚀**

