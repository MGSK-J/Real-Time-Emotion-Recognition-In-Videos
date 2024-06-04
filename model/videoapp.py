import cv2
import numpy as np

from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K

# Define the custom metric function
def custom_f1_score(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2 * (precision * recall) / (precision + recall + K.epsilon())
    return f1_val

# Load the saved model
loaded_model = load_model("model.h5", custom_objects={'customized_f1_score': custom_f1_score})

# Define the list of classes
classes = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Load the face cascade to get the cascade by capturing the face
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Open the webcam
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    # Process each face detected
    for (x, y, w, h) in faces:
        # Crop the face region from the frame
        face_img = frame[y:y+h, x:x+w]
        
        # Preprocess the face image
        face_img = cv2.resize(face_img, (48, 48))
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        face_img = np.expand_dims(face_img, axis=0)
        face_img = face_img / 255.0
        
        # Predict the emotion
        result = loaded_model.predict(face_img)
        emotion_index = np.argmax(result[0])
        emotion = classes[emotion_index]
        
        # Draw a rectangle around the face and display the predicted emotion
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
    
    # Display the annotated frame
    cv2.imshow('Facial Emotion Recognition', frame)
    
    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows

cap.release()
cv2.destroyAllWindows()

# simple demonstration of opening video cam and output string
