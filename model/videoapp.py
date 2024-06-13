import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
import tkinter as tk
from tkinter import Label, Button, Frame
from PIL import Image, ImageTk

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

class CustomerSatisfactionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Customer Satisfaction Recognition")
        
        self.root.configure(bg="#2e2e2e")

        title_label = Label(root, text="Customer Satisfaction Recognition", font=("Helvetica", 20, "bold"), fg="white", bg="#2e2e2e")
        title_label.pack(pady=10)

        self.video_frame = Frame(root, bg="#2e2e2e")
        self.video_frame.pack()

        self.video_label = Label(self.video_frame, bg="#2e2e2e")
        self.video_label.grid(row=0, column=0, padx=10, pady=10)

        self.emotion_frame = Frame(root, bg="#2e2e2e")
        self.emotion_frame.pack(pady=10)

        self.emotion_labels = {}
        for idx, emotion in enumerate(classes):
            label = Label(self.emotion_frame, text=f"{emotion}: 0", font=("Helvetica", 14), fg="white", bg="#2e2e2e")
            label.grid(row=idx // 2, column=idx % 2, padx=10, pady=5, sticky="w")
            self.emotion_labels[emotion] = label

        button_frame = Frame(root, bg="#2e2e2e")
        button_frame.pack(pady=10)

        self.start_button = Button(button_frame, text="Start", command=self.start_video, font=("Helvetica", 14), bg="#4CAF50", fg="white", padx=20, pady=5)
        self.start_button.grid(row=0, column=0, padx=10)

        self.stop_button = Button(button_frame, text="Stop", command=self.stop_video, font=("Helvetica", 14), bg="#f44336", fg="white", padx=20, pady=5)
        self.stop_button.grid(row=0, column=1, padx=10)

        self.cap = None
        self.running = False

    def start_video(self):
        self.cap = cv2.VideoCapture(0)
        self.running = True
        self.video_stream()

    def stop_video(self):
        self.running = False
        if self.cap is not None:
            self.cap.release()
        self.video_label.config(image='')

    def video_stream(self):
        if self.running:
            ret, frame = self.cap.read()
            if ret:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
                
                emotion_counts = {emotion: 0 for emotion in classes}

                for (x, y, w, h) in faces:
                    face_img = frame[y:y+h, x:x+w]
                    face_img = cv2.resize(face_img, (48, 48))
                    face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                    face_img = np.expand_dims(face_img, axis=0)
                    face_img = face_img / 255.0

                    result = loaded_model.predict(face_img)
                    emotion_index = np.argmax(result[0])
                    emotion = classes[emotion_index]
                    emotion_counts[emotion] += 1

                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                for emotion, count in emotion_counts.items():
                    self.emotion_labels[emotion].config(text=f"{emotion}: {count}")

                cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
                img = Image.fromarray(cv2image)
                imgtk = ImageTk.PhotoImage(image=img)
                self.video_label.imgtk = imgtk
                self.video_label.config(image=imgtk)
            
            self.root.after(10, self.video_stream)

if __name__ == "__main__":
    root = tk.Tk()
    app = CustomerSatisfactionApp(root)
    root.mainloop()
