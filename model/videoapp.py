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
loaded_model = load_model("model.h5", custom_objects={'custom_f1_score': custom_f1_score})

# Define the list of classes
classes = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Load the face cascade to get the cascade by capturing the face
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

class CustomerSatisfactionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Customer Satisfaction Recognition")
        self.root.geometry("1200x800")  # Set window size

        # Load background image
        self.bg_image = Image.open("smileys-5776137_1280.jpg")
        self.bg_image = self.bg_image.resize((1200, 800), Image.LANCZOS)
        self.bg_image = ImageTk.PhotoImage(self.bg_image)

        self.background_label = Label(root, image=self.bg_image)
        self.background_label.place(x=0, y=0, relwidth=1, relheight=1)

        # Load company logo
        self.logo_image = Image.open("moodmirror-01.png")
        self.logo_image = self.logo_image.resize((150, 150), Image.LANCZOS)
        self.logo_image = ImageTk.PhotoImage(self.logo_image)

        self.logo_label = Label(root, image=self.logo_image, bg="white")
        self.logo_label.place(x=10, y=10)

        title_label = Label(root, text="Customer Satisfaction Recognition", font=("Helvetica", 24, "bold"), fg="black", bg="white")
        title_label.place(x=180, y=30)

        self.video_label = Label(root, bg="white")
        self.video_label.place(x=10, y=180, width=580, height=400)

        self.emotion_labels = {}
        for idx, emotion in enumerate(classes):
            label = Label(root, text=f"{emotion}: 0", font=("Helvetica", 16), fg="black", bg="white")
            label.place(x=620, y=180 + idx*40)
            self.emotion_labels[emotion] = label

        self.start_button = Button(root, text="Start", command=self.start_video, font=("Helvetica", 16), bg="#4CAF50", fg="white", padx=20, pady=10)
        self.start_button.place(x=620, y=480, width=120)

        self.stop_button = Button(root, text="Stop", command=self.stop_video, font=("Helvetica", 16), bg="#f44336", fg="white", padx=20, pady=10)
        self.stop_button.place(x=760, y=480, width=120)

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
