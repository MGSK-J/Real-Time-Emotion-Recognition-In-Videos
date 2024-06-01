import numpy as np
import matplotlib.pyplot as plt
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
loaded_model = load_model("model.h5", custom_objects={'custom_f1_score': custom_f1_score})

# Define the list of classes
classes = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

def predict_emotion(image_path):
    # Load and preprocess the image
    img = image.load_img(image_path, target_size=(48, 48), color_mode='rgb')
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize pixel values

    # Predict the emotion
    result = loaded_model.predict(img_array)
    emotion_index = np.argmax(result[0])
    emotion = classes[emotion_index]

    return emotion

# Example usage
if __name__ == "__main__":
    image_path = input("Enter the path to the image file: ")
    try:
        emotion = predict_emotion(image_path)
        print("Predicted facial emotion:", emotion)
    except Exception as e:
        print("Error:", e)
