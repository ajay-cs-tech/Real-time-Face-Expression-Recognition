import cv2
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
import tensorflow as tf

import matplotlib.pyplot as plt
import glob 
import os
import random
%matplotlib inline
import warnings 
warnings.filterwarnings('ignore')


def detect_faces(frame, face_cascade):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    return faces, gray

def create_model():
    model = Sequential()
    
    model.add(Conv2D(32, (3, 3), input_shape=(48, 48, 1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Flatten())
    
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    
    model.add(Dense(7, activation='softmax'))  # Output layer for 7 emotions
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train_model(model):
    train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=30, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
    validation_datagen = ImageDataGenerator(rescale=1./255)
    
    train_generator = train_datagen.flow_from_directory(
        'D:/ml_projects/data/train', target_size=(48, 48), color_mode='grayscale', class_mode='categorical', batch_size=32)
    validation_generator = validation_datagen.flow_from_directory(
        'D:/ml_projects/data/train', target_size=(48, 48), color_mode='grayscale', class_mode='categorical', batch_size=32)
    
    model.fit(train_generator, validation_data=validation_generator, epochs=25)
    model.save('expression_model.h5')

def real_time_expression_detection():
    model = tf.keras.models.load_model('expression_model.h5')
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        faces, gray = detect_faces(frame, face_cascade)
        
        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            roi_gray = cv2.resize(roi_gray, (48, 48))
            roi_gray = roi_gray / 255.0
            roi_gray = np.reshape(roi_gray, (1, 48, 48, 1))
            
      
            emotion_prediction = model.predict(roi_gray)
            max_index = np.argmax(emotion_prediction)
            emotion_label = emotions[max_index]
            
            
            cv2.putText(frame, emotion_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        
        cv2.imshow('Real-time Face Expression Detection', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
 
    model = create_model()
    print("Training the model...")
    train_model(model)
    
    print("Training completed. Starting real-time expression detection...")
    real_time_expression_detection()


#plot loss
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.title("Model Loss")
plt.legend(["Train","Validation"],loc = "upper left")
plt.subplots_adjust(top=1.0,bottom=0.0,right=0.95,hspace=0.25,wspace=0.35)

#plot accuracy
plt.plot(history.history["accuracy"])
plt.plot(history.history["val_accuracy"])
plt.title("Model Accuracy")
plt.legend(["Train","Validation"],loc = "upper left")
plt.subplots_adjust(top=1.0,bottom=0.0,right=0.95,hspace=0.25,wspace=0.35)


#save the model
model_json = model.to_json()
with open("model_a.json","w") as json_file:
    json_file.write(model_json)