import os
import cv2
import numpy as np
from keras.applications.vgg16 import VGG16, preprocess_input
from keras_facenet import FaceNet
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import streamlit as st


# **1. Load HAAR Cascade for Face Detection**
def extract_face(image_path, output_size=(160, 160)):
    face_cascade = cv2.CascadeClassifier('C:\ Users\ User\PycharmProjects\PythonProject2\haarcascade\haarcascade_frontalface_default.xml')

    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    for (x, y, w, h) in faces:
        face = img[y:y + h, x:x + w]
        face_resized = cv2.resize(face, output_size)
        return face_resized
    return None

#2.Load Faces from Directory
def load_faces(directory, output_size=(160, 160)):
    faces = []
    for filename in os.listdir(directory):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            image_path = os.path.join(directory, filename)
            face = extract_face(image_path, output_size)
            if face is not None:
                faces.append(face)
    return faces

#3.Load Dataset from Parent Directory
def load_dataset(parent_directory, output_size=(160, 160)):
    dataset = []
    labels = []
    for label, person_dir in enumerate(os.listdir(parent_directory)):
        person_dir_path = os.path.join(parent_directory, person_dir)
        if os.path.isdir(person_dir_path):
            faces = load_faces(person_dir_path, output_size)
            dataset.extend(faces)
            labels.extend([label] * len(faces))
    return dataset, labels

#5.Get FaceNet Embedding Function
def get_embedding(model, face_image):
    face_image = np.expand_dims(face_image, axis=0)
    embeddings = model.embeddings(face_image)
    return embeddings


# 6.Get VGG16 Embedding Function
def get_vgg16_embedding(face_image):
    model = VGG16(weights='imagenet', include_top=False, input_shape=(160, 160, 3))
    model = Model(inputs=model.inputs, outputs=model.layers[-1].output)

    face_image = np.expand_dims(face_image, axis=0)
    face_image = preprocess_input(face_image)

    embedding = model.predict(face_image)
    return embedding

#7.Load Train and Test Datasets
train_directory = 'C:\ Users\ User\PycharmProjects\PythonProject2\dataset\Train'
test_directory = 'C:\ Users\ User\PycharmProjects\PythonProject2\dataset\Test'

train_faces, train_labels = load_dataset(train_directory)
test_faces, test_labels = load_dataset(test_directory)

#8.Generate Embeddings for Train and Test Faces
#Use FaceNet or VGG16 here
model = FaceNet()  # Use FaceNet model, or switch to VGG16 embedding
train_embeddings = [get_embedding(model, face) for face in train_faces]
test_embeddings = [get_embedding(model, face) for face in test_faces]

#9.Save Generated Embeddings
np.savez('train_embeddings.npz', embeddings=train_embeddings, labels=train_labels)
np.savez('test_embeddings.npz', embeddings=test_embeddings, labels=test_labels)

#10.Train Classifier (SVM in this case)
#Flatten the embeddings to pass to the classifier
train_embeddings_flat = [emb.flatten() for emb in train_embeddings]
test_embeddings_flat = [emb.flatten() for emb in test_embeddings]

# Train an SVM classifier
clf = SVC(kernel='linear', probability=True)
clf.fit(train_embeddings_flat, train_labels)

#Predict and evaluate the model
test_predictions = clf.predict(test_embeddings_flat)
print(classification_report(test_labels, test_predictions))
