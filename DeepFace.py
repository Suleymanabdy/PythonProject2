import os
import cv2
import numpy as np
import streamlit as st
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.models import Model
from keras_facenet import FaceNet
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from PIL import Image


# **1. Load HAAR Cascade for Face Detection**
def extract_face(image, output_size=(160, 160)):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    img = np.array(image)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    for (x, y, w, h) in faces:
        face = img[y:y + h, x:x + w]
        face_resized = cv2.resize(face, output_size)
        return face_resized
    return None

# **2. Get FaceNet Embedding Function**
def get_embedding(model, face_image):
    face_image = np.expand_dims(face_image, axis=0)
    embeddings = model.embeddings(face_image)
    return embeddings

# **3. Get VGG16 Embedding Function**
def get_vgg16_embedding(face_image):
    model = VGG16(weights='imagenet', include_top=False, input_shape=(160, 160, 3))
    model = Model(inputs=model.inputs, outputs=model.layers[-1].output)

    face_image = np.expand_dims(face_image, axis=0)
    face_image = preprocess_input(face_image)

    embedding = model.predict(face_image)
    return embedding

# **4. Train SVM Classifier**
def train_svm_classifier(train_embeddings, train_labels):
    train_embeddings_flat = [emb.flatten() for emb in train_embeddings]
    clf = SVC(kernel='linear', probability=True)
    clf.fit(train_embeddings_flat, train_labels)
    return clf

# **Streamlit UI**
st.title("Face Recognition with Streamlit")

# Upload training and test data directories
train_dir = st.text_input("Enter path for the training dataset", "path/to/train")
test_dir = st.text_input("Enter path for the test dataset", "path/to/test")

if st.button("Train Model"):
    # Load your train dataset (You'll want to update this code to handle file uploads as well)
    train_faces, train_labels = load_dataset(train_dir)
    test_faces, test_labels = load_dataset(test_dir)
    
    # Use FaceNet model
    model = FaceNet()

    # Generate Embeddings for Train and Test Faces
    train_embeddings = [get_embedding(model, face) for face in train_faces]
    test_embeddings = [get_embedding(model, face) for face in test_faces]

    # Train SVM Classifier
    clf = train_svm_classifier(train_embeddings, train_labels)

    # Save embeddings (optional)
    np.savez('train_embeddings.npz', embeddings=train_embeddings, labels=train_labels)
    np.savez('test_embeddings.npz', embeddings=test_embeddings, labels=test_labels)

    # Evaluate the model
    test_embeddings_flat = [emb.flatten() for emb in test_embeddings]
    test_predictions = clf.predict(test_embeddings_flat)
    report = classification_report(test_labels, test_predictions)
    st.text(report)

# File uploader for testing
uploaded_file = st.file_uploader("Upload a test image for prediction", type=["jpg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Process the uploaded image
    face = extract_face(image)

    if face is not None:
        # Use FaceNet or VGG16 model to get embeddings
        model = FaceNet()
        embeddings = get_embedding(model, face)

        # Perform prediction (you can use the trained SVM classifier here)
        embeddings_flat = embeddings.flatten().reshape(1, -1)
        prediction = clf.predict(embeddings_flat)
        st.write(f"Predicted Label: {prediction[0]}")
    else:
        st.write("No face detected in the uploaded image.")

