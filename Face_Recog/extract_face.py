
import numpy as np
from PIL import Image
from mtcnn.mtcnn import MTCNN
from sklearn.preprocessing import LabelEncoder
import cv2
from imutils import paths
import random
import os


def extract_face(filename, size = (224, 224)):
    img = Image.open(filename)
    img = img.convert('RGB')
    #   Convert image to array
    img = np.asarray(img)
    #   Create a detector by using defaut weights
    detector = MTCNN()
    #   Detect face in image
    result = detector.detect_faces(img)
    #   Extract the bounding box from face
    x, y, width, height = result[0]['box']
    x1, y1 = abs(x), abs(y)
    x2, y2 = x1 + width, y1 + height
    #   Extract the face from co-ordinate of the bounding box
    face = img[y1:y2, x1:x2]
    face = np.asarray(cv2.resize(face, size)).astype('float32')
    #   Normalize face images
    mean, std = np.mean(face), np.std(face)
    face = (face - mean)/std
    return face

if __name__ == "__main__":
    img_path = list(paths.list_images('Datasets\\face-recognition-data\\trainset'))
    random.shuffle(img_path)
    labels_name = [p.split(os.path.sep)[-2] for p in img_path]
    
    #   Create datasets
    X_data = []
    for path in img_path:
        face = extract_face(path)
        X_data.append(face)
        
    #   Generate labels
    lab = LabelEncoder()
    lab.fit(labels_name)
    labels = lab.fit_transform(labels_name)
    
    #   Save dataset
    np.savez("Datasets\\face-recognition-data\\faces_224x224.npz", x = X_data, y = labels)
