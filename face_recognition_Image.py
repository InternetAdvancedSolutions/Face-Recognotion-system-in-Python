# importingthe libraries
import os
import cv2
import numpy as np



def sample():
   #  sample Image Variable -----------------
   image = cv2.imread('image.jpg')

   # function call to predict the image
   image_label, predicted_image = predict(image)
   return image_label, predicted_image


# function to detect the face in the image
def detection( img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ## face classifier used here ----------------- is lbpcascade
    face_cascade = cv2.CascadeClassifier('lbpcascade_frontalface.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5);
    if (len(faces) == 0):
        return None, None
    (x, y, w, h) = faces[0]
    return gray[y:y+w, x:x+h], faces[0]

### function to draw rectangle on image
### according to given (x, y) coordinates and
### given width and heigh

def draw_rectangle(img, rect):
 (x, y, w, h) = rect
 cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)


image_name_list = []
# training the model
def prepare_training_data(data_folder_path):
    #dirs = os.listdir(data_folder_path)
    faces = []
    labels = []
    i = 0
    subject_images_names = os.listdir(data_folder_path)
    for image_name in subject_images_names:
            image_path = data_folder_path + "/" + image_name
            image = cv2.imread(image_path)
            image_resize=cv2.resize(image, (400,500))
            cv2.imshow("Training on images...", image_resize)
            cv2.waitKey(20)
            face, rect = detection(image)
            if face is not None:
                faces.append(face)
                image_name_list.append(image_path)
                labels.append(i)
                i = i+1

    return faces, labels


# function to predict the new face
def predict(test_img):

    img = test_img.copy()
    face, rect = detection(img)
    label, confidence = face_recognizer.predict(face)
    label_text = image_name_list[label]
    draw_rectangle(img, rect)
    return label_text, img


#  preparing the data
faces, labels = prepare_training_data("data_folder_path")


# For face recognition we are using the LBPH Face Recognizer
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.train(faces, np.array(labels))


##  --------  function call to predict image -----------------------------------
image_label_text, predicted_image = sample()
cv2.imshow('IMAGE', predicted_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
