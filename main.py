import os
import face_recognition
from PIL import Image
import numpy
import cv2

face_match_tol = 0.4

# Rotate Image, Enhance Image, Find Face, Crop ROI
for fileName in os.listdir('Passports'):
    image = face_recognition.load_image_file(os.path.join('Passports', fileName))

    while True:
        face_location = face_recognition.face_locations(image)

        if len(face_location) == 0:
            # Rotate Image
            image = numpy.asarray(Image.fromarray(image).rotate(90, expand=True))
            # Rotate Image
        else:
            # Enhance Image
            image_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            h, s, v = image_hsv[:, :, 0], image_hsv[:, :, 1], image_hsv[:, :, 2]
            clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
            v = clahe.apply(v)
            image_hsv = numpy.dstack((h, s, v))
            image = cv2.cvtColor(image_hsv, cv2.COLOR_HSV2RGB)
            # Enhance Image

            # Find Face
            top, right, bottom, left = face_location[0]
            face_image = image[top:bottom, left:right]
            Image.fromarray(face_image).save(os.path.join('Faces', fileName))
            # Find Face

            # Crop ROI
            image = numpy.asarray(Image.fromarray(image).resize((962, 1343)))
            w, h, r = image.shape
            h = h // 2
            image = image[h + 150:, :w]
            Image.fromarray(image).save(os.path.join('Cropped', fileName))
            # Crop ROI

            break
# Rotate Image, Enhance Image, Find Face, Crop ROI

# Compare Passport Face With System Saved Face
for fileName in os.listdir('Cropped'):
    image = face_recognition.load_image_file(os.path.join('Cropped', fileName))
    face_enc = face_recognition.face_encodings(image)[0]

    match_count = 0

    for fn in os.listdir('Faces'):
        face_image = face_recognition.load_image_file(os.path.join('Faces', fn))
        fe = face_recognition.face_encodings(face_image)[0]

        matches = face_recognition.compare_faces([face_enc], fe, face_match_tol)

        if True in matches:
            print(fileName + ' matched with ' + fn)
            match_count += 1

    if match_count == 0:
        raise Exception('No matches found.')

    if match_count > 1:
        raise Exception('Multiple matches found.')
# Compare Passport Face With System Saved Face


