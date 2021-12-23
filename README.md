# Passport Immigration Automation
This software has been used for automate the passport immigration process. Python programming language has been used in this software. This software first rotate the passport image into correct direction then enhance the image for better quality, crop ROI and then extract face from passport and save it into folder and then compare saved faces with passport faces file.

## Features
### Rotate The Passport File Into Correct Direction
![](https://i.ibb.co/9qyFZPb/rotate.png)
```python
image = face_recognition.load_image_file(os.path.join('Passports', 'Passport-001.jpg'))

while True:
    face_location = face_recognition.face_locations(image)
    
    if len(face_location) == 0:
        image = numpy.asarray(Image.fromarray(image).rotate(90, expand=True))
    else:
        break
```
### Crop ROI
![](https://i.ibb.co/Qcmj1p8/rotate.png)
```python
image = face_recognition.load_image_file(os.path.join('Passports', 'Passport-001.jpg'))

image = numpy.asarray(Image.fromarray(image).resize((962, 1343)))
w, h, r = image.shape
h = h // 2
image = image[h + 150:, :w]
```
### Enhance Image
![](https://i.ibb.co/hXW3SYh/a.png)
```python
image = face_recognition.load_image_file(os.path.join('Passports', 'Passport-001.jpg'))

image_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
h, s, v = image_hsv[:, :, 0], image_hsv[:, :, 1], image_hsv[:, :, 2]
clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
v = clahe.apply(v)
image_hsv = numpy.dstack((h, s, v))
image = cv2.cvtColor(image_hsv, cv2.COLOR_HSV2RGB)
```
### Extract face from passport
![](https://i.ibb.co/QPgkj0t/rotate.png)
```python
for fileName in os.listdir('Passports'):
    image = face_recognition.load_image_file(os.path.join('Passports', fileName))

    face_location = face_recognition.face_locations(image)

    top, right, bottom, left = face_location[0]
    face_image = image[top:bottom, left:right]
```
### Compare passport face
![](https://i.ibb.co/Y8TTmMm/rotate.png)
```python
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
```
