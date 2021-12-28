# Passport Immigration Automation
This software has been used for automate the passport immigration process. Python programming language has been used in this software. This software first rotate the passport image into correct direction then enhance the image for better quality, crop ROI and then extract face from passport and save it into folder and then compare saved faces with passport faces file and verify live cemera photo with passport extracted faces.

## Features
### Rotate The Passport File Into Correct Direction
![](https://i.ibb.co/9qyFZPb/rotate.png)
```python
for fileName in os.listdir('Passports'):
    image = face_recognition.load_image_file(os.path.join('Passports', fileName))

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
 image = numpy.asarray(Image.fromarray(image).resize((962, 1343)))
 w, h, r = image.shape
 h = h // 2
 image = image[h + 150:, :w]
```
### Enhance Image
![](https://i.ibb.co/hXW3SYh/a.png)
```python
 image_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
 h, s, v = image_hsv[:, :, 0], image_hsv[:, :, 1], image_hsv[:, :, 2]
 clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
 v = clahe.apply(v)
 image_hsv = numpy.dstack((h, s, v))
 image = cv2.cvtColor(image_hsv, cv2.COLOR_HSV2RGB)
```
### Extract Face From Passport
![](https://i.ibb.co/QPgkj0t/rotate.png)
```python
 top, right, bottom, left = face_location[0]
 face_image = image[top:bottom, left:right]
```
### Compare Passport Face
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
### Verify Live Cemera Photo
```python
video_capture = cv2.VideoCapture(0)
while True:
    ret, frame = video_capture.read()

    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = small_frame[:, :, ::-1]

    if cv2.waitKey(1) & 0xFF == ord('c'):
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)[0]

        match_count = 0

        for fn in os.listdir('Faces'):
            face_image = face_recognition.load_image_file(os.path.join('Faces', fn))
            fe = face_recognition.face_encodings(face_image)[0]

            matches = face_recognition.compare_faces([face_encodings], fe, 0.4)

            if True in matches:
                print('Matched with ' + fn)
                match_count += 1

        if match_count == 0:
            raise Exception('No matches found.')

        if match_count > 1:
            raise Exception('Multiple matches found.')

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
```
## Installation
### Requirements
  * Python
  * Anaconda
  * PyCharm
### Installation Options
#### Installing on Windows
  * First make sure you have already installed Python, Anaconda, and PyCharm
  * Install Pillow, Numpy, OpenCV module from pypi using `pip`

```bash
pip install Pillow
```
```bash
pip install numpy
```
```bash
pip install opencv-python
```
  * Install Face Recognition module from powershell using `conda`
```bash
conda create -n test_env python anaconda
```
```bash
conda activate test_env
```
```bash
conda install -c conda-forge face_recognition
```
  * Setting Up Conda Environment in PyCharm
    *   Open pycharm and open cloned project and then in the top left menu click on **File** and then go to **Settings** 
    
        ![](https://i.ibb.co/Lt9GPyd/1.jpg)
    *   On the left side of menu expand project and then go to **Python Interpreter** and then on top right side click on **Add**
    
        ![](https://i.ibb.co/cx5L7GD/3.jpg)
    *   On the left side menu click on **Conda Environmen** and then click on existing environment and then add **Interpreter** location and click ok
    
        ![](https://i.ibb.co/ftsQTXK/2.jpg)
