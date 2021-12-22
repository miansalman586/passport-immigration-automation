# Passport Immigration Automation
This software has been used for automate the passport immigration process. Python programming language has been used in this software. This software first rotate the passport image into correct direction, crop ROI and then extract face from passport and then compare faces with passport file.

## Features
### Rotate the passport file into correct direction
![](https://i.ibb.co/9qyFZPb/rotate.png)
```python
for fileName in os.listdir('Passports'):
    image = face_recognition.load_image_file(os.path.join('Passports', fileName))

    while True:
        face_location = face_recognition.face_locations(image)

        if len(face_location) == 0:
            image = numpy.asarray(Image.fromarray(image).rotate(90, expand=True))
```
### Crop ROI
![](https://i.ibb.co/Qcmj1p8/rotate.png)
```python
 image = numpy.asarray(Image.fromarray(image).resize((962, 1343)))
 w, h, r = image.shape
 h = h // 2
 image = image[h + 150:, :w]
```
### Extract face from passport
![](https://i.ibb.co/QPgkj0t/rotate.png)
```python
top, right, bottom, left = face_location[0]
face_image = image[top:bottom, left:right]
```
