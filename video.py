import cv2
import numpy as np
from pandas import DataFrame
# Load the cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Create a VideoCapture object and read from input file
# If the input is the camera, pass 0 instead of the video file name
cap = cv2.VideoCapture('mall_video.mp4')
# Check if camera opened successfully
if (cap.isOpened() == False):
    print("Error opening video stream or file")
name = 1
# Read until video is completed
while(cap.isOpened()):
    # Capture frame-by-frame
    for i in range(25):
        ret, img = cap.read()
    if ret == True:
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Detect the faces
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        # Draw the rectangle around each face
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            croped_image = img[y:y+h, x:x+w]
            print(croped_image.shape)
            if(croped_image.shape[0] and croped_image.shape[1] and croped_image.shape[2]):
                cv2.imshow('Frame', croped_image)
                cv2.imwrite(f'annotated_images/{name}.jpg', croped_image)
                name += 1
        # Display the resulting frame
        # cv2.imshow('Frame', img)
        # cv2.imwrite(f'annotated_images/{name}.jpg', img)
        # name += 1
        # Press Q on keyboard to  exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    # Break the loop
    else:
        break

# When everything done, release the video capture object
cap.release()

# Closes all the frames
cv2.destroyAllWindows()
