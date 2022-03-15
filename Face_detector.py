import cv2
from random import randrange
# from PIL import Image      

#load some pre-trained data on face frontals from opencv (haar cascade algorithm)
trained_face_data = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

#choose an image to detect faces
img = cv2.imread("1.jpg")

#must convert it to grayscale
# grayscaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#detect faces
face_coordinates = trained_face_data.detectMultiScale(img) #use grayscaled_img if converted the image

# draw rectangles around faces
# (x , y , w , h) = face_coordinates[0] #(for a single face detection)
for (x,y,w,h) in face_coordinates:
    cv2.rectangle(img, (x,y),(x+w,y+h),(randrange(256),randrange(256),randrange(256)),1) 
    #can directly use digits instead of randrange
    
# print(face_coordinates)


# Display the image with faces
cv2.imshow("Image 1",img)
cv2.waitKey()