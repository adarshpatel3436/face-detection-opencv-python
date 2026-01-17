import cv2 
import os
import cv2.data

print(os.listdir(cv2.data.haarcascades))
file_name = 'haarcascade_frontalface_default.xml'
data_path = cv2.data.haarcascades + '/' + file_name
model = cv2.CascadeClassifier(data_path)

cam = cv2.VideoCapture(0)
while True:
    status, frame=cam.read()
    if not status:
        print("Camera not working...")
        break

    faces = model.detectMultiScale(frame, 1.3, 4)
    print(faces)

    for face in faces:
        x1,y1,x2,y2 = face[0], face[1], face[0]+face[2], face[1]+face[3]
        cv2.rectangle(frame, (x1,y1) , (x2,y2), (0,255,255), 4)
    cv2.imshow("face",frame)
    
    if cv2.waitKey(1)==ord("q"):
        break

cam.release()
cv2.destroyAllWindows()
