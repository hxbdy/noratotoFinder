'''テスト用の顔を保存する'''

import cv2
import os

classifier = cv2.CascadeClassifier('lbpcascade_animeface.xml')

msec=0
cnt=0
output_dir = 'F:/faces150/test/'

cap=cv2.VideoCapture("D:/python/noratotoFinder/noratoto/08.mp4")
while(cap.isOpened()):
    cap.set(0,msec*1000)
    ret, frame = cap.read()
    if ret:
        gray_image = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        faces = classifier.detectMultiScale(gray_image)
        for i, (x,y,w,h) in enumerate(faces):
            print(str(msec)+"[sec]")
            face_image = frame[y:y+h, x:x+w]
            face_image_resize=cv2.resize(face_image,(150,150))
            cv2.imwrite(output_dir+str(cnt)+".jpg",face_image_resize)
            cnt+=1
        msec+=1
    else:
        break
