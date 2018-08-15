'''
顔の分類を行う
'''

import cv2
import os

classifier = cv2.CascadeClassifier('lbpcascade_animeface.xml')

asu=64
ida=9
michi=63
nobu=83
other=24
pato=89
ru=42
shachi=53
tanaka=38
yu=37

msec=0
output_dir = 'F:/faces50/'

cap=cv2.VideoCapture("D:/python/noratotoFinder/noratoto/07.mp4")
while(cap.isOpened()):
    cap.set(0,msec*1000)
    ret, frame = cap.read()
    if ret:
        gray_image = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        faces = classifier.detectMultiScale(gray_image)
        for i, (x,y,w,h) in enumerate(faces):
            print(str(msec)+"[sec]")
            print("0:明日原     1:井田    2:黒木   3:ノブチナ 4:その他")
            print("5:パトリシア 6:ルーシア 7:シャチ 8:田中     9:ユーラシア")
            cv2.imshow("FRAME",frame)
            face_image = frame[y:y+h, x:x+w]
            face_image_resize=cv2.resize(face_image,(50,50))
            cv2.imshow("DETECT",face_image_resize)
            flg=cv2.waitKey(0)
            if flg==48:#0
    	        output_path = output_dir+'asu/'+'{0}.jpg'.format(michi)
    	        asu+=1
            elif flg==49:#1
    	        output_path = output_dir+'ida/'+'{0}.jpg'.format(ida)
    	        ida+=1
            elif flg==50:#2
    	        output_path = output_dir+'michi/'+'{0}.jpg'.format(michi)
    	        michi+=1
            elif flg==51:#3
    	        output_path = output_dir+'nobu/'+'{0}.jpg'.format(nobu)
    	        nobu+=1
            elif flg==52:#4
    	        output_path = output_dir+'other/'+'{0}.jpg'.format(other)
    	        other+=1
            elif flg==53:#5
    	        output_path = output_dir+'pato/'+'{0}.jpg'.format(pato)
    	        pato+=1
            elif flg==54:#6
    	        output_path = output_dir+'ru/'+'{0}.jpg'.format(ru)
    	        ru+=1
            elif flg==55:#7
    	        output_path = output_dir+'shachi/'+'{0}.jpg'.format(shachi)
    	        shachi+=1
            elif flg==56:#8
    	        output_path = output_dir+'tanaka/'+'{0}.jpg'.format(tanaka)
    	        tanaka+=1
            elif flg==57:#9
    	        output_path = output_dir+'yu/'+'{0}.jpg'.format(yu)
    	        yu+=1
            elif flg==113:#q
                exit(-1)
            print("save "+output_path)
            cv2.imwrite(output_path,face_image_resize)
        msec+=1
    else:
        break
