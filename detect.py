import cv2
import os

classifier = cv2.CascadeClassifier('lbpcascade_animeface.xml')

michi=63
nora=0
pato=88
shachi=53
asu=92
other=240

msec=0
output_dir = 'F:/faces/'

cap=cv2.VideoCapture("D:/python/kurokiFinder/noratoto/07.mp4")
while(cap.isOpened()):
    cap.set(0,msec*1000)
    ret, frame = cap.read()
    if ret:
        gray_image = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        faces = classifier.detectMultiScale(gray_image)
        for i, (x,y,w,h) in enumerate(faces):
            print(str(msec)+"[sec] 0:黒木 1:ノラ 2:パト 3:シャチ 4:明日原 5:その他 No.=",end="")
            cv2.imshow("FRAME",frame)
            face_image = frame[y:y+h, x:x+w]
            face_image_resize=cv2.resize(face_image,(150,150))
            cv2.imshow("DETECT",face_image_resize)
            flg=cv2.waitKey(0)
            print(flg)
            if flg==48:#0
    	        output_path = output_dir+'michi/'+'{0}.jpg'.format(michi)
    	        print("save "+output_path)
    	        cv2.imwrite(output_path,face_image_resize)
    	        michi+=1
            elif flg==49:#1
    	        output_path = output_dir+'nora/'+'{0}.jpg'.format(nora)
    	        print("save "+output_path)
    	        cv2.imwrite(output_path,face_image_resize)
    	        nora+=1
            elif flg==50:#2
    	        output_path = output_dir+'pato/'+'{0}.jpg'.format(pato)
    	        print("save "+output_path)
    	        cv2.imwrite(output_path,face_image_resize)
    	        pato+=1
            elif flg==51:#3
    	        output_path = output_dir+'shachi/'+'{0}.jpg'.format(shachi)
    	        print("save "+output_path)
    	        cv2.imwrite(output_path,face_image_resize)
    	        shachi+=1
            elif flg==52:#4
    	        output_path = output_dir+'asu/'+'{0}.jpg'.format(asu)
    	        print("save "+output_path)
    	        cv2.imwrite(output_path,face_image_resize)
    	        asu+=1
            elif flg==53:#5
    	        output_path = output_dir+'other/'+'{0}.jpg'.format(other)
    	        print("save "+output_path)
    	        cv2.imwrite(output_path,face_image_resize)
    	        other+=1
            elif flg==113:#q
                exit(-1)
        msec+=1
    else:
        break
