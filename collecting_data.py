import cv2
import os
import pickle





 # Harcascade file face rcognition
#smile_detector=cv2.CascadeClassifier('cascades/data/haarcascade_smile.xml')

face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#use of recognizer
recognizer=cv2.face.LBPHFaceRecognizer_create()

#open the recoginzer in read mode
recognizer.read('trainer.yml')
# ulabels open with pickle and load into dictionary
labels={'person_name':1}
with open('labels.pickle','rb') as f:
    og_labels=pickle.load(f)
    labels={v:k for k,v in og_labels.items()}
    


#start the capuring of images
print("\n [INFO] Initializing face capture. Look the camera and wait ...")

cap = cv2.VideoCapture(0)
while(True):

    print('capturing')
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)
    #print('capturing')
    for (x,y,w,h) in faces:
        #print(x,y,w,h)
        roi_gray=gray[y:y+h,x:x+h]
        roi_color=img[y:y+h,x:x+w]

        id_,conf=recognizer.predict(roi_gray)
        if conf>=45:#and conf<=85:
            print(id_)
            print(labels[id_])

        #     #Text labeling on the image
        font=cv2.FONT_HERSHEY_SIMPLEX
        name=labels[id_]
        color=(255,0,0)
        stroke=2
        cv2.putText(img,name,(x,y),font,1,color,stroke,cv2.LINE_AA)
        img_item='my-image.jpg'
        cv2.imwrite(img_item,roi_gray)
        color=(255,0,0)
        stroke=2
        end_cord_x=x+w
        end_cord_y=y+h
        cv2.rectangle(img,(x,y),(end_cord_x,end_cord_y),color,stroke)
        # smile=smile_detector.detectMultiScale(roi_gray)
        # for (ex,ey,ew,eh) in smile:
        #     cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),color,stroke)
        #     cv2.putText(roi_color,'smiled',(x,y),font,1,color,stroke,cv2.LINE_AA)

        #cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)     
        #cv2.imwrite("images." + str(face_id) + '.' + str(count) + ".jpg", gray[y:y+h,x:x+w])

        cv2.imshow('image', img)

    k = cv2.waitKey(100) & 0xff # Press 'ESC' for exiting video
    if k == 27:
        break

# Do a bit of cleanup

cap.release()
cv2.destroyAllWindows()