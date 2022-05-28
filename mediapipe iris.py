import cv2
import mediapipe as mp
import time
import numpy as np

cap = cv2.VideoCapture(0)
pTime = 0 
Draw = mp.solutions.drawing_utils
FaceMesh = mp.solutions.face_mesh
faceMesh =FaceMesh.FaceMesh(max_num_faces=1)
drawSpec = Draw.DrawingSpec(thickness=1,circle_radius=1)

RIGHT_EYE=[ 33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161 , 246 ] 
LEFT_EYE =[ 362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385,384, 398 ]

LEFT_IRIS=[474,475,476,477]
RIGHT_IRIS=[469,470,471,472]

while True:
    success, img = cap.read()
    if not success:
        break
    key =cv2.waitKey(1)
    if key == ord('e'):
        cv2.destroyAllWindows()
        break
    
    
    img = cv2.flip(img,1)
    img_h,img_w = img.shape[:2]
    
    
    imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    results = faceMesh.process(imgRGB)
    if results.multi_face_landmarks:
        for faceLms in results.multi_face_landmarks:
            #Draw.draw_landmarks(img,faceLms,FaceMesh.FACEMESH_CONTOURS,drawSpec,drawSpec)
            mesh_points=np.array([np.multiply([p.x,p.y],[img_w,img_h]).astype(int) for p in results.multi_face_landmarks[0].landmark])
            
            # cv2.polylines(img,[mesh_points[LEFT_EYE]],True,(0,255,0),1,cv2.LINE_AA)
            # cv2.polylines(img,[mesh_points[RIGHT_EYE]],True,(0,255,0),1,cv2.LINE_AA)
            
            (l_cx,l_cy), l_radius = cv2.minEnclosingCircle(mesh_points[LEFT_IRIS])
            (r_cx,r_cy), r_radius = cv2.minEnclosingCircle(mesh_points[RIGHT_IRIS])
            center_left = np.array([l_cx,l_cy], dtype=np.int64)
            center_right = np.array([r_cx,r_cy], dtype=np.int64)
            cv2.circle(img, center_left, int(l_radius), (0,255,0),1,cv2.LINE_AA)
            cv2.circle(img, center_right, int(r_radius), (0,255,0),1,cv2.LINE_AA)
            
        for id,lm in enumerate(faceLms.landmark):
             #print (lm)
             ih,iw,ic= img.shape
             x,y = int(lm.x*iw), int(lm.y*ih)
             if id == 10:
                 print (id,x,y)
             
    cTime =  time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
  
    cv2.putText(img,str(int(fps)),(10,70),cv2.FONT_HERSHEY_PLAIN,3,(31, 147, 224),2)
    cv2.imshow("image",img)
    cv2.waitKey(1)
    
    
    