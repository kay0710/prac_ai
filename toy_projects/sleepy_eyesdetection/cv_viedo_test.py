import cv2

cap = cv2.VideoCapture('/home/user/test_kay/test_sleepy_eyesdetection/mask_video.mp4')

if not cap.isOpened():
    print("Error! Could not open video")
    exit()
    
while cap.isOpened():
    ret, img_ori = cap.read()
    
    if not ret:
        break

cap.release()
cv2.destroyAllWindows()
