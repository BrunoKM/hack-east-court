# Stream Video with OpenCV from an Android running IP Webcam (https://play.google.com/store/apps/details?id=com.pas.webcam)
# Code Adopted from http://stackoverflow.com/questions/21702477/how-to-parse-mjpeg-http-stream-from-ip-camera

import numpy as np
import cv2
import time

cap = cv2.VideoCapture(0)
img_count = 0

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    # laplacian = cv2.Laplacian(frame, cv2.CV_64F)
    # edges = cv2.Canny(frame, 100, 150)
    # sobely = cv2.Sobel(frame, cv2.CV_64F, 0, 1, ksize=5)

    # laplacian_avg = laplacian.copy()
    # for past_img in last:
    #     laplacian_avg += past_img
    # laplacian_avg /= 5
    # Display the resulting frame
    cv2.rectangle(frame, (100, 450), (250, 570), (0, 255, 0), 3)
    # print(frame.dtype)
    # tot_brightness = frame[:, :, 2].astype(np.float32) + frame[:, :, 1].astype(np.float32) + frame[:, :, 0].astype(np.float32)
    # ratio = frame[:, :, 2].astype(np.float32) / (tot_brightness + 1e-10)
    # is_red = ratio > 0.4
    # ratio = (is_red.astype(np.int32) * 255).astype(np.uint8)

    # frame =np.where(np.tile(np.expand_dims(is_red, 2), reps=[1, 1, 3]), frame, 0)
    cv2.imshow('frame', frame)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break
    if chr(cv2.waitKey()) == 'p':
        print('picture')
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        crop = gray_frame[450:570, 100:250]
        cv2.imwrite(f'chip_cnn/data/green-5/img_num{img_count}.png', crop)
        img_count += 1
        frame[:, :, :] = 255
        cv2.imshow('frame', frame)
        time.sleep(0.3)
    else:
        pass

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()