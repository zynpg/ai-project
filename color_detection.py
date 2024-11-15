import cv2
import numpy as np
from collections import deque

buffer_size = 16
pts = deque(maxlen = buffer_size) # points

# mavi renk uzayı hsv -> renk uzayı
blueLower = (84, 98, 0)
blueUpper = (179, 255, 255)

cap = cv2.VideoCapture(0)
cap.set(3, 960)
cap.set(4, 480)

while True:
    success, imgOriginal = cap.read()

    if success:
        # bulanıklaştırma
        blurred = cv2.GaussianBlur(imgOriginal, (11,11), 0)

        # hsv
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        cv2.imshow("HSV", hsv)

        # maske
        mask = cv2.inRange(hsv, blueLower, blueUpper)
        cv2.imshow("MASKE", mask)

        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)

        # kontur
        (contours, _) = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 0:
            c = max(contours, key=cv2.contourArea)
            rect = cv2.minAreaRect(c)

            # kutucuk
            box = cv2.boxPoints(rect)
            box = np.int64(box)

            # moment
            M = cv2.moments(c)
            center = (
                int(M["m10"]/M["m00"]),
                int(M["m01"]/M["m00"])
            )

            # kontoru çiz
            cv2.drawContours(imgOriginal, [box], 0, (0,255,255), 2)

            # circle
            cv2.circle(imgOriginal, center, 5, (255, 0, 255), -1)

            # iz ekleme
            pts.appendleft(center)
            for i in range(1, len(pts)):
                if pts[i-1] is None or pts[i] is None:
                     continue # devam etsin
                cv2.line(imgOriginal,  pts[i-1], pts[i], (90,90,90), 3)

            # tespit
            cv2.imshow("Tespit", imgOriginal)

    # tuş
    if cv2.waitKey(1) & 0xFF == ord('q'):
            break



