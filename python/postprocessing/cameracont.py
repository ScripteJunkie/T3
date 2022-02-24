import cv2
import numpy as np
# from scipy.misc import bytescale

cap = cv2.VideoCapture('/Your/Path/Here.mp4')
# cap = cv2.VideoCapture(0)

first_iter = True
backSub = cv2.createBackgroundSubtractorMOG2()
points = []

# print(np.shape(cap))

while(1):

    _, frame = cap.read()
    if (_ == False):
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        # for i in range(1, len(points)-1):
        #         print(i, points[i])
        #         if ((points[i+1][0] + points[i+1][1]) != 0):
        #             if ((points[i+1][0] + points[i+1][1]) == 0):
        #                 points[i+1][0] = (2 * points[i][0])-points[i-1][0]
        #                 points[i+1][1] = (2 * points[i][1])-points[i-1][1]
        #             if ((points[i-1][0] + points[i-1][1]) == 0):
        #                 points[i-1][0] = (2 * points[i][0])-points[i+1][0]
        #                 points[i-1][1] = (2 * points[i][1])-points[i+1][1]
    else:
        # frame = frame[100:980,200:1720]
        # cv2.imshow("gray", cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))

        fgMask = backSub.apply(frame)    
        cv2.rectangle(frame, (10, 2), (100,20), (255,255,255), -1)
        cv2.putText(frame, str(cap.get(cv2.CAP_PROP_POS_FRAMES)), (15, 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))
        cv2.imshow('FG Mask', fgMask)
        cut = cv2.bitwise_and(frame, frame, mask=fgMask)
        cut_blur = cv2.medianBlur(cut, 5)
        cv2.imshow("Cut", cut_blur)

        hsv = cv2.cvtColor(cut, cv2.COLOR_BGR2HSV)

        # define range of white color in HSV
        # change it according to your need !
        # lower_white = np.array([12, 160, 201], dtype=np.uint8) # test2
        # upper_white = np.array([20, 255, 255], dtype=np.uint8) # test2
        # lower_white = np.array([10,176,0], dtype=np.uint8)
        # upper_white = np.array([72,215,255], dtype=np.uint8)
        lower_white = np.array([12, 150, 200], dtype=np.uint8) # closest
        upper_white = np.array([20, 255, 255], dtype=np.uint8) # closest
        # lower_white = np.array([11, 40, 195], dtype=np.uint8)
        # upper_white = np.array([20, 232, 255], dtype=np.uint8)


        # Threshold the HSV image to get only white colors
        mask = cv2.inRange(hsv, lower_white, upper_white)
        # Bitwise-AND mask and original image
        res = cv2.bitwise_and(frame,frame, mask= mask)

        contours,_ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        #sorting the contour based of area
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        # gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
        # Blur the image to reduce noise
        # img_blur = cv2.medianBlur(gray, 5)
        # Apply hough transform on the image
        # circles = cv2.HoughCircles(img_blur, cv2.HOUGH_GRADIENT, 1, frame.shape[0]/64, param1=200, param2=10, minRadius=5, maxRadius=30)
        # Draw detected circles
        # if circles is not None:
        #     circles = np.uint16(np.around(circles))
        #     for i in circles[0]:
                # Draw outer circle
                # blur = cv2.GaussianBlur(frame, (5, 5), 25)
                # cv2.circle(frame, (i[0], i[1]), i[2], (0, 255, 255), 2)
                # points.append((i[0], i[1]))
                # cv2.imshow('blur',blur)
                # Draw inner circle
                # cv2.circle(frame, (i[0], i[1]), 2, (255, 0, 0), 3)
        
        if contours:
            #if any contours are found we take the biggest contour and get bounding box
            (x_min, y_min, box_width, box_height) = cv2.boundingRect(contours[0])
            #drawing a rectangle around the object with 15 as margin

        if (10 < box_width < 150 and 10 < box_height < 100):
                points.append([int(x_min + (box_width/2)), int(y_min + (box_height/2))])
                cv2.circle(frame, (points[-1][0], points[-1][1]), 20, (100, 150, 255), -1)
                cv2.rectangle(frame, (x_min - 10, y_min - 10),(x_min + box_width + 10, y_min + box_height + 10),(0,255,0), 4)
                print(box_width, box_height)
                # if ((points[0][0] + points[0][1]) == 0):
                #     points[0][0] = (2 * points[-1][0])-points[0][0]
                #     points[0][1] = (2 * points[-1][1])-points[0][1]

        else:
            points.append([0, 0])
                
        if first_iter:
            avg = np.float32(frame)
            first_iter = False

        m = 0
        draw = True
        for i in range(1, len(points)):
            # print(points[i])
            if ((points[i][0] + points[i][1]) != 0):
                # print(points[i][0])
                cv2.circle(frame, (points[i][0], points[i][1]), 2, (0, 0, 255), 2)
                if ((i - m) > 2):
                    # print("current:", i, "previous:", m)
                    draw = False
                if (draw == True):
                    cv2.line(frame, (points[m][0], points[m][1]), (points[i][0], points[i][1]), (0, 0, 100), 2)
                else:
                    # cv2.line(frame, (points[i][0], points[i][1]), (points[i][0], points[i][1]), (0, 0, 100), 2)
                    draw = True
            # if (len(points) > 10):
            #     del points[-1]

                m = i


        # cv2.accumulateWeighted(frame, avg, 0.005)
        # result = cv2.convertScaleAbs(avg)
        # cv2.imshow('avg',result)

        cv2.imshow('res',res)
        cv2.imshow('mask',mask)
        cv2.imshow('frame',frame)

        # fps = cap.get(cv2.CAP_PROP_FPS)
        # print("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))

        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        if key == ord('p'):
            cv2.waitKey(-1) #wait until any key is pressed

cv2.destroyAllWindows()