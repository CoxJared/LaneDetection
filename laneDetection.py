#!/usr/local/bin/python3

import cv2
import numpy as np
import matplotlib.pyplot as plt


def getCannyOfImage(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0 )
    canny = cv2.Canny(blur, 70, 100)
    return canny

def getMaskedImage(image, polygon = [(0, 300), (0, 315), (630, 315), (630, 300), (290,200)]):
    polygons= np.array([polygon])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)
    maskedImage= cv2.bitwise_and(image, mask)
    return maskedImage

def createCoordinates(image, lineParameters):
    #Could/should update this to polar coordinates to avoid vertical(horizontal aswell?) errors
    slope, yIntercept = lineParameters
    y1 = image.shape[0]
    y2 = int(y1 * (4/6))
    x1 = int((y1 - yIntercept) / slope)
    x2 = int((y2 - yIntercept) / slope)
    return np.array([x1, y1, x2, y2])

def averageSlopeIntercept(image, lines, currentFit):

    leftFitLines = []
    rightFitLines =[]

    if not lines is None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            parameters = np.polyfit((x1, x2), (y1, y2), 1)
            slope = parameters[0]
            yIntercept = parameters[1]
            if slope < 0:
                leftFitLines.append((slope, yIntercept))
            else:
                rightFitLines.append((slope, yIntercept))

        if leftFitLines == [] :
            #if no lines are found for the left 
            leftLine = currentFit[0]
        else:
            leftAverageFit = np.average(leftFitLines, axis = 0)  
            leftLine = createCoordinates(image, leftAverageFit)

        if rightFitLines == []:
            #if no lines are found for the right
            rightLine = currentFit[1]
        else:
            rightAverageFit= np.average(rightFitLines, axis = 0)
            rightLine = createCoordinates(image, rightAverageFit)

        return np.array([leftLine, rightLine]) 


    else:
        #if no lines are found that are long enough
        return currentFit

def displayLines(image, lines):
    lineImage= np.zeros_like(image)
    if lines is not None:
        for line in lines:
            try:
                x1, y1, x2, y2 = line.reshape(4)
                cv2.line(lineImage, (x1, y1), (x2, y2), (255, 0, 0), 10)  
            except:
                continue

    return lineImage

if __name__ == '__main__':
    print('\nstarting\n')
    print('Options:\n\n0: Original\n1: Canny Image\n2: Masked Image\n3: Canny Image with lines\n4: Original with lines')
    choice = '0'

    print('Switch between these at any point in the video')

    cap = cv2.VideoCapture('Videos/highwayDash.mp4')

    averagedLines = None

    while (cap.isOpened()):
        ret , frame= cap.read()

        #frame does not exist
        if not ret: 
            break

        cannyImage = getCannyOfImage(frame)
        maskedImage = getMaskedImage(cannyImage)
        lines = cv2.HoughLinesP(maskedImage, 2, np.pi / 180, 100, np.array([]), minLineLength = 70, maxLineGap = 5)
        averagedLines = averageSlopeIntercept(frame, lines, averagedLines)
        lineImage = displayLines(frame, averagedLines)
        cannyImageInRGB = cv2.cvtColor(cannyImage, cv2.COLOR_GRAY2RGB)
        cannyComboImage = cv2.addWeighted( cannyImageInRGB, 0.8, lineImage, 1, 1)
        originalComboImage = cv2.addWeighted(frame, 0.8, lineImage, 1, 1)

        if choice == '0':
            cv2.imshow("result", frame)

        elif choice == '1':
            cv2.imshow("result", cannyImage)
        
        elif choice == '2':
            cv2.imshow("result", maskedImage)

        elif choice == '3':
            cv2.imshow("result", cannyComboImage)

        elif choice == '4': 
            cv2.imshow("result", originalComboImage)

        if cv2.waitKey(1) == ord('q'):
            break
        elif cv2.waitKey(1) == ord('0'):
            choice = '0'
        elif cv2.waitKey(1) == ord('1'):
            choice = '1'
        elif cv2.waitKey(1) == ord('2'):
            choice = '2'
        elif cv2.waitKey(1) == ord('3'):
            choice = '3'
        elif cv2.waitKey(1) == ord('4'):
            choice = '4'

    cap.release()
    cv2.destroyAllWindows()

    print('\nfinished\n')

