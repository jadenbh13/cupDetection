import cv2 as cv
import numpy as np
import time

#img = cv.imread('realCup.png')

# Load names of classes and get random colors
classes = open('coco.names').read().strip().split('\n')
np.random.seed(42)
colors = np.random.randint(0, 255, size=(len(classes), 3), dtype='uint8')

# Give the configuration and weight files for the model and load the network.
net = cv.dnn.readNetFromDarknet('yolov3.cfg', 'yolov3.weights')
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
# net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

# determine the output layer
ln = net.getLayerNames()

lnD = []
it = 0
while it < len(net.getUnconnectedOutLayers()):
    lnB = ln[(net.getUnconnectedOutLayers()[it]) - 1]
    lnD.append(lnB)
    it += 1
ln = lnD
print(ln)
confidence = 50/100

def getCupBox(img):
    blob = cv.dnn.blobFromImage(img, 1/255.0, (416, 416), swapRB=True, crop=False)
    r = blob[0, 0, :, :]

    #cv.imshow('blob', r)

    net.setInput(blob)
    outputs = net.forward(ln)
    #print('time=', t-t0)

    boxes = []
    confidences = []
    classIDs = []
    h, w = img.shape[:2]
    centerXs = []
    centerYs = []
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            if confidence > 0.5:
                box = detection[:4] * np.array([w, h, w, h])
                (centerX, centerY, width, height) = box.astype("int")
                centerXs.append(centerX)
                centerYs.append(centerY)
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                box = [x, y, int(width), int(height)]
                boxes.append(box)
                confidences.append(float(confidence))
                classIDs.append(classID)
    indices = cv.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    boxMainXC = []
    boxMainYC = []
    boxMainW = []
    boxMainH = []
    boxMainXT = []
    boxMainYT = []
    if len(indices) > 0:
        for i in indices.flatten():
            if classes[classIDs[i]] == 'cup':
                xCent = int(centerXs[i])
                yCent = int(centerYs[i])
                xCorn = int(xCent - (boxes[i][2] / 2))
                yCorn = int(yCent - (boxes[i][3] / 2))
                boxMainXC.append(xCent)
                boxMainYC.append(yCent)
                boxMainW.append(boxes[i][2])
                boxMainH.append(boxes[i][3])
                boxMainXT.append(xCorn)
                boxMainYT.append(yCorn)
                #cv.circle(img, (xCorn, yCorn), 10, (255, 0, 0), cv.FILLED)
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])
                #begin = (int(xCorn), int(yCent))
                #end = (int(xCorn + boxes[i][2]), int(yCent))
                #cv.line(img, (begin, end), (255, 0, 0), 9)
                #color = [int(c) for c in colors[classIDs[i]]]
                cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 5)
                #text = "{}: {:.4f}".format(classes[classIDs[i]], confidences[i])
                #cv.putText(img, text, (x, y - 5), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return boxMainXC, boxMainYC, boxMainW, boxMainH, boxMainXT, boxMainYT

def distance_to_camera(knownWidth, focalLength, perWidth):
	return (knownWidth * focalLength) / perWidth

KNOWN_DISTANCE = 30
# initialize the known object width, which in this case, the piece of
# paper is 12 inches wide
KNOWN_WIDTH = 10

image = cv.imread("30CmCup.png")
marx = getCupBox(image)
marker = (marx[0][0], marx[1][0], marx[2][0], marx[3][0])
focalLength = (marker[2] * KNOWN_DISTANCE) / KNOWN_WIDTH
print(focalLength)
# construct a blob from the image
cap = cv.VideoCapture(0)
while True:
    _, img = cap.read()
    he, we, ce = img.shape
    boundBox = getCupBox(img)
    #print(boundBox[0], boundBox[1])
    if len(boundBox[0]) > 0:
        midX = we / 2
        midY = he / 2
        cents = distance_to_camera(KNOWN_WIDTH, focalLength, boundBox[2][0])
        print(int(round(cents)))
        cv.circle(img, (boundBox[0][0], boundBox[1][0]), 10, (255, 0, 0), cv.FILLED)
        xV = boundBox[0][0]
        yV = boundBox[1][0]
        eachPix = KNOWN_WIDTH / boundBox[2][0]
        xMarg = 0
        yMarg = 0
        if xV < midX:
            xMarg = (0 - (midX - xV))
            xMag = abs(xMarg)
        elif xV > midX:
            xMarg = (xV - midX)
            xMag = abs(xMarg)
        if yV > midY:
            yMarg = (yV - midY)
            yMag = abs(yMarg)
        elif yV < midY:
            yMarg = (0 - (midY - yV))
            yMag = abs(yMarg)
        xDistance = xMarg * eachPix
        #print(xDistance)
        xDir = ""
        if xDistance < 1:
            xDir = "Left"
        elif xDistance > 1:
            xDir = "Right"
        print(xDir)
    cv.imshow('window', img)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()

cv.destroyAllWindows()
