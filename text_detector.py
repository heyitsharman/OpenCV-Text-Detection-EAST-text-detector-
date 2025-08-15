import cv2
import numpy as np
import pytesseract

# Windows users: set the tesseract path
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Load EAST model
net = cv2.dnn.readNet("frozen_east_textDetection.pb")

# Image path
image_path = "images/sample1.jpg"
image = cv2.imread(image_path)
orig = image.copy()
(H, W) = image.shape[:2]

# EAST model requires dimensions multiple of 32
newW, newH = (320, 320)
rW = W / float(newW)
rH = H / float(newH)

image = cv2.resize(image, (newW, newH))
blob = cv2.dnn.blobFromImage(image, 1.0, (newW, newH),
                             (123.68, 116.78, 103.94), swapRB=True, crop=False)
net.setInput(blob)
(layerNames) = [
    "feature_fusion/Conv_7/Sigmoid",
    "feature_fusion/concat_3"]
(scores, geometry) = net.forward(layerNames)

# Decode EAST output
def decode_predictions(scores, geometry, min_confidence):
    (numRows, numCols) = scores.shape[2:4]
    rects = []
    confidences = []

    for y in range(numRows):
        scoresData = scores[0, 0, y]
        xData0 = geometry[0, 0, y]
        xData1 = geometry[0, 1, y]
        xData2 = geometry[0, 2, y]
        xData3 = geometry[0, 3, y]
        anglesData = geometry[0, 4, y]

        for x in range(numCols):
            if scoresData[x] < min_confidence:
                continue

            offsetX, offsetY = x * 4.0, y * 4.0
            angle = anglesData[x]
            cos, sin = np.cos(angle), np.sin(angle)

            h = xData0[x] + xData2[x]
            w = xData1[x] + xData3[x]

            endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
            endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
            startX = int(endX - w)
            startY = int(endY - h)

            rects.append((startX, startY, endX, endY))
            confidences.append(float(scoresData[x]))

    return rects, confidences

rects, confidences = decode_predictions(scores, geometry, 0.5)
boxes = cv2.dnn.NMSBoxesRotated(
    [((x1+x2)//2, (y1+y2)//2, abs(x2-x1), abs(y2-y1), 0) for (x1, y1, x2, y2) in rects],
    confidences, 0.5, 0.4
)

# Draw bounding boxes & OCR
for i in range(len(rects)):
    if i in boxes:
        (startX, startY, endX, endY) = rects[i]
        startX, startY = int(startX * rW), int(startY * rH)
        endX, endY = int(endX * rW), int(endY * rH)

        roi = orig[startY:endY, startX:endX]
        text = pytesseract.image_to_string(roi)
        print(f"Detected Text: {text.strip()}")

        cv2.rectangle(orig, (startX, startY), (endX, endY), (0, 255, 0), 2)

cv2.imshow("Text Detection", orig)
cv2.waitKey(0)
cv2.destroyAllWindows()
