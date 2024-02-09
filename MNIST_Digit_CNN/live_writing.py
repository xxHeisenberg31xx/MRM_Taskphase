import cv2 as cv
import numpy as np
from model import CNN
import torch
import torch.nn.functional as F
from torchvision import transforms as tf

if __name__ == "__main__":
    
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CNN()
    model.load_state_dict(torch.load('weights.pth', map_location = device))
    model.eval()
    model = model.to(device)
    
    frameHeight = 1280
    frameWidth = 720
    canvas = np.zeros((720, 1280, 3))
    
    def empty(a):
        pass
    
    cv.namedWindow("Parameters")
    cv.resizeWindow("Parameters", 640, 240)
    cv.createTrackbar("Threshold1", "Parameters", 30, 255, empty)
    cv.createTrackbar("Threshold2", "Parameters", 19, 255, empty)

    
    cap = cv.VideoCapture(0)
    cap.set(3, frameWidth)
    cap.set(4, frameHeight)
    
    def stackImages(scale,imgArray):
        rows = len(imgArray)
        cols = len(imgArray[0])
        rowsAvailable = isinstance(imgArray[0], list)
        width = imgArray[0][0].shape[1]
        height = imgArray[0][0].shape[0]
        if rowsAvailable:
            for x in range ( 0, rows):
                for y in range(0, cols):
                    if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
                        imgArray[x][y] = cv.resize(imgArray[x][y], (0, 0), None, scale, scale)
                    else:
                        imgArray[x][y] = cv.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                    if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv.cvtColor( imgArray[x][y], cv.COLOR_GRAY2BGR)
            imageBlank = np.zeros((height, width, 3), np.uint8)
            hor = [imageBlank]*rows
            hor_con = [imageBlank]*rows
            for x in range(0, rows):
                hor[x] = np.hstack(imgArray[x])
            ver = np.vstack(hor)
        else:
            for x in range(0, rows):
                if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                    imgArray[x] = cv.resize(imgArray[x], (0, 0), None, scale, scale)
                else:
                    imgArray[x] = cv.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
                if len(imgArray[x].shape) == 2: imgArray[x] = cv.cvtColor(imgArray[x], cv.COLOR_GRAY2BGR)
            hor= np.hstack(imgArray)
            ver = hor
        return ver
    
        
    def getContours(img, imgContour):
        
        contours, hierarchy = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        
        for cnt in contours:
            area = cv.contourArea(cnt)
            if area > 100:
                imgContour = cv.drawContours(imgContour, cnt, -1, (255, 0, 255), 3)
                peri = cv.arcLength(cnt, True)
                approx = cv.approxPolyDP(cnt, 0.02 * peri, True)
                x, y, w, h = cv.boundingRect(approx)
                bbox = cv.rectangle(imgContour, (x - 10, y - 10), (x + w + 10, y + h + 10), (0, 255, 0), 3)
                canvas = cv.rectangle(canvas, (x - 10, y - 10), (x + w + 10, y + h + 10), (0, 255, 0), 3)
                
                
                ROI = approx[x + 10:x + w + 10, y - 10:y + h + 10]
                ROI = cv.cvtColor(ROI, cv.COLOR_BGR2GRAY)
                
                tran = tf.Compose([tf.ToTensor(), tf.Resize((28, 28)), tf.Normalize((0.5,), (0.5,))])
                ROI_tensor = tran(ROI)
                with torch.no_grad():
                    output = model(ROI_tensor.unsqueeze(0))
                
                probabilities = F.softmax(output, dim=1)
                predicted_class = torch.argmax(output, dim=1).item()
                cv.putText(frame, f'Prediction : {predicted_class}', (x - 10, y - 10 - 5), cv.FONT_HERSHEY_DUPLEX, 1.5, (0, 0, 0), 2)
                print(probabilities)
                
    if not cap.isOpened():
        print("Cannot open/access camera")
        exit()

    while True:
        isTrue, frame = cap.read()
        frame = cv.flip(frame, 1)
        imgContour = frame.copy()
        
        if canvas is not None:
            canvas = np.zeros_like(frame)
        
        blur = cv.GaussianBlur(frame, (7,7), 1)
        gray = cv.cvtColor(blur, cv.COLOR_BGR2GRAY)
        thresh1 = cv.getTrackbarPos("Threshold1", "Parameters")
        thresh2 = cv.getTrackbarPos("Threshold2", "Parameters")
        canny = cv.Canny(gray, thresh1, thresh2)
        kernel = np.ones((3, 3))
        dilate = cv.dilate(canny, kernel, iterations = 1)
        
        contours, hierarchy = cv.findContours(dilate, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        
        for cnt in contours:
            area = cv.contourArea(cnt)
            if area > 100:
                imgContour = cv.drawContours(imgContour, cnt, -1, (255, 0, 255), 3)
                peri = cv.arcLength(cnt, True)
                approx = cv.approxPolyDP(cnt, 0.02 * peri, True)
                x, y, w, h = cv.boundingRect(approx)
                imgContour = cv.rectangle(imgContour, (x - 10, y - 10), (x + w + 10, y + h + 10), (0, 255, 0), 3)
                canvas = cv.rectangle(canvas, (x - 10, y - 10), (x + w + 10, y + h + 10), (0, 255, 0), 3)
                
                ROI = approx[x + 10:x + w + 10, y - 10:y + h + 10]
                
                if not ROI.size <= 0:
                
                    ROI = cv.cvtColor(ROI, cv.COLOR_BGR2GRAY)
                    
                    tran = tf.Compose([tf.ToTensor(), tf.Resize((28, 28)), tf.Normalize((0.5,), (0.5,))])
                    ROI_tensor = tran(ROI)
                    with torch.no_grad():
                        output = model(ROI_tensor.unsqueeze(0))
                    
                    probabilities = F.softmax(output, dim=1)
                    predicted_class = torch.argmax(output, dim=1).item()
                    cv.putText(imgContour, f'Prediction : {predicted_class}', (x - 10, y - 10 - 5), cv.FONT_HERSHEY_DUPLEX, 1.5, (0, 0, 0), 2)
                    print(probabilities)
                
                    imgContour = cv.add(canvas, imgContour)
        
        stack = stackImages(0.8, ([frame, blur, gray],
                                  [canny, dilate, imgContour]))
        
        if not isTrue:
            print("Cannot recieve frame. Exiting...")
            break
        
        
        
        cv.imshow('frame', stack)
        if cv.waitKey(1) == ord('q'):
            break
        
    cap.release()
    cv.destroyAllWindows()   