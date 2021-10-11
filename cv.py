import asyncio
import cv2 as cv
import numpy as np
import imutils



async def canny_img(img):
    """
    Canny edge detection
    """
    img = cv.Canny(img, 75, 120)
    img = cv.cvtColor(img, cv.COLOR_GRAY2RGB)
    return img

async def cartoonify(frame):

    gray = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)
    gray = cv.medianBlur(gray, 3)
    edges = cv.adaptiveThreshold(
        gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 9, 10)
  # Making a Cartoon of the image
    color = cv.bilateralFilter(frame, 12, 250, 250)
    cartoon = cv.bitwise_and(color, color, mask=edges)
    cartoon_image = cv.stylization(frame, sigma_s=150, sigma_r=0.25)
    frame = cartoon_image
    #frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    frame = cv.resize(frame, (640, 480))
    return frame


async def watercolor(frame):
    frame = cv.stylization(frame, sigma_s=60, sigma_r=0.6)
    #frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    frame = cv.resize(frame, (640, 480))
    return frame


async def pencil(frame):
    pencil, color = cv.pencilSketch(frame, sigma_s=60, sigma_r=0.5, shade_factor=0.010)
    #frame = cv.cvtColor(pencil, cv.COLOR_BGR2RGB)
    frame = cv.resize(frame, (640, 480)) 
    return frame

async def econify(frame):
    canny = canny_img(frame)

    blue, g, r = cv.split(canny) 
    blank = np.zeros(canny.shape[:2], dtype='uint8')

    green = cv.merge([blank,g,blank])
        
    frame = green
    #frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    frame = cv.resize(frame, (640, 480))
    return frame

async def negative(frame):
    frame = cv.bitwise_not(frame)
    #frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    frame = cv.resize(frame, (640, 480))
    return frame



async def style_transfer(image, model_path):
    model = cv.dnn.readNetFromTorch(model_path)
    
    (h, w) = image.shape[:2]
    # image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR) #PIL Jpeg to Opencv image

    blob = cv.dnn.blobFromImage(image, 1.0, (w, h), (103.939, 116.779, 123.680), swapRB=False, crop=False)
    model.setInput(blob)
    output = model.forward()

    output = output.reshape((3, output.shape[2], output.shape[3]))
    output[0] += 103.939
    output[1] += 116.779
    output[2] += 123.680
    output /= 255.0
    output = output.transpose(1, 2, 0)
    output = np.clip(output, 0.0, 1.0)
    output = imutils.resize(output, width=500)
    return output
