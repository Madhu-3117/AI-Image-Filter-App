import cv2
import numpy as np
from PIL import Image, ImageEnhance

def cartoonify(image):
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.medianBlur(img_gray, 5)
    edges = cv2.adaptiveThreshold(img_gray, 255, 
                                  cv2.ADAPTIVE_THRESH_MEAN_C, 
                                  cv2.THRESH_BINARY, 9, 9)
    color = cv2.bilateralFilter(image, 9, 250, 250)
    cartoon = cv2.bitwise_and(color, color, mask=edges)
    return cartoon

def pencil_sketch(image):
    gray, sketch = cv2.pencilSketch(image, sigma_s=60, sigma_r=0.07, shade_factor=0.05)
    return sketch

def sepia(image):
    kernel = np.array([[0.272, 0.534, 0.131],
                       [0.349, 0.686, 0.168],
                       [0.393, 0.769, 0.189]])
    sepia_img = cv2.transform(image, kernel)
    sepia_img = np.clip(sepia_img, 0, 255)
    return sepia_img

def adjust_brightness_contrast(image, brightness=1.0, contrast=1.0):
    pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    enhancer_b = ImageEnhance.Brightness(pil_img)
    enhancer_c = ImageEnhance.Contrast(enhancer_b.enhance(brightness))
    enhanced_img = enhancer_c.enhance(contrast)
    return cv2.cvtColor(np.array(enhanced_img), cv2.COLOR_RGB2BGR)

def blur_faces(image):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        face = image[y:y+h, x:x+w]
        face = cv2.GaussianBlur(face, (99, 99), 30)
        image[y:y+h, x:x+w] = face
    return image
