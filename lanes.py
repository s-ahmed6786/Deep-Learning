import cv2
import numpy as np

def canny(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    kernel = 5
    blur = cv2.GaussianBlur(gray,(kernel, kernel),0)
    canny = cv2.Canny(gray, 50, 150)
    return canny

def region_of_interest(canny):
    height = canny.shape[0]
   
    triangle = np.array([[
    (200, height),
    (550, 250),
    (1100, height),]])
    
    mask = np.zeros_like(canny)
    
    cv2.fillPoly(mask, triangle, 255)
    masked_image = cv2.bitwise_and(canny, mask)
    return masked_image

def houghLines(img):
    lines = cv2.HoughLinesP(img, 2, np.pi/180, 100, np.array([]), minLineLength=40,maxLineGap=5)
    return lines

def make_coordinates(image, line):
    slope, intercept = line
    y1 = int(image.shape[0])# bottom of the image
    y2 = int(y1*3/5)         # slightly lower than the middle
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)
    return [[x1, y1, x2, y2]]
 
def average_slope_intercept(image, lines):
    left_fit    = []
    right_fit   = []
    if lines is None:
        return None
    for line in lines:
        x1,y1,x2,y2 = line.reshape(4)
        slope, intercept = np.polyfit((x1,x2), (y1,y2), 1)
        if slope < 0: 
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))
    
    # if len(left_fit) and len(right_fit):
    # ##over-simplified if statement (should give you anQ idea of why the error occurs)
    #     left_fit_average  = np.average(left_fit, axis=0)
    #     right_fit_average = np.average(right_fit, axis=0)

    #     left_line  = make_coordinates(image, left_fit_average)
    #     right_line = make_coordinates(image, right_fit_average)

    #     averaged_lines = [left_line, right_line]
    #     return averaged_lines

    left_fit_average  = np.average(left_fit, axis=0)
    right_fit_average = np.average(right_fit, axis=0)

    left_line  = make_coordinates(image, left_fit_average)
    right_line = make_coordinates(image, right_fit_average)
    
    averaged_lines = np.array([left_line, right_line])
    return averaged_lines
 

 
def display_lines(img,lines):
    line_image = np.zeros_like(img)
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),10)
    return line_image
 


cap = cv2.VideoCapture("test2.mp4")
while(cap.isOpened()):
    _, frame = cap.read()
    canny_image = canny(frame)
    masked_image = region_of_interest(canny_image)
    lines = houghLines(masked_image)
    averaged_lines = average_slope_intercept(frame, lines)
    line_image = display_lines(frame, averaged_lines)
    combo_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
    
    cv2.imshow("result", combo_image)
    cv2.waitKey(1)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
