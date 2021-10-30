import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import os, glob
import cv2
import math
from moviepy.editor import VideoFileClip
from collections import deque

QUEUE_LENGTH=50

def show_images(images, cmap='gray'):
    cols = 2
    rows = (len(images)+1)//cols
    
    plt.figure(figsize=(10, 11))
    for i, image in enumerate(images):
        plt.subplot(rows, cols, i+1)
        # use gray scale color map if there is only one channel
        cmap = 'gray' if len(image.shape)==2 else cmap
        plt.imshow(image, cmap=cmap)
        plt.xticks([])
        plt.yticks([])
    plt.tight_layout(pad=0, h_pad=0, w_pad=0)
    plt.show()

#seperate white and yellow colors
def convert_hls(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2HLS)

def select_white_yellow(img):

    converted = convert_hls(img)
    
    lower = np.uint8([  0, 200,   0])
    upper = np.uint8([255, 255, 255])
    white_mask = cv2.inRange(converted, lower, upper)
    
    lower = np.uint8([ 10,   0, 100])
    upper = np.uint8([ 40, 255, 255])
    yellow_mask = cv2.inRange(converted, lower, upper)
    
    mask = cv2.bitwise_or(white_mask, yellow_mask)
    return cv2.bitwise_and(img, img, mask = mask)


#in order to detect edges, we should do three things
#1 convert the image into gray using cv2.cvtColor
#2 smooth out rough edges using cv2.GaussianBlur
#3 find edges using cv2.Canny

#1
def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

#2
def gaussian_blur(img, kernel_size=15):
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

#3
def canny(img, low_threshold=50, high_threshold=150):
    return cv2.Canny(img, low_threshold, high_threshold)


#seperate the region that we wanna work on
def region_of_interest(img, vertices):
    
    mask = np.zeros_like(img)   
    
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def select_region(image):
   
    rows, cols = image.shape[:2]
    bottom_left  = [cols*0.1, rows*0.95]
    top_left     = [cols*0.4, rows*0.6]
    bottom_right = [cols*0.9, rows*0.95]
    top_right    = [cols*0.6, rows*0.6] 

    vertices = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)
    return region_of_interest(image, vertices)

def hough_lines(image):
    return cv2.HoughLinesP(image, rho=1, theta=np.pi/180, threshold=20, minLineLength=20, maxLineGap=300)

#draw red lines
def draw_lines(image, lines, color=[255, 0, 0], thickness=2):
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(image, (x1, y1), (x2, y2), color, thickness)
    return image

def average_slope_intercept(lines):
    left_lines    = []
    left_weights  = []
    right_lines   = []
    right_weights = []

    for line in lines:
        for x1, y1, x2, y2 in line:
            if x2==x1:
                continue
            slope = (y2-y1)/(x2-x1)
            intercept = y1 - slope*x1
            length = np.sqrt((y2-y1)**2+(x2-x1)**2)
            if slope < 0:
                left_lines.append((slope, intercept))
                left_weights.append((length))
            else:
                right_lines.append((slope, intercept))
                right_weights.append((length))
    
     
    left_lane  = np.dot(left_weights,  left_lines) /np.sum(left_weights)  if len(left_weights) >0 else None
    right_lane = np.dot(right_weights, right_lines)/np.sum(right_weights) if len(right_weights)>0 else None
    
    return left_lane, right_lane

def make_line_points(y1, y2, line):
    
    if line is None:
        return None
    
    slope, intercept = line
    
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)
    y1 = int(y1)
    y2 = int(y2)
    
    return ((x1, y1), (x2, y2))

def lane_lines(image, lines):
    left_lane, right_lane = average_slope_intercept(lines)
    
    y1 = image.shape[0] 
    y2 = y1*0.6         

    left_line  = make_line_points(y1, y2, left_lane)
    right_line = make_line_points(y1, y2, right_lane)
    
    return left_line, right_line

#draw highlighted lines  
def draw_lane_lines(image, lines, color=[255, 0, 0], thickness=20):

    line_image = np.zeros_like(image)
    for line in lines:
        if line is not None:
            cv2.line(line_image, *line,  color, thickness)

    return cv2.addWeighted(image, 0.8, line_image, 1.0, 0.)
             
def process_video(video_input, video_output):
    detector = LaneDetector()

    clip = VideoFileClip(os.path.join('test_videos', video_input))
    processed = clip.fl_image(detector.process)
    processed.write_videofile(os.path.join('output_videos', video_output), audio=False)

class LaneDetector:
    def __init__(self):
        self.left_lines  = deque(maxlen=QUEUE_LENGTH)
        self.right_lines = deque(maxlen=QUEUE_LENGTH)

    def process(self, image):
        white_yellow = select_white_yellow(image)
        gray         = grayscale(white_yellow)
        smooth_gray  = gaussian_blur(gray)
        edges        = canny(smooth_gray)
        regions      = select_region(edges)
        lines        = hough_lines(regions)
        left_line, right_line = lane_lines(image, lines)

        def mean_line(line, lines):
            if line is not None:
                lines.append(line)

            if len(lines)>0:
                line = np.mean(lines, axis=0, dtype=np.int32)
                line = tuple(map(tuple, line)) # make sure it's tuples not numpy array for cv2.line to work
            return line

        left_line  = mean_line(left_line,  self.left_lines)
        right_line = mean_line(right_line, self.right_lines)

        return draw_lines(image, lines)


images = [plt.imread(path) for path in glob.glob('test_images/*.jpg')]
white_yellow = list(map(select_white_yellow, images))
gray = list(map(grayscale, white_yellow))
blurred = list(map(gaussian_blur, gray))
edge = list(map(canny, blurred))
select = list(map(select_region, edge))
lines_selected = list(map(hough_lines, select))

#for drawing red lines
line_images = []
for image0, lines0 in zip(images, lines_selected):
    line_images.append(draw_lines(image0, lines0))

#for highlighting lights
# lane_images = []
# for image1, lines1 in zip(images, lines_selected):
#     lane_images.append(draw_lane_lines(image1, lane_lines(image1, lines1)))
# show_images(lane_images)
process_video('challenge.mp4', 'challengeRedLine.mp4')
process_video('solidWhiteRight.mp4', 'whiteRedLine.mp4')
process_video('solidYellowLeft.mp4', 'yellowRedLine.mp4')