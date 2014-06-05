import cv2
import os
import numpy as np
import math

#############################################
#          configurable part here:          #
#############################################
all_images_path = 'images/'
extension = '.jpg'

image_to_process_path = 'eiffel.jpg'

num_horizontal_images = 75
num_vertical_images = 100
#############################################

class Color:
    def __init__(self, r, g, b):
        self.r = r
        self.g = g
        self.b = b

def distance(color_1, color_2):
    #just return the 3d euclidean distance
    return math.sqrt((color_1.r - color_2.r) ** 2 + 
                     (color_1.g - color_2.g) ** 2 + 
                     (color_1.b - color_2.b) ** 2)

def get_avg_color_in_area(image, tl_x, tl_y, br_x, br_y):
    #if this is an integral image
    ##############
    #  A-----B   #
    #  |     |   #
    #  D-----C   #
    ##############
    #then the area of the ABCD rectangle can be computed
    #as easy as c - b - d + a
    a = image[tl_y, tl_x]
    b = image[tl_y, br_x + 1]
    c = image[br_y + 1, br_x + 1]
    d = image[br_y + 1, tl_x]

    #sum of all values in the b, g and r channels
    sum_bgr = c - b - d + a

    #this is the current area height times the current area width
    area = (br_y + 1 - tl_y) * (br_x + 1 - tl_x)

    #and here we get the average values for each channel
    avg_bgr = sum_bgr / area

    return Color(avg_bgr[2], avg_bgr[1], avg_bgr[0])

#opencv windows
cv2.namedWindow('image_to_process', flags = cv2.cv.CV_WINDOW_NORMAL)

#all the images from given path
image_paths = sorted([os.path.join(all_images_path, s) 
                    for s in os.listdir(all_images_path) 
                        if s.endswith(extension)])

#here we'll store the average color of an image at given path
color_path_list = []

curr = 0

#calculate the average color for each image in our image set
for image_path in image_paths:
    #read current image 
    current_image = cv2.imread(image_path)

    print current_image.dtype

    #calculate the integral image
    current_integral_image = cv2.integral(current_image)
    print current_integral_image.dtype

    #get the average color for the whole image
    avg_color = get_avg_color_in_area(current_integral_image, 0, 0, 
                                      current_image.shape[1] - 1, 
                                      current_image.shape[0] - 1)
    
    #let's save this info somewhere
    color_path_list.append((avg_color, image_path))

    #and just print how many images we processed
    curr += 1
    print curr, '/', len(image_paths)
    

#aaaand let's process the image we want to process
image_to_process = cv2.imread(image_to_process_path)

#look, it's here
cv2.imshow('image_to_process', image_to_process)

#calculate its integral image because integral images are cool
image_to_process_integral = cv2.integral(image_to_process)

#get the dimensions of small images
image_h = image_to_process.shape[0] / num_vertical_images
image_w = image_to_process.shape[1] / num_horizontal_images

#here we'll store the distances of all the colors we have from
#the color we need. and then we just use the nearest available
distances = np.zeros(len(color_path_list))

#just to make things a bit faster, we'll store the already 
#resized small images into this dictionary as we use them
#so that we don't have to load each image again and again.
cached = {}

for r in xrange(num_vertical_images):
    for c in xrange(num_horizontal_images):
        #the average color of the current image patch
        avg_color = get_avg_color_in_area(image_to_process_integral,
                                          c * image_w, r * image_h,
                                          (c + 1) * image_w - 1, 
                                          (r + 1) * image_h - 1)

        #let's calculate the distance of each color in our set 
        #from the current average color
        for i in xrange(len(color_path_list)):
            distances[i] = distance(avg_color, color_path_list[i][0])

        #the index of the closest color
        index = (np.abs(distances)).argmin()

        #and the path of the image which has this average color
        nearest_image_path = color_path_list[index][1]

        #if we haven't already used this image, then load it and resize it
        if (not nearest_image_path in cached):
            nearest_image = cv2.imread(nearest_image_path)
            resized = cv2.resize(nearest_image, (image_w, image_h))

            cached[nearest_image_path] = resized
        else:
            #otherwise just use the cached one
            resized = cached[nearest_image_path]


        #replace the pixels in the original image with the small image with
        #the correct average color
        image_to_process[r * image_h : (r + 1) * image_h,
                         c * image_w : (c + 1) * image_w] = resized[:, :]

        #display the current progress of our mosaic
        cv2.imshow('image_to_process', image_to_process)
        
        #wait a bit
        cv2.waitKey(10)

#save the generated image
cv2.imwrite('mosaic.png', image_to_process)

#wait for the user to press a key
cv2.waitKey(0)