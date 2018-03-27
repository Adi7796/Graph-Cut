from __future__ import division
import cv2
# to show the image
from matplotlib import pyplot as plt
import numpy as np
from math import cos, sin

green = (0, 255, 0)


def show(image):
    # Figure size in inches
    plt.figure(figsize=(10, 10))

    # Show image, with nearest neighbour interpolation
    plt.imshow(image, interpolation='nearest')


def overlay_mask(mask, image):
    # make the mask rgb
    rgb_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
    # calculates the weightes sum of two arrays. in our case image arrays
    # input, how much to weight each.
    # optional depth value set to 0 no need
    img = cv2.addWeighted(rgb_mask, 0.5, image, 0.5, 0)
    return img


def find_biggest_contour(image):
    # Copy
    image = image.copy()

    image,contours, hierarchy = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # Isolate largest contour
    contour_sizes = [(cv2.contourArea(contour), contour) for contour in contours]
    biggest_contour = max(contour_sizes, key=lambda x: x[0])[1]

    mask = np.zeros(image.shape, np.uint8)
    cv2.drawContours(mask, [biggest_contour], -1, 255, -1)
    return biggest_contour, mask


def circle_contour(image, contour):
    # Bounding ellipse
    image_with_ellipse = image.copy()
    # easy function
    ellipse = cv2.fitEllipse(contour)
    # add it
    cv2.ellipse(image_with_ellipse, ellipse, green, 2, cv2.LINE_AA)
    return image_with_ellipse


def find_object(image):

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Make a consistent size
    # get largest dimension
    max_dimension = max(image.shape)
    # The maximum window size is 700 by 660 pixels. make it fit in that
    scale = 700 / max_dimension
    # resize it. same width and hieght none since output is 'image'.
    image = cv2.resize(image, None, fx=scale, fy=scale)

    # we want to eliminate noise from our image. clean. smooth colors without
    # dots
    # Blurs an image using a Gaussian filter. input, kernel size, how much to filter, empty)
    image_blur = cv2.GaussianBlur(image, (7, 7), 0)

    image_blur_hsv = cv2.cvtColor(image_blur, cv2.COLOR_RGB2HSV)

    # Filter by colour
    # 0-10 hue
    # minimum red amount, max red amount
    min_red = np.array([0, 100, 80])
    max_red = np.array([10, 256, 256])
    # layer
    mask1 = cv2.inRange(image_blur_hsv, min_red, max_red)

    # birghtness of a color is hue
    # 170-180 hue
    min_red2 = np.array([170, 100, 80])
    max_red2 = np.array([180, 256, 256])
    mask2 = cv2.inRange(image_blur_hsv, min_red2, max_red2)


    mask = mask1 + mask2
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))

    mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    # erosion followed by dilation. It is useful in removing noise
    mask_clean = cv2.morphologyEx(mask_closed, cv2.MORPH_OPEN, kernel)

    # Find biggest strawberry
    # get back list of segmented object and an outline for the biggest one
    big_object_contour, mask_object = find_biggest_contour(mask_clean)

    # Overlay cleaned mask on image
    # overlay mask on image, object now segmented
    overlay = overlay_mask(mask_clean, image)

    # Circle biggest object
    # circle the biggest one
    circled = circle_contour(overlay, big_object_contour)
    show(circled)


    bgr = cv2.cvtColor(circled, cv2.COLOR_RGB2BGR)

    return bgr



image = cv2.imread('berry.jpg')

result = find_object(image)

cv2.imwrite('berry2.jpg', result)