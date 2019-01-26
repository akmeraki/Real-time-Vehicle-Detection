import numpy as np
import cv2
from scipy.ndimage.measurements import label
from copy import copy

buffer_weights=[0.1,0.2,0.3,0.4]
Heatmap_buffer = []
N_buffer = 3

# Define a function to draw bounding boxes
def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):

    imcopy = np.copy(img)
   
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)

    return imcopy


def add_heatmap(image, heatmap, bbox_list):

    im = copy(image)

    # Iterate through list of draw_bboxes
    for box in bbox_list:
        cv2.rectangle(im,(box[0][0],box[1][1]), (box[1][0],box[0][1]),(0,255,0),3)
       
        # Add 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    return heatmap


def nms(heatmap, threshold):

    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0

    return heatmap


def draw_bboxes(img, heatmap_buffer, heatmap_pre, N_buffer):

    heatmap_buffer.append(heatmap_pre)

    if len(heatmap_buffer) > N_buffer: # remove the first component if it is more than N_buffer elements
        heatmap_buffer.pop(0)

    # weight the heatmap based on current frame and previous N frames
    idxs = range(N_buffer)
    for b, w, idx in zip(heatmap_buffer, buffer_weights, idxs):
        heatmap_buffer[idx] = b * w

    heatmap = np.sum(np.array(heatmap_buffer), axis=0)
    heatmap = nms( heatmap, threshold= sum(buffer_weights[0:N_buffer])*2)

    # Find final boxes from heatmap using label function
    labels = label(heatmap)

    bboxes = []

    # locate the bounding box
    for car_number in range(1, labels[1]+1):

        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()

        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        # Define a bounding box based on min/max x and y
        bbox_tmp = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        bboxes.append(bbox_tmp)


    for bbox in bboxes:
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 4)

   
    return img, heatmap, bboxes


def generate_heatmaps(image, windows_list):

    heat = np.zeros_like(image[:,:,0]).astype(np.float)

    # Add heat to each box in box list
    heat = add_heatmap(image, heat, windows_list)

    # Apply threshold to help remove false positives
    heat = nms(heat, 1)

    # Visualize the heatmap when displaying
    heatmap = np.clip(heat, 0, 255)

    return heatmap


def get_fileNames(rootdir):

    data=[]
    for root, dirs, files in walk(rootdir, topdown=True):
        for name in files:
            _, ending = path.splitext(name)

            if ending != ".jpg" and ending != ".jepg" and ending != ".png":
                continue

            else:
                data.append(path.join(root, name))

    return data