from os import path
import pickle
from copy import copy
from utils import find_cars
from bbox import draw_bboxes, generate_heatmaps
from timeit import default_timer as timer
import numpy as np
from moviepy.editor import VideoFileClip

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("video_path", help="Enter the path to test video")
parser.add_argument("weights_path", help="Enter the path to weights")

args = parser.parse_args()


### Parameters

buffer_weights=[0.1,0.2,0.3,0.4]
Heatmap_buffer = []
N_buffer = 3


# SEARCH REGION AND SLIDING WINDOW
y_start_stop = [400, 720] # Min and max in y to search in slide_window()
ystart_0 = y_start_stop[0]
ystop_0 = ystart_0 + 64*2
ystart_1 = ystart_0
ystop_1 = y_start_stop[1]
ystart_2 = ystart_0
ystop_2 = y_start_stop[1]
ystarts = [ystart_1, ystart_2]
ystops = [ystop_1-100, ystop_2]
search_window_scales = [1.5, 2]  # (64x64), (96x96), (128x128)


clf_path = args.weights_path


def vehicle_detection_svm(image):

    # if svm classifer exist, load it; otherwise, compute the svm classifier
    if path.isfile(clf_path):

        print('loading existing classifier...')
        with open(clf_path, 'rb') as file:
            clf_pickle = pickle.load(file)
            svc = clf_pickle["svc"]
            X_scaler = clf_pickle["scaler"]
            orient = clf_pickle["orient"]
            pix_per_cell = clf_pickle["pix_per_cell"]
            cell_per_block = clf_pickle["cell_per_block"]
            spatial_size = clf_pickle["spatial_size"]
            hist_bins = clf_pickle["hist_bins"]
            color_space = clf_pickle["color_space"]
            
    start = timer()

    windows_list = []
    for search_window_scale, ystart, ystop in zip(search_window_scales, ystarts, ystops):
        windows_list_tmp = find_cars(np.copy(image), ystart, ystop, search_window_scale, svc, X_scaler, orient, pix_per_cell, cell_per_block,
                            spatial_size, hist_bins)
        windows_list.extend(windows_list_tmp)

    heatmap_pre = generate_heatmaps(image, windows_list)

    draw_img, heatmap_post, bboxes = draw_bboxes(np.copy(image), copy(Heatmap_buffer), heatmap_pre, min(len(Heatmap_buffer)+1,N_buffer) )

    if len(Heatmap_buffer) >= N_buffer:
        Heatmap_buffer.pop(0)

    fps = 1.0 / (timer() - start)

    return draw_img

if __name__ == '__main__':
    video_output = 'op.mp4'
    clip1 = VideoFileClip(args.video_path).subclip(30,32)
    clip = clip1.fl_image(vehicle_detection_svm)
    clip.write_videofile(video_output, audio=False)