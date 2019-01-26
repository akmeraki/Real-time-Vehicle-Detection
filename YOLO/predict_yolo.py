import numpy as np
import cv2
import tensorflow as tf
from timeit import default_timer as timer
import pdb
from model import small_yolo
import numpy as np
from moviepy.editor import VideoFileClip

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("video_path", help="Enter the path to test video")
parser.add_argument("weights_path", help="Enter the path to weights")

args = parser.parse_args()


def detect_from_cvmat(yolo,img):
	yolo.h_img,yolo.w_img,_ = img.shape
	img_resized = cv2.resize(img, (448, 448))
	img_resized_np = np.asarray( img_resized )
	inputs = np.zeros((1,448,448,3),dtype='float32')
	inputs[0] = (img_resized_np/255.0)*2.0-1.0
	in_dict = {yolo.x: inputs}
	net_output = yolo.sess.run(yolo.fc_32,feed_dict=in_dict)
	result = predict_bbox(img, yolo, net_output[0])
	yolo.result_list = result


def detect_from_file(yolo,filename):
	detect_from_cvmat(yolo, filename)


def nms(boxes, probs, threshold):
	
	filter_mat_probs = np.array(probs>=yolo.threshold,dtype='bool')
	filter_mat_boxes = np.nonzero(filter_mat_probs)
	boxes_filtered = boxes[filter_mat_boxes[0],filter_mat_boxes[1],filter_mat_boxes[2]]
	probs_filtered = probs[filter_mat_probs]
	classes_num_filtered = np.argmax(filter_mat_probs,axis=3)[filter_mat_boxes[0],filter_mat_boxes[1],filter_mat_boxes[2]]

	argsort = np.array(np.argsort(probs_filtered))[::-1]
	boxes_filtered = boxes_filtered[argsort]
	probs_filtered = probs_filtered[argsort]
	classes_num_filtered = classes_num_filtered[argsort]

	for i in range(len(boxes_filtered)):
		
		if probs_filtered[i] == 0 : 
			continue
		
		for j in range(i+1,len(boxes_filtered)):
			if calculate_iou(boxes_filtered[i],boxes_filtered[j]) > yolo.iou_threshold :
				probs_filtered[j] = 0.0

	filter_iou = np.array(probs_filtered>0.0,dtype='bool')
	boxes_filtered = boxes_filtered[filter_iou]
	probs_filtered = probs_filtered[filter_iou]
	classes_num_filtered = classes_num_filtered[filter_iou]

	return classes_num_filtered, boxes_filtered, probs_filtered

def predict_bbox(img, yolo,output):
	
	probs = np.zeros((7,7,2,20))
	class_probs = np.reshape(output[0:980],(7,7,20))
	scales = np.reshape(output[980:1078],(7,7,2))
	boxes = np.reshape(output[1078:],(7,7,2,4))
	offset = np.transpose(np.reshape(np.array([np.arange(7)]*14),(2,7,7)),(1,2,0))
	im = img.copy()
	boxes[:,:,:,0] += offset
	boxes[:,:,:,1] += np.transpose(offset,(1,0,2))
	boxes[:,:,:,0:2] = boxes[:,:,:,0:2] / 7.0
	boxes[:,:,:,2] = np.multiply(boxes[:,:,:,2],boxes[:,:,:,2])
	boxes[:,:,:,3] = np.multiply(boxes[:,:,:,3],boxes[:,:,:,3])

	boxes[:,:,:,0] *= yolo.w_img
	boxes[:,:,:,1] *= yolo.h_img
	boxes[:,:,:,2] *= yolo.w_img
	boxes[:,:,:,3] *= yolo.h_img

	bossd = boxes.reshape(-1,4)

	for box in bossd:
		cv2.rectangle(im,(int(box[0]-box[2]/2),int(box[1]-box[3]/2)), (int(box[0]+box[2]/2),int(box[1]+box[3]/2)),(0,255,0),3)

	dx, dy = int(1280/7),int(720/7)

	grid_color = -1
	im[:,::dy] = grid_color
	im[::dx,:] = grid_color
	
	for i in range(2):
		for j in range(20):
			probs[:,:,i,j] = np.multiply(class_probs[:,:,j],scales[:,:,i])

	classes_num_filtered, boxes_filtered, probs_filtered = nms(boxes, probs, yolo.threshold)

	result = []
	
	for i in range(len(boxes_filtered)):
		result.append([yolo.classes[classes_num_filtered[i]],boxes_filtered[i][0],boxes_filtered[i][1],boxes_filtered[i][2],boxes_filtered[i][3],probs_filtered[i]])
	
	return result


def draw_results(img, yolo, fps):
	
	img_cp = img.copy()
	results = yolo.result_list

	window_list = []
	for i in range(len(results)):
		x = int(results[i][1])
		y = int(results[i][2])
		w = int(results[i][3])//2
		h = int(results[i][4])//2
		cv2.rectangle(img_cp,(x-w,y-h),(x+w,y+h),(0,0,255),4)
	
	return img_cp

def calculate_iou(box1,box2):

	tb = min(box1[0]+0.5*box1[2],box2[0]+0.5*box2[2])-max(box1[0]-0.5*box1[2],box2[0]-0.5*box2[2])
	lr = min(box1[1]+0.5*box1[3],box2[1]+0.5*box2[3])-max(box1[1]-0.5*box1[3],box2[1]-0.5*box2[3])
	
	if tb < 0 or lr < 0 : 
		intersection = 0

	else : 
		intersection =  tb*lr

	union = (box1[2]*box1[3] + box2[2]*box2[3] - intersection)

	return intersection / union


yolo = small_yolo(args.weights_path)

def vehicle_detection_yolo(image):
	
	# set the timer
	start = timer()
	detect_from_file(yolo, image)

	# compute frame per second
	fps = 1.0 / (timer() - start)
	
	# draw visualization on frame
	yolo_result = draw_results(image, yolo, fps)

	return yolo_result

if __name__ == '__main__':
    video_output = 'op.mp4'
    clip1 = VideoFileClip(args.video_path).subclip(30,32)
    clip = clip1.fl_image(vehicle_detection_yolo)
    clip.write_videofile(video_output, audio=False)