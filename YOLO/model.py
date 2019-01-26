import tensorflow as tf
from layers import *

class small_yolo:
	w_img = 1280
	h_img = 720

	alpha = 0.1
	threshold = 0.3
	iou_threshold = 0.4

	result_list = None
	classes =  ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair",
				"cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant",
				"sheep", "sofa", "train","tvmonitor"]

	def __init__(self, weights_file):
		self.weights_file = weights_file
		self.build_networks()

	def build_networks(self):
		print("Building YOLO_small graph...")
		
		self.x = tf.placeholder('float32',[None,448,448,3])
		self.conv_1 = self.conv_layer(1,self.x,64,7,2)
		self.pool_2 = self.pooling_layer(2,self.conv_1,2,2)

		self.conv_3 = self.conv_layer(3,self.pool_2,192,3,1)
		self.pool_4 = self.pooling_layer(4,self.conv_3,2,2)
		
		self.conv_5 = self.conv_layer(5,self.pool_4,128,1,1)
		self.conv_6 = self.conv_layer(6,self.conv_5,256,3,1)
		self.conv_7 = self.conv_layer(7,self.conv_6,256,1,1)
		self.conv_8 = self.conv_layer(8,self.conv_7,512,3,1)
		self.pool_9 = self.pooling_layer(9,self.conv_8,2,2)
		
		self.conv_10 = self.conv_layer(10,self.pool_9,256,1,1)
		self.conv_11 = self.conv_layer(11,self.conv_10,512,3,1)
		self.conv_12 = self.conv_layer(12,self.conv_11,256,1,1)
		self.conv_13 = self.conv_layer(13,self.conv_12,512,3,1)
		self.conv_14 = self.conv_layer(14,self.conv_13,256,1,1)
		self.conv_15 = self.conv_layer(15,self.conv_14,512,3,1)
		self.conv_16 = self.conv_layer(16,self.conv_15,256,1,1)
		self.conv_17 = self.conv_layer(17,self.conv_16,512,3,1)
		self.conv_18 = self.conv_layer(18,self.conv_17,512,1,1)
		self.conv_19 = self.conv_layer(19,self.conv_18,1024,3,1)
		self.pool_20 = self.pooling_layer(20,self.conv_19,2,2)
		
		self.conv_21 = self.conv_layer(21,self.pool_20,512,1,1)
		self.conv_22 = self.conv_layer(22,self.conv_21,1024,3,1)
		self.conv_23 = self.conv_layer(23,self.conv_22,512,1,1)
		self.conv_24 = self.conv_layer(24,self.conv_23,1024,3,1)
		self.conv_25 = self.conv_layer(25,self.conv_24,1024,3,1)
		self.conv_26 = self.conv_layer(26,self.conv_25,1024,3,2)
		self.conv_27 = self.conv_layer(27,self.conv_26,1024,3,1)
		self.conv_28 = self.conv_layer(28,self.conv_27,1024,3,1)
		
		self.fc_29 = self.fc_layer(29,self.conv_28,512,flat=True,linear=False)
		self.fc_30 = self.fc_layer(30,self.fc_29,4096,flat=False,linear=False)
		self.fc_32 = self.fc_layer(32, self.fc_30, 1470, flat=False, linear=True)
		
		self.sess = tf.Session()
		self.sess.run(tf.global_variables_initializer())
		
		print("Loading Weights...")
		self.saver = tf.train.Saver()
		self.saver.restore(self.sess, self.weights_file)
		
		print("Loading complete!")

	def conv_layer(self,idx,inputs,filters,size,stride):
		
		return conv_layer(idx, inputs, filters, size, stride, self.alpha)

	def pooling_layer(self,idx,inputs,size,stride):

		return pooling_layer(idx, inputs, size, stride)

	def fc_layer(self,idx,inputs,hiddens,flat = False,linear = False):
		
		return fc_layer(idx, inputs, hiddens, self.alpha, flat = flat, linear = linear)