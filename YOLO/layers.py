import tensorflow as tf
import numpy as np

def conv_layer(idx,inputs,filters,size,stride, alpha):
	
	channels = inputs.get_shape()[3]
	weight = tf.Variable(tf.truncated_normal([size,size,int(channels),filters], stddev=0.1))
	biases = tf.Variable(tf.constant(0.1, shape=[filters]))

	pad_size = size//2
	pad_mat = np.array([[0,0],[pad_size,pad_size],[pad_size,pad_size],[0,0]])
	inputs_pad = tf.pad(inputs,pad_mat)

	conv = tf.nn.conv2d(inputs_pad, weight, strides=[1, stride, stride, 1], padding='VALID',name=str(idx)+'_conv')
	conv_biased = tf.add(conv,biases,name=str(idx)+'_conv_biased')
	
	print('Layer  %d : Type = Conv, Size = %d * %d, Stride = %d, Filters = %d, Input channels = %d' % (idx,size,size,stride,filters,int(channels)))
	
	return tf.maximum(alpha*conv_biased,conv_biased,name=str(idx)+'_leaky_relu')


def pooling_layer(idx,inputs,size,stride):
	
	print ('Layer  %d : Type = Pool, Size = %d * %d, Stride = %d' % (idx,size,size,stride))
	
	return tf.nn.max_pool(inputs, ksize=[1, size, size, 1],strides=[1, stride, stride, 1], padding='SAME',name=str(idx)+'_pool')


def fc_layer(idx,inputs,hiddens, alpha, flat = False,linear = False):
	
	input_shape = inputs.get_shape().as_list()
	
	if flat:
		dim = input_shape[1]*input_shape[2]*input_shape[3]
		inputs_transposed = tf.transpose(inputs,(0,3,1,2))
		inputs_processed = tf.reshape(inputs_transposed, [-1,dim])
	
	else:
		dim = input_shape[1]
		inputs_processed = inputs
	
	weight = tf.Variable(tf.truncated_normal([dim,hiddens], stddev=0.1))
	biases = tf.Variable(tf.constant(0.1, shape=[hiddens]))
	print ('Layer  %d : Type = Full, Hidden = %d, Input dimension = %d, Flat = %d, Activation = %d' % (idx,hiddens,int(dim),int(flat),1-int(linear))	)
	
	if linear :
		return tf.add(tf.matmul(inputs_processed,weight),biases,name=str(idx)+'_fc')
	
	ip = tf.add(tf.matmul(inputs_processed,weight),biases)

	return tf.maximum(alpha*ip,ip,name=str(idx)+'_fc')
