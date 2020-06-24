import tensorflow as tf
import numpy as np

""" linear operation """
def linear(input_, output_size, stddev=0.02, bias_start=0.0):
	shape = input_.get_shape().as_list()
	matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
			tf.random_normal_initializer(stddev=stddev))
	bias = tf.get_variable("bias", [output_size],
			initializer=tf.constant_initializer(bias_start))
	return tf.matmul(input_, matrix) + bias

""" negative log_likelihood """
def tf_normal(y, mu, s): # arguments are vectors
	with tf.variable_scope('normal'):
		result = 0.5 * tf.reduce_sum( # sum along axis=1
			tf.div(tf.square(tf.subtract(y,mu)), tf.maximum(1e-10,tf.square(s)))
			+ tf.log(2 * np.pi * tf.maximum(1e-10,tf.square(s))), axis=1)
	return result


""" cross_entropy """
def tf_cross_entropy(labels,logits): # arguments are vectors
	with tf.variable_scope('cross_entropy'):
		result = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels,logits=logits)
		# result = tf.nn.softmax_cross_entropy_with_logits(labels=labels,logits=logits)
	return result


def get_dlt(input_dt,lstep):
	""" Calculate delta time if performing l-step forward prediction
	"""
	shape = input_dt.get_shape().as_list()
	result = []

	seq_lens = shape[1]
	for i in range(seq_lens-lstep):
		aslice = tf.slice(input_dt, [0,i,0], [shape[0],lstep,1]) # slice lstep
		result.append(tf.reduce_sum(aslice,axis=1))

	for i in range(seq_lens-lstep,seq_lens):
		tmp = tf.squeeze(tf.slice(input_dt, [0,i,0], [shape[0],1,1]), [2])
		result.append(tmp)

	return tf.transpose(tf.stack(result), perm=[1,0,2])