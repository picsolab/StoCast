import tensorflow as tf
import numpy as np
from modelutils import *

class TLSTMCell(tf.contrib.rnn.RNNCell):

	def __init__(self, n_x, n_h, batch_size):
		self.n_x = n_x # feature dimension except t
		self.n_h = n_h
		self.batch_size = batch_size

	@property
	def state_size(self):
		return (self.n_h,self.n_h)

	@property
	def output_size(self):
		return (self.n_h,self.n_h)

	def __call__(self, input, state, scope=None):

		with tf.variable_scope(scope or type(self).__name__):
			h, c = state

		# slice x and dt
		x = tf.slice(input, [0,0], [self.batch_size,self.n_x])
		dt = tf.slice(input, [0,self.n_x], [self.batch_size,1])
		dlt = tf.slice(input, [0,self.n_x+1], [self.batch_size,1])


		with tf.variable_scope("log_elapse_time"):
			log_dt = self.time_decay(dt)

		with tf.variable_scope("log_elapse_time",reuse=True):
			log_dlt = self.time_decay(dlt)

		# T_LSTM major contribution
		with tf.variable_scope("adjusted_memory"):
			adj_c = self.adjusted_memory(x,log_dt,c)

		with tf.variable_scope("adjusted_memory",reuse=True):
			adj_c_dlt = self.adjusted_memory(x,log_dlt,c)

		with tf.variable_scope("input_gate"):
			with tf.variable_scope("linear_x"):
				linear_x = linear(x,self.n_h)
			with tf.variable_scope("linear_h"):
				linear_h = linear(h,self.n_h)
			input_gate = tf.sigmoid(linear_x + linear_h)

		with tf.variable_scope("forget_gate"):
			with tf.variable_scope("linear_x"):
				linear_x = linear(x,self.n_h)
			with tf.variable_scope("linear_h"):
				linear_h = linear(h,self.n_h)
			forget_gate = tf.sigmoid(linear_x + linear_h)

		with tf.variable_scope("output_gate"):
			with tf.variable_scope("linear_x"):
				linear_x = linear(x,self.n_h)
			with tf.variable_scope("linear_h"):
				linear_h = linear(h,self.n_h)
			output_gate = tf.sigmoid(linear_x + linear_h)

		with tf.variable_scope("candidate_values"):
			with tf.variable_scope("linear_x"):
				linear_x = linear(x,self.n_h)
			with tf.variable_scope("linear_h"):
				linear_h = linear(h,self.n_h)
			candidate_values = tf.tanh(linear_x + linear_h)

		new_c = forget_gate * adj_c + input_gate * candidate_values # current memory
		new_h = output_gate * tf.tanh(new_c) # current hidden state

		# Testing
		new_c_dlt = forget_gate * adj_c_dlt + input_gate * candidate_values
		new_h_dlt = output_gate * tf.tanh(new_c_dlt)

		new_state = tf.nn.rnn_cell.LSTMStateTuple(new_c, new_h)

		return (new_h,new_h_dlt),new_state

	def time_decay(self,dt):
		Ones = tf.ones([1, self.n_h], dtype=tf.float32)
		log_dt = tf.map_fn(lambda x: tf.div( 1.0, tf.log(tf.exp(1.0) + x)), dt)
		log_dt = tf.matmul(log_dt, Ones)
		return log_dt

	def adjusted_memory(self,x,log_dt,c):
		with tf.variable_scope("short_term_memory"):
			ST_C = tf.nn.tanh(linear(x, self.n_h))
			DIS_ST_C = tf.matmul(log_dt, ST_C)	# discounted short term memory
		with tf.variable_scope("adjusted_memory"):
			adj_c = c - ST_C + DIS_ST_C
		return adj_c


class tlstm():
	def __init__(self, n_demo, n_x, n_y, max_seq_len, batch_size, n_h, n_z, lstep, feature_weight):

		self.n_x = n_x
		self.n_y = n_y
		self.max_seq_len = max_seq_len
		self.batch_size = batch_size
		self.n_h = n_h
		self.n_z = n_z
		self.lstep = lstep
		self.feature_weight = feature_weight

		""" placeholder for inputs """
		self.input_demo = tf.placeholder(dtype=tf.float32, shape=[batch_size,n_demo], name="input_demographics")
		self.input_x = tf.placeholder(dtype=tf.float32, shape=[batch_size,max_seq_len,n_x], name="input_numeric_data")
		self.input_y = tf.placeholder(dtype=tf.float32, shape=[batch_size,max_seq_len,n_y], name="input_discrete_data")
		self.input_dt = tf.placeholder(dtype=tf.float32, shape=[batch_size,max_seq_len,1], name="input_dt")
		self.seqlens = tf.placeholder(dtype=tf.int32, shape=[batch_size], name="sequence_lengths")
		self.mask = tf.placeholder(dtype=tf.float32, shape=[batch_size, max_seq_len], name="mask")

		""" define cell type """
		self.cell = TLSTMCell(n_x + n_y + n_h, n_h, batch_size) 
		
		""" initial_c, initial_h depends on demographic info """
		with tf.variable_scope("phi_demo"):
			phi_demo = tf.layers.dense(inputs=self.input_demo, units=n_h, activation=tf.sigmoid)

		""" inputs """
		with tf.variable_scope("input_num"):
			input_num = tf.slice(self.input_x,[0,0,0],[batch_size,max_seq_len-1,n_x])
			input_num = tf.unstack(input_num,num=max_seq_len-1,axis=1)
		with tf.variable_scope("input_dsc"):
			input_dsc = tf.slice(self.input_y,[0,0,0],[batch_size,max_seq_len-1,n_y])
			input_dsc = tf.layers.dense(inputs=tf.reshape(input_dsc,[batch_size * (max_seq_len-1), -1]),units=n_y,activation=tf.sigmoid)
			input_dsc = tf.reshape(input_dsc,[batch_size,max_seq_len-1,-1])
			input_dsc = tf.unstack(input_dsc,num=max_seq_len-1,axis=1)
		with tf.variable_scope("input_dt"):
			input_dt = tf.slice(self.input_dt,[0,1,0],[batch_size,max_seq_len-1,1])
			input_dt = tf.unstack(input_dt,num=max_seq_len-1,axis=1)
		with tf.variable_scope("input_dlt"):
			input_dlt = tf.slice(self.input_dt,[0,1,0],[batch_size,max_seq_len-1,1])
			input_dlt = get_dlt(input_dlt,lstep)
			input_dlt = tf.unstack(input_dlt,num=max_seq_len-1,axis=1)
		
		inputs = [tf.concat([u,v,phi_demo,t,dlt],1) for u,v,t,dlt in zip(input_num,input_dsc,input_dt,input_dlt)]

		""" dynamic time_aware_lstm """
		initial_c,initial_h = self.cell.zero_state(batch_size,dtype=tf.float32)
		outputs,last_state = tf.contrib.rnn.static_rnn(self.cell, inputs, 
			initial_state=tf.nn.rnn_cell.LSTMStateTuple(initial_c,initial_h),
			dtype=tf.float32, sequence_length=self.seqlens-1)
		
		flat_package = {}
		names = ["new_h","new_h_dlt"]
		for i,name in enumerate(names):
			with tf.variable_scope(name):
				tensor_of_name = tf.stack([output_k[i] for output_k in outputs])
				tensor_of_name = tf.transpose(tensor_of_name,[1,0,2])
				tensor_of_name = tf.layers.dropout(inputs=tf.reshape(tensor_of_name,[batch_size * (max_seq_len-1),-1]),rate=0.2)
				flat_package[name] = tensor_of_name

		
		""" outputs map to parameters """
		new_h = flat_package['new_h']
		with tf.variable_scope('outputs'):
			output_pi_flat,output_mu_flat,output_sigma_flat = self.outputs(new_h,n_x,n_y)

		# Testing
		new_h_dlt = flat_package['new_h_dlt']
		with tf.variable_scope("outputs",reuse=True):
			output_pi_flat_dlt,output_mu_flat_dlt,output_sigma_flat_dlt = self.outputs(new_h_dlt,n_x,n_y)
		
		""" target flat """
		with tf.variable_scope("target_num"):
			target_num_flat = tf.reshape(tf.slice(self.input_x,[0,1,0],[batch_size,max_seq_len-1,n_x]),[-1,n_x])
		with tf.variable_scope("target_dsc"):
			target_dsc_flat = tf.reshape(tf.slice(self.input_y,[0,1,0],[batch_size,max_seq_len-1,n_y]),[-1,n_y])
		
		mask_flat = tf.reshape(tf.slice(self.mask,[0,1],[batch_size,max_seq_len-1]),[-1])
		
		pkg = {"target_num_flat": target_num_flat,
				"target_dsc_flat": target_dsc_flat,
				"output_mu_flat": output_mu_flat,
				"output_sigma_flat": output_sigma_flat,
				"output_pi_flat": output_pi_flat,
				"mask_flat": mask_flat,
				"output_pi_flat_dlt":output_pi_flat_dlt,
				"output_mu_flat_dlt":output_mu_flat_dlt,
				"output_sigma_flat_dlt":output_sigma_flat_dlt}

		""" loss & train """
		loss = self.calculate_loss(pkg)
		with tf.variable_scope("loss"):
			self.loss = loss
		tf.summary.scalar("loss", self.loss)
		self.lr = tf.Variable(0.0, trainable=False)
		tvars = tf.trainable_variables()
		grads = tf.gradients(self.loss, tvars)
		optimizer = tf.train.AdamOptimizer(self.lr)
		self.train_op = optimizer.apply_gradients(zip(grads, tvars))

		""" model APIs """
		self.pred_pi,self.pred_mu,self.pred_sigma = self.output_preds(pkg)


	def output_preds(self,pkg):

		pred_pi = tf.reshape(pkg["output_pi_flat_dlt"],[self.batch_size,-1,self.n_y])
		pred_pi = tf.slice(pred_pi,[0,0,0],[self.batch_size,self.max_seq_len-self.lstep,self.n_y])

		pred_mu = tf.reshape(pkg["output_mu_flat_dlt"],[self.batch_size,-1,self.n_x])
		pred_mu = tf.slice(pred_mu,[0,0,0],[self.batch_size,self.max_seq_len-self.lstep,self.n_x])

		pred_sigma = tf.reshape(pkg["output_sigma_flat_dlt"],[self.batch_size,-1,self.n_x])
		pred_sigma = tf.slice(pred_sigma,[0,0,0],[self.batch_size,self.max_seq_len-self.lstep,self.n_x])

		return pred_pi,pred_mu,pred_sigma

	""" calculate loss """
	def calculate_loss(self,pkg):
		with tf.variable_scope("loss_entropy"):
			# cross_entropy_vec = tf.nn.softmax_cross_entropy_with_logits_v2(labels=pkg["target_dsc_flat"], logits=pkg["output_pi_flat"])
			cross_entropy_vec = tf.nn.softmax_cross_entropy_with_logits(labels=pkg["target_dsc_flat"],logits=pkg["output_pi_flat"])
		with tf.variable_scope("normal"):
			normal_loss_vec = 0.5 * tf.reduce_sum(
				tf.div(tf.square(tf.subtract(pkg["target_num_flat"],pkg["output_mu_flat"])), tf.maximum(1e-10,tf.square(pkg["output_sigma_flat"])))
				+ tf.log(2 * np.pi * tf.maximum(1e-10,tf.square(pkg["output_sigma_flat"]))), axis=1)
		cost = cross_entropy_vec + normal_loss_vec * self.feature_weight
		cost = 1. * tf.reduce_sum(cost * pkg["mask_flat"]) / tf.reduce_sum(pkg["mask_flat"])
		return cost

	

	def outputs(self,new_h,n_x,n_y):
		
		with tf.variable_scope("output_pi"):
			output_pi_flat = tf.layers.dense(inputs=new_h,units=n_y,activation=tf.nn.softplus)
			output_pi_flat = tf.nn.softmax(logits = output_pi_flat, axis = 1)
		with tf.variable_scope("output_mu"):
			output_mu_flat = tf.layers.dense(inputs=new_h,units=n_x,activation=None)
		with tf.variable_scope("output_sigma"):
			output_sigma_flat = tf.layers.dense(inputs=new_h,units=n_x,activation=tf.nn.softplus)

		return output_pi_flat,output_mu_flat,output_sigma_flat




