import tensorflow as tf
import numpy as np


class retain():

	def __init__(self, n_demo, n_x, n_y, max_seq_len, batch_size, n_h, n_z, lstep, feature_weight):


		self.n_x = n_x
		self.n_y = n_y
		self.max_seq_len = max_seq_len
		self.batch_size = batch_size
		self.n_h = n_h
		self.n_z = n_z
		self.lstep = lstep
		self.feature_weight = feature_weight

		""" input placeholder """
		self.input_demo = tf.placeholder(dtype=tf.float32, shape=[batch_size, n_demo], name="input_demographics")
		self.input_x = tf.placeholder(dtype=tf.float32, shape=[batch_size,max_seq_len,n_x], name="input_numeric_data")
		self.input_y = tf.placeholder(dtype=tf.float32, shape=[batch_size,max_seq_len,n_y], name="input_discrete_data")
		self.input_dt = tf.placeholder(dtype=tf.float32, shape=[batch_size, max_seq_len, 1], name="input_dt")
		self.seqlens = tf.placeholder(dtype=tf.int32, shape=[batch_size], name="sequence_lengths")
		self.mask = tf.placeholder(dtype=tf.float32, shape=[batch_size, max_seq_len], name="mask")
		

		""" cell type """
		self.cell_alpha = tf.contrib.rnn.BasicLSTMCell(n_h,state_is_tuple=True)
		self.cell_beta = tf.contrib.rnn.BasicLSTMCell(n_h,state_is_tuple=True)

		""" initial_c, initial_h """
		self.initial_h = tf.layers.dense(inputs=self.input_demo, units=n_h, activation=tf.sigmoid)
		self.initial_c = self.initial_h

		""" inputs """
		with tf.variable_scope("input_num"):
			input_num = tf.slice(self.input_x,[0,0,0],[batch_size,max_seq_len-1,n_x])
			input_num = tf.unstack(input_num,num=max_seq_len-1,axis=1)
		with tf.variable_scope("input_dsc"):
			input_dsc = tf.slice(self.input_y,[0,0,0],[batch_size,max_seq_len-1,n_y])
			input_dsc = tf.layers.dense(inputs=tf.reshape(input_dsc,[batch_size * (max_seq_len-1),-1]),units=n_y,activation=tf.sigmoid)
			input_dsc = tf.reshape(input_dsc,[batch_size,max_seq_len-1,-1])
			input_dsc = tf.unstack(input_dsc,num=max_seq_len-1,axis=1)

		inputs = [tf.concat([u,v],1) for u,v in zip(input_num,input_dsc)]

		""" attention step """
		def attentionStep(att_t):
			# slice and reverse input [att_t x batch_size x n_x]
			att_inputs = inputs[:att_t][::-1]
			
			with tf.variable_scope("RNN_alpha"):
				att_outputs_alpha,_ = tf.contrib.rnn.static_rnn(self.cell_alpha,att_inputs,
					initial_state=tf.nn.rnn_cell.LSTMStateTuple(self.initial_c,self.initial_h),dtype=tf.float32)
				
				att_outputs_alpha = tf.stack(att_outputs_alpha,axis=0) # [att_t x batch_size x n_h]
				att_outputs_alpha = tf.reshape(att_outputs_alpha,[att_t * batch_size, -1])
				alpha = tf.layers.dense(inputs=att_outputs_alpha,units=1,activation=None)
				alpha = tf.reshape(alpha,[att_t, batch_size])
				alpha = tf.nn.softmax(alpha, dim=0) # [att_t x batch_size] dim/axis
			
			with tf.variable_scope("RNN_beta"):
				att_outputs_beta,_ = tf.contrib.rnn.static_rnn(self.cell_beta,att_inputs,
					initial_state=tf.nn.rnn_cell.LSTMStateTuple(self.initial_c,self.initial_h),dtype=tf.float32)
				
				att_outputs_beta = tf.stack(att_outputs_beta,axis=0) # [att_t x batch_size x n_h]
				att_outputs_beta = tf.reshape(att_outputs_beta,[att_t * batch_size, -1])
				beta = tf.layers.dense(inputs=att_outputs_beta,units=(n_x+n_y),activation=tf.nn.tanh)
				beta = tf.reshape(beta,[att_t,batch_size,-1]) # [att_t x batch_size x n_x]

			c_t = tf.reduce_sum(alpha[:,:,None] * beta * att_inputs,axis=0)
			return c_t # [batch_size x n_x]

		""" rnn RETAIN """
		with tf.variable_scope("RETAIN") as varscope:
			cc_t = []
			steps = np.arange(max_seq_len-1) + 1
		
			for i,u in enumerate(steps):
				if i > 0:
					varscope.reuse_variables()
				c_t = attentionStep(u)
				cc_t.append(c_t)
		
		cc_t = tf.stack(cc_t,axis=0)
		cc_t = tf.transpose(cc_t,perm=[1,0,2])
		cc_t = tf.reshape(cc_t,[batch_size * (max_seq_len-1), -1])
		
		with tf.variable_scope("output_mu"):
			output_mu_flat = tf.layers.dense(inputs=cc_t,units=n_x,activation=None)
		with tf.variable_scope("output_sigma"):
			output_sigma_flat = tf.layers.dense(inputs=cc_t,units=n_x,activation=tf.nn.softplus)
		with tf.variable_scope("output_pi"):
			output_pi_flat = tf.layers.dense(inputs=cc_t,units=n_y,activation=tf.nn.softplus)
			output_pi_flat = tf.nn.softmax(logits = output_pi_flat, axis = 1)

		""" target flat """
		with tf.variable_scope("target_num"):
			target_num_flat = tf.reshape(tf.slice(self.input_x,[0,1,0],[batch_size,max_seq_len-1,n_x]),[batch_size * (max_seq_len-1), -1])
		with tf.variable_scope("target_dsc"):
			target_dsc_flat = tf.reshape(tf.slice(self.input_y,[0,1,0],[batch_size,max_seq_len-1,n_y]),[batch_size * (max_seq_len-1), -1])
		
		""" in accordence to target """
		mask_flat = tf.reshape(tf.slice(self.mask,[0,1],[batch_size,max_seq_len-1]),[-1])

		pkg = {"target_num_flat": target_num_flat,
				"target_dsc_flat": target_dsc_flat,
				"output_mu_flat": output_mu_flat,
				"output_sigma_flat": output_sigma_flat,
				"output_pi_flat": output_pi_flat,
				"mask_flat": mask_flat}
		
		""" loss and train """
		loss = self.calculate_loss(pkg)
		with tf.variable_scope("loss"):
			self.loss = loss
		tf.summary.scalar("loss", self.loss)
		self.lr = tf.Variable(0.0, trainable=False)
		tvars = tf.trainable_variables()
		grads = tf.gradients(self.loss, tvars)
		optimizer = tf.train.AdamOptimizer(self.lr)
		self.train_op = optimizer.apply_gradients(zip(grads, tvars))

		self.pred_pi,self.pred_mu,self.pred_sigma = self.output_preds(pkg)

	
	def output_preds(self,pkg):

		pred_pi = tf.reshape(pkg['output_pi_flat'],[self.batch_size,-1,self.n_y])
		pred_pi = tf.slice(pred_pi,[0,0,0],[self.batch_size,self.max_seq_len-self.lstep,self.n_y])

		pred_mu = tf.reshape(pkg['output_mu_flat'],[self.batch_size,-1,self.n_x])
		pred_mu = tf.slice(pred_mu,[0,0,0],[self.batch_size,self.max_seq_len-self.lstep,self.n_x])

		pred_sigma = tf.reshape(pkg['output_sigma_flat'],[self.batch_size,-1,self.n_x])
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
		loss = cross_entropy_vec + normal_loss_vec * self.feature_weight
		loss = 1. * tf.reduce_sum(loss * pkg["mask_flat"]) / tf.reduce_sum(pkg["mask_flat"])
		return loss


