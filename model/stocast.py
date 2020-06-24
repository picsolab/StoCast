import tensorflow as tf
import numpy as np
from modelutils import *


class stocastCell(tf.contrib.rnn.RNNCell):
	def __init__(self, n_x, n_y, n_h, n_z, batch_size, input_demo):
		self.n_x = n_x
		self.n_y = n_y
		self.n_h = n_h
		self.n_z = n_z
		
		self.n_phi_x = n_x # phi(x)
		self.n_phi_z = n_z # phi(z)
		self.n_phi_y = n_y # phi(y)
		
		self.n_prior_hidden = n_z
		self.n_enc_hidden = n_z
		self.n_dec_hidden = n_x
		
		self.batch_size = batch_size
		self.input_demo = input_demo

		self.lstm = tf.contrib.rnn.LSTMCell(self.n_h, state_is_tuple = True)

	@property
	def state_size(self):
		return (self.n_h,self.n_h)

	@property
	def output_size(self):
		return (self.n_z,self.n_z,self.n_z,self.n_z,self.n_x,self.n_x,self.n_y,self.n_z,self.n_x,self.n_x,self.n_y)


	def __call__(self, input, state, scope=None):

		""" start """
		with tf.variable_scope(scope or type(self).__name__):
			h, c = state

		# slice x and y from input
		y = tf.slice(input, [0,0], [self.batch_size,self.n_y])
		x = tf.slice(input, [0,self.n_y], [self.batch_size,self.n_x])
		dt = tf.slice(input, [0,self.n_y+self.n_x], [self.batch_size,1])
		dlt = tf.slice(input, [0,self.n_y+self.n_x+1], [self.batch_size,1])

		# phi(log(dt))
		with tf.variable_scope("log_elapse_time"):
			phi_log_dt = self.time_decay(dt)

		with tf.variable_scope("log_elapse_time", reuse=True):
			phi_log_dlt = self.time_decay(dlt)

		with tf.variable_scope("phi_demo"):
			phi_demo = tf.nn.sigmoid(linear(self.input_demo,self.n_h))

		# Prior
		with tf.variable_scope("Prior"):
			prior_z_mu,prior_z_sigma = self.prior(h,phi_demo,phi_log_dt)

		# Encoder
		with tf.variable_scope("Encoder"):
			enc_z_mu,enc_z_sigma = self.encode(x,y,h,phi_demo,phi_log_dt)

		# Sampling q(z|*)
		eps = tf.random_normal((self.batch_size, self.n_z),0.0,1.0,dtype=tf.float32)
		z = tf.add(enc_z_mu, tf.multiply(enc_z_sigma, eps))

		# Decoder
		with tf.variable_scope("Decoder_y"):
			dec_y_pi = self.decode_y(z,h,phi_demo,phi_log_dt)
		
		with tf.variable_scope("Decoder_x"):
			dec_x_mu,dec_x_sigma = self.decode_x(z,h,phi_demo,phi_log_dt)

		# Testing
		with tf.variable_scope("Prior", reuse=True):
			prior_z_mu_dlt,prior_z_sigma_dlt = self.prior(h,phi_demo,phi_log_dlt)
			prior_eps = tf.random_normal((self.batch_size,self.n_z),0.0,1.0,dtype=tf.float32)
			prior_z_dlt = tf.add(prior_z_mu_dlt, tf.multiply(prior_z_sigma_dlt, prior_eps))
		with tf.variable_scope("Decoder_y",reuse=True):
			nex_y_pi = self.decode_y(prior_z_dlt,h,phi_demo,phi_log_dlt) # after `dlt` days
			nex_y_pi = tf.nn.softmax(logits = nex_y_pi, axis = 1)
		with tf.variable_scope("Decoder_x",reuse=True):
			nex_x_mu,nex_x_sigma = self.decode_x(prior_z_dlt,h,phi_demo,phi_log_dlt)

		with tf.variable_scope("h_update"):
			state_updated = self.update(x,y,z,phi_log_dt,state)

		return (prior_z_mu,prior_z_sigma,
				enc_z_mu,enc_z_sigma,
				dec_x_mu,dec_x_sigma,dec_y_pi,z,
				nex_x_mu,nex_x_sigma,nex_y_pi),state_updated

	def time_decay(self,dt):
		log_dt = tf.map_fn(lambda x: tf.div(1.0, tf.log(tf.exp(1.0) + x)), dt)
		phi_log_dt = linear(log_dt, 1)
		return phi_log_dt

	def prior(self,h,phi_demo,phi_log_dt):
		with tf.variable_scope("hidden"):
			prior_hidden = tf.nn.relu(linear(tf.concat(values=(h,phi_demo,phi_log_dt),axis=1), self.n_prior_hidden))
		with tf.variable_scope("mu"):
			prior_z_mu = linear(prior_hidden, self.n_z) # linear
		with tf.variable_scope("sigma"):
			prior_z_sigma = tf.nn.softplus(linear(prior_hidden, self.n_z))
		return prior_z_mu,prior_z_sigma

	def encode(self,x,y,h,phi_demo,phi_log_dt):
		with tf.variable_scope("phi_x"):
			phi_x = tf.nn.relu(linear(x,self.n_phi_x))
		with tf.variable_scope("phi_y"):
			phi_y = tf.nn.relu(linear(y,self.n_phi_y))
		with tf.variable_scope("hidden"):
			enc_hidden_z = tf.nn.relu(linear(tf.concat(values=(phi_x,phi_y,h,phi_demo,phi_log_dt),axis=1),self.n_enc_hidden))
		with tf.variable_scope("mu"):
			enc_z_mu = linear(enc_hidden_z,self.n_z)
		with tf.variable_scope("sigma"):
			enc_z_sigma = tf.nn.softplus(linear(enc_hidden_z,self.n_z))
		return enc_z_mu,enc_z_sigma

	def decode_y(self,z,h,phi_demo,phi_log_dt):
		# layer 1
		with tf.variable_scope("phi_z"):
			phi_z = tf.nn.relu(linear(z,self.n_phi_z))
		# layer 2: skip connections
		with tf.variable_scope("hidden_y"):
			dec_hidden_y = tf.nn.relu(linear(tf.concat(values=(phi_z,z,h,phi_demo,phi_log_dt),axis=1),self.n_y))
		# output layer: skip connections
		with tf.variable_scope("pi"):
			dec_y_pi = tf.nn.softplus(linear(tf.concat(values=(dec_hidden_y,z),axis=1),self.n_y))

		return dec_y_pi

	def decode_x(self,z,h,phi_demo,phi_log_dt):
		# layer 1
		with tf.variable_scope("phi_z"):
			phi_z = tf.nn.relu(linear(z,self.n_phi_z))
		# layer 2: skip connections
		with tf.variable_scope("hidden_x"):
			dec_hidden_x = tf.nn.relu(linear(tf.concat(values=(phi_z,z,h,phi_demo,phi_log_dt),axis=1),self.n_dec_hidden))
		# output layer: skip connections
		with tf.variable_scope("mu"):
			dec_x_mu = linear(tf.concat(values=(dec_hidden_x,z),axis=1),self.n_x)
		# output layer: skip connections
		with tf.variable_scope("sigma"):
			dec_x_sigma = tf.nn.softplus(linear(tf.concat(values=(dec_hidden_x,z),axis=1),self.n_x))
		return dec_x_mu,dec_x_sigma

	def update(self,x,y,z,phi_log_dt,state):
		with tf.variable_scope("phi_z"):
			phi_z = tf.nn.relu(linear(z, self.n_phi_z))
		with tf.variable_scope("phi_x"):
			phi_x = tf.nn.relu(linear(x, self.n_phi_x))
		with tf.variable_scope("phi_y"):
			phi_y = tf.nn.relu(linear(y, self.n_phi_y))
		_,state_updated = self.lstm(tf.concat(values=(phi_x,phi_y,phi_z,phi_log_dt), axis=1), state)
		return state_updated

class stocast():
	
	def __init__(self, n_demo, n_x, n_y, max_seq_len, batch_size, n_h, n_z, lstep, feature_weight):

		self.n_x = n_x
		self.n_y = n_y
		self.max_seq_len = max_seq_len
		self.batch_size = batch_size
		self.n_h = n_h
		self.n_z = n_z
		self.lstep = lstep
		self.feature_weight = feature_weight
		
		""" input placeholders """
		self.input_demo = tf.placeholder(dtype=tf.float32, shape=[batch_size, n_demo], name="input_demo")
		self.input_x = tf.placeholder(dtype=tf.float32, shape=[batch_size, max_seq_len, n_x], name="input_x")
		self.input_y = tf.placeholder(dtype=tf.float32, shape=[batch_size, max_seq_len, n_y], name="input_y")
		self.input_dt = tf.placeholder(dtype=tf.float32, shape=[batch_size, max_seq_len, 1], name="input_dt")
		self.seqlens = tf.placeholder(dtype=tf.int32, shape=[batch_size], name="sequence_lengths")
		self.mask = tf.placeholder(dtype=tf.float32, shape=[batch_size, max_seq_len], name="mask")

		""" SPVRNN """
		self.cell = stocastCell(n_x, n_y, n_h, n_z, batch_size, self.input_demo)
		
		""" concatenate y,dt,x to generate input """
		with tf.variable_scope("input_x"):
			input_x = tf.unstack(self.input_x,num=max_seq_len,axis=1)
		with tf.variable_scope("input_y"):
			input_y = tf.unstack(self.input_y,num=max_seq_len,axis=1)
		with tf.variable_scope("input_dt"):
			input_dt = tf.unstack(self.input_dt,num=max_seq_len,axis=1)
		with tf.variable_scope("input_dlt"):
			input_dlt = get_dlt(self.input_dt,lstep)
			input_dlt = tf.unstack(input_dlt,num=max_seq_len,axis=1)

		
		
		inputs = [tf.concat(values=(y,x,dt,dlt),axis=1) for y,x,dt,dlt in zip(input_y,input_x,input_dt,input_dlt)]

		""" run """
		initial_state_c,initial_state_h = self.cell.zero_state(batch_size=batch_size, dtype=tf.float32)
		outputs, last_state = tf.contrib.rnn.static_rnn(self.cell, inputs,
			initial_state=tf.nn.rnn_cell.LSTMStateTuple(initial_state_c,initial_state_h),
			dtype=tf.float32, sequence_length=self.seqlens)

		""" outputs """
		names = ["prior_z_mu_flat",
				"prior_z_sigma_flat",
				"enc_z_mu_flat", 
				"enc_z_sigma_flat",
				"dec_x_mu_flat", 
				"dec_x_sigma_flat", 
				"dec_y_pi_flat", 
				"z_flat",
				"nex_x_mu_flat", 
				"nex_x_sigma_flat", 
				"nex_y_pi_flat"]
		
		pkg = {}
		for i,name in enumerate(names):
			with tf.variable_scope(name):
				tensor_of_name = tf.stack([output_k[i] for output_k in outputs])
				tensor_of_name = tf.transpose(tensor_of_name,[1,0,2])
				tensor_of_name = tf.reshape(tensor_of_name,[batch_size * max_seq_len, -1])
				pkg[name] = tensor_of_name
		
		pkg["x_flat"] = tf.reshape(self.input_x, [-1, n_x])
		pkg["y_flat"] = tf.reshape(self.input_y, [-1, n_y])
		pkg["mask_flat"] = tf.reshape(self.mask, [-1])
		
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

		""" predicted results """
		self.pred_pi,self.pred_mu,self.pred_sigma = self.output_preds(pkg)

	"""loss function"""
	def calculate_loss(self,pkg):
		# log[p(z|h)]
		log_prior_z = tf_normal(pkg["z_flat"],pkg["prior_z_mu_flat"],pkg["prior_z_sigma_flat"])
		# log[p(z|x,y,h)]
		log_post_z = tf_normal(pkg["z_flat"],pkg["enc_z_mu_flat"],pkg["enc_z_sigma_flat"])
		# log[p(x,y|z,h)]
		log_lik_x = tf_normal(pkg["x_flat"],pkg["dec_x_mu_flat"],pkg["dec_x_sigma_flat"])
		log_lik_y = tf_cross_entropy(labels=pkg["y_flat"],logits=pkg["dec_y_pi_flat"])
		neg_log_lik = log_prior_z - log_post_z + log_lik_x * self.feature_weight + log_lik_y
		loss = pkg["mask_flat"] * neg_log_lik
		loss = 1. * tf.reduce_sum(loss) / tf.reduce_sum(pkg["mask_flat"])
		return loss

	def output_preds(self,pkg):
		"""
		Outputs predicted results
			
			{pred_pi} [batch_size, len, n_y]: float
				the predicted pi at a target future time step
			
			{pred_mu} [batch_size, len, n_x]: float
				the predicted feature means
			
			{pred_sigma} [batch_size, len, n_x]: float
				the predicted feature deviations
		
		If lstep==4, the t,dt and dlt are as follows. So curr_labels should be sliced from t=0, target_labels
		should be sliced from t=4. The predicted results should be sliced from t=1 because the unit at t=1 gives
		prediction results for t=4.
		
		t  :[ 0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17  18  19 ]
		dt :[ 0.  6.  5.  7.  6.  6.  6.  7.  5.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.]
		dlt:[18. 24. 24. 25. 25. 24. 18. 12.  5.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.]
		
		"""
		
		# ground truth

		# curr_labels = tf.slice(self.input_y,[0,0,0],[self.batch_size, self.max_seq_len - self.lstep, self.n_y])
		# curr_labels = tf.argmax(curr_labels, axis=1)
		# target_labels = tf.slice(self.input_y,[0,self.lstep,0],[self.batch_size, self.max_seq_len - self.lstep, self.n_y])
		# target_labels = tf.argmax(target_labels, axis=1)
		# target_features = tf.slice(self.input_x,[0,self.lstep,0],[self.batch_size,self.max_seq_len - self.lstep,self.n_x])
		# output_mask = tf.slice(self.mask,[0,self.lstep],[self.batch_size,self.max_seq_len-self.lstep])

		pred_pi = tf.reshape(pkg["nex_y_pi_flat"],[self.batch_size,self.max_seq_len,-1])
		pred_pi = tf.slice(pred_pi,[0,1,0],[self.batch_size,self.max_seq_len - self.lstep,self.n_y])
		
		pred_mu = tf.reshape(pkg["nex_x_mu_flat"],[self.batch_size,self.max_seq_len,self.n_x])
		pred_mu = tf.slice(pred_mu,[0,1,0],[self.batch_size,self.max_seq_len - self.lstep,self.n_x])

		pred_sigma = tf.reshape(pkg["nex_x_sigma_flat"],[self.batch_size,self.max_seq_len,-1])
		pred_sigma = tf.slice(pred_sigma,[0,1,0],[self.batch_size,self.max_seq_len - self.lstep,self.n_x])

		return pred_pi,pred_mu,pred_sigma