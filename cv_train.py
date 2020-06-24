import numpy as np
import random
import tensorflow as tf
import argparse
import time
from datetime import datetime
import os
import _pickle
import json
from datautils import *
from utils import *
from model import (stocast,rnn,retain,tlstm)


def cv_train(modelname,config,kfold,lstep,n_samples,feature_weight,
			 pkl_filename="data/adni.pkl",json_filename="conf/dataConfig.json"):

	"""
	Cross validation training process.
	
	Description:
	
	In each fold, we split data into train and validation parts.
	If validation loss has no improvement for continuous 10 epoches, the current fold is stop and we go
	to the next epoch. In each fold, the parameters would inherit from the saved `best-model` of the past fold. Thus, the first
	fold would take longer time, and the following folds would be faster. After cross validation is done, we load the best model, and apply it to test data.

	Arguments
	---------
		modelname: string
			the name of chosen model class
		config: dictionary
			the configuration of training
		kfold: int 
			the number of folds in cross validation
		lstep: int
			the value of step in forward prediction
		n_samples: int
			the number of samples to draw in testing stage
		pkl_filename: string
			the filename of pkl file storing "demo","dync","max_len"
		json_filename: string
			the filename of json file storing data configuration
	Returns
	-------
		model parameters saved in `save` folder
	"""

	"""load data and data config
	The data file *pkl contains three parts: {'demo','dync','max_len'}
		'demo' : a list of dataframes storing patients' demographics information
		'dync' : a list of dataframes storing patients' dynamic information, including continous features and diganosis
		'max_len' : int, the maximum lengths of patient sequence
	"""
	adni = _pickle.load(open(pkl_filename,"rb"))
	dataConfig = json.load(open(json_filename))
	max_len = adni["max_len"]
	
	"""
	record val_loss change / trends;
	the smallest loss value for each fold is given by the best model
	"""
	loss_curve = {}

	# build model graph
	if modelname == "rnn":
		model = rnn(len(dataConfig["demo_vars"]),len(dataConfig["input_x_vars"]),len(dataConfig["input_y_vars"]),
							   max_len,config["batch_size"],config["n_h"],config["n_z"],lstep,feature_weight)
	elif modelname == "stocast":
		model = stocast(len(dataConfig["demo_vars"]),len(dataConfig["input_x_vars"]),len(dataConfig["input_y_vars"]),
						max_len,config["batch_size"],config["n_h"],config["n_z"],lstep,feature_weight)
	elif modelname == "storn":
		model = storn(len(dataConfig["demo_vars"]),len(dataConfig["input_x_vars"]),len(dataConfig["input_y_vars"]),
						max_len,config["batch_size"],config["n_h"],config["n_z"],lstep,feature_weight)
	elif modelname == "retain":
		model = retain(len(dataConfig["demo_vars"]),len(dataConfig["input_x_vars"]),len(dataConfig["input_y_vars"]),
						max_len,config["batch_size"],config["n_h"],config["n_z"],lstep,feature_weight)
	elif modelname == "tlstm":
		model = tlstm(len(dataConfig["demo_vars"]),len(dataConfig["input_x_vars"]),len(dataConfig["input_y_vars"]),
						max_len,config["batch_size"],config["n_h"],config["n_z"],lstep,feature_weight)



	# saving ...
	dirname = "save/{} {}".format(modelname,time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())))
	if not os.path.exists(dirname):
		os.makedirs(dirname)

	with tf.Session() as sess:
		summary_writer = tf.summary.FileWriter('logs/' + datetime.now().isoformat().replace(':', '-'), sess.graph)
		merged = tf.summary.merge_all()
		saver = tf.train.Saver(tf.global_variables(),max_to_keep=1)

		start = time.time()
		totaltime = 0
		for k in range(kfold):
			
			""" if k = 0, random initialization params """
			""" else, inherit previous best model's params """
			if k == 0:
				tf.global_variables_initializer().run()
			else:
				ckpt = tf.train.get_checkpoint_state(dirname)
				saver.restore(sess, ckpt.model_checkpoint_path)

			# split into trainining, validation, testing data
			train_ds,valid_ds = split_data(k,kfold,adni,dataConfig,max_len)

			# train
			minVlloss = 1e10
			loss_curve[k] = []
			n_batchs = int(train_ds.num_examples / config["batch_size"])

			# each epoch
			no_improvement = 0 # number of no improvements
			e = 0
			while e < config["num_epochs"]:
				sess.run(tf.assign(model.lr, config["learning_rate"] * (config["decay_rate"] ** e)))

				for b in range(n_batchs):
				
					wrap = train_ds.next_batch(config["batch_size"])
					
					feed = {model.input_demo: wrap['input_demo'],
							model.input_x: wrap['input_x'],
							model.input_y: wrap['input_y'],
							model.input_dt: wrap['input_dt'],
							model.seqlens: wrap['seqlens'],
							model.mask: wrap['mask']}
				
					_,loss,summary = sess.run([model.train_op, model.loss, merged],feed)
					summary_writer.add_summary(summary, e * n_batchs + b)

				# validation
				vloss = val_loss(sess,model,valid_ds,config["batch_size"])
				loss_curve[k].append(vloss)
				
				print("  |- FOLD:%d, EPOCH:%d, VLOSS:%.4f" % (k,e,vloss))

				if minVlloss > vloss:
					minVlloss = vloss
					checkpoint_path = os.path.join(dirname, "best_model_k={}_e={}.ckpt".format(k,e))
					saver.save(sess, checkpoint_path, global_step=e * n_batchs + k * config["num_epochs"] * n_batchs)
					print("  |- Best model saved to {}".format(checkpoint_path))
					no_improvement = 0
				else:
					no_improvement += 1
				
				# if the number of improvement reaches 10, stop running
				if no_improvement < 10:
					e += 1
					continue
				else:
					break

			end = time.time()
			print("|- %2d fold costs %.4f seconds.\n" % (k,end-start))
			totaltime += end-start
			start = time.time()
		print("Total train time is %.4f seconds." % totaltime)

		# testing
		print("Starting testing")
		ckpt = tf.train.get_checkpoint_state(dirname)
		if ckpt:
			saver.restore(sess, ckpt.model_checkpoint_path)
			print("Loading model: ",ckpt.model_checkpoint_path)
		
		test_ds = DataSet(dataConfig,adni["demo"],adni["dync"],max_len)
		test_res = test(sess,model,modelname,test_ds,
			config["batch_size"],max_len,dataConfig["input_y_vars"],lstep,n_samples=n_samples)
		
		print("Saving test results...")
		"""The results are saved in the following format.
		res = pickle.load(open(filename,'rb'))
			res : a list of dicts, with each dict() stores the prediction results corresponding to a specific patient
			res[i] : the dict for patient i, {'curr_labels','target_labels','pred_pi','target_features','pred_mu'}
				'curr_labels' : a list of labels
				'target_labels' : a list of target labels
				'pred_pi' : a list of predictions, the length is timesteps
					- pred_pi[t] is a 1d array for deterministic methods, or a 2d array for stocast with size (n_samples, 3)
				'target_features' : list, the length is timesteps
					- target_features[t] : a 1d array
				'pred_mu' : list, the length is timesteps
					- pred_mu[t] : a 1d array for deterministic methods, or a 2d array for stocast
		"""

		dirname = "result_fw={}/{}".format(feature_weight,modelname)
		if not os.path.exists(dirname):
			os.makedirs(dirname)
		_pickle.dump(test_res,open(os.path.join(dirname,"lstep{}_nsamples{}_result.pkl".format(lstep,n_samples)),"wb"))
		_pickle.dump(loss_curve,open(os.path.join(dirname,"lstep{}_nsamples{}_losses.pkl".format(lstep,n_samples)),"wb"))

####################################################################################

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--model', type=str, default="stocast", help='model')
	parser.add_argument('--k', type=int, default=10, help='how many K-fold')
	parser.add_argument('--l', type=int, default=1, help='l-step forward prediction')
	parser.add_argument('--n_samples', type=int, default=100, help="specify how many samples to draw")
	parser.add_argument('--fw', type=float, default=1.0, help="specify feature weight in loss")
	args = parser.parse_args()

	config = json.load(open("config.json"))

	cv_train(modelname = args.model, config = config, kfold = args.k, lstep = args.l, n_samples=args.n_samples, feature_weight=args.fw)





