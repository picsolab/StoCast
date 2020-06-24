import numpy as np


def split_data(k,kfold,data,dataConfig,max_len):
	"""
	Split the input data into train and validation parts in k-th fold
	
	Arguments
		k: int
			the current slice identifier in fold cross validation
		
		kfold: int
			the number of folds in cross validation
		
		data: dictionary {"demo":demo,"dync":dync}
			the input data

		dataConfig: dictionary
			data configuration
	
	Returns
		trainining data fetching object
		
		validation data fetching object

	"""
	n = len(data['demo'])
	index = range(n)

	valid_idx = [i for i in index if i % kfold == k]
	train_idx = [i for i in index if i not in valid_idx]
	
	train_demo = [u for idx,u in enumerate(data['demo']) if idx in train_idx]
	train_dync = [u for idx,u in enumerate(data['dync']) if idx in train_idx]

	valid_demo = [u for idx,u in enumerate(data['demo']) if idx in valid_idx]
	valid_dync = [u for idx,u in enumerate(data['dync']) if idx in valid_idx]

	train_ds = DataSet(dataConfig,train_demo,train_dync,max_len)
	valid_ds = DataSet(dataConfig,valid_demo,valid_dync,max_len)	

	return train_ds,valid_ds


class DataSet(object):
	"""
	Create a DataSet class for next batch fetching

	"""
	def __init__(self,dataConfig,demo,dync,max_len):
		"""
		Args:
			dataConfig
				data configuration

			demo: a list of data frames
				demographics information
			
			dync: a list of data frames
				EHR data frame
		"""

		self.demo = demo
		self.dync = dync

		self.dataConfig = dataConfig

		self.num_examples = len(dync)
		self.seqlens = [u.shape[0] for u in self.dync]
		self.max_seq_len = max_len
		self.input_x_vars = dataConfig['input_x_vars']
		self.input_y_vars = dataConfig['input_y_vars']
		self.input_dt_vars = dataConfig['input_dt_vars']
		self.n_demo = len(dataConfig["demo_vars"])
		self.n_x = len(dataConfig["input_x_vars"])
		self.n_y = len(dataConfig["input_y_vars"])
		
		# for next_batch function
		self.epochs_completed = 0
		self.index_in_epoch = 0

	def next_batch(self,batch_size,shuffle=False):
		"""
		Return:

		"""
		start = self.index_in_epoch
		
		# very first beginning
		if self.epochs_completed==0 and start==0 and shuffle:
			idx = np.arange(self.num_examples)
			np.random.shuffle(idx)
			self.dync = [self.dync[i] for i in idx]
			self.demo = [self.demo[i] for i in idx]
		if start + batch_size > self.num_examples:
			self.epochs_completed += 1
			rest_num_examples = self.num_examples - start
			dync_rest_part = self.dync[start:self.num_examples]
			demo_rest_part = self.demo[start:self.num_examples]
			if shuffle:
				idx = np.arange(self.num_examples)
				np.random.shuffle(idx)
				self.dync = [self.dync[i] for i in idx]
				self.demo = [self.demo[i] for i in idx]
			
			start = 0
			self.index_in_epoch = batch_size - rest_num_examples
			end = self.index_in_epoch
			dync_new_part = self.dync[start:end]
			demo_new_part = self.demo[start:end]
			dync = dync_rest_part + dync_new_part
			demo = demo_rest_part + demo_new_part
			return self.padZeros(demo,dync)
		else:
			self.index_in_epoch += batch_size
			end = self.index_in_epoch
			dync = self.dync[start:end]
			demo = self.demo[start:end]
			return self.padZeros(demo,dync)
	
	def padZeros(self,demo,dync):
		"""
		Args:
			demo & dync are sliced data frames of length `batch_size`
		Return:
			a wrapper including all necessary arrays
		"""
		
		n_samples = len(dync)

		input_demo = np.zeros((n_samples, self.n_demo))
		input_x = np.zeros((n_samples, self.max_seq_len, self.n_x))
		input_y = np.zeros((n_samples, self.max_seq_len, self.n_y))
		input_dt = np.zeros((n_samples, self.max_seq_len, 1))

		mask = np.zeros((n_samples, self.max_seq_len))
		
		seqlens = np.array([u.shape[0] for u in dync])
		
		for i in range(n_samples):
			
			input_demo[i] = demo[i]
			step = dync[i].shape[0]
			input_x[i,:step,:] = dync[i][self.input_x_vars]
			input_y[i,:step,:] = dync[i][self.input_y_vars]
			input_dt[i,:step,:] = dync[i][self.input_dt_vars]

			mask[i,:step] = 1.0
		
		wrapper = {'input_demo': input_demo,
				   'input_x': input_x,
				   'input_y': input_y,
				   'input_dt': input_dt,
				   'seqlens': seqlens,
				   'mask': mask}
		
		return wrapper