import torch as t

class LossWrapper(t.nn.Module):

	"""
	Class that wraps any pytorch loss allowing for ignore index. In the Matrix Factorization context it may be useful to define a value indicating missing values even when performing a regression, for example if the goal is to predict a sparsely observed real-valued matrix.
	"""

	def __init__(self, loss:t.nn.Module, type:str==None, ignore_index:int):
		"""
		Constructor for the wrapper.

		Parameters
		----------
		loss : t.nn.Module
			The argument can be any pytorch compatible loss functioni
		type:
			Specifies wheter is a regression or a binay prediction (deprecate?)
		ignore_index : int
			Specifies which value should be ignored while computing the loss, to allow for the presence of missing values in the matrix/relation.

		Returns
		-------
		"""
		super(LossWrapper, self).__init__()
		self.loss = loss
		self.ignore_index = ignore_index
		self.type = type
	
	def __call__(self, input, target):
		"""
		Function defining the forward pass for this wrapper. It implements the ignore_index filtering and then it calls the actual self.loss on the remaining values.

		Parameters
		----------
		input : t.nn.Tensor
			Pytorch tensor containing the predicted values
		target : t.nn.Tensor
			Pytorch tensor containing the target values

		Returns
		-------
		Loss score computed only for the target values that are not equal to self.ignore_index.
		"""
		input = input.view(-1)
		target = target.view(-1)
		if self.ignore_index != None:
			mask = target.ne(self.ignore_index)
			input = t.masked_select(input, mask)
			target = t.masked_select(target, mask)
		#t2 = time.time()
		
		r = self.loss(input, target)
		#t3 = time.time()
		#print t2-t1, t3-t2
		return r
