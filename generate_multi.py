import numpy as np

def solve(weights, TreeDepth, TreeWidth):
	'''
		structure: Tree, 3 - 3 - 3 - 3 - 3
	'''
	w = weights - 0.5
	solution = np.zeros(TreeDepth * TreeWidth)
	for i in range(TreeWidth):
		opt_sum = 0.0
		opt_idx = 0
		cur_sum = 0.0
		for j in range(TreeDepth):
			cur_sum += w[i * TreeDepth + j]
			if cur_sum > opt_sum:
				opt_sum, opt_idx = cur_sum, j + 1
		for j in range(opt_idx):
			solution[i * TreeDepth + j] = 1
	return solution

if __name__ == '__main__':
	number = 10000
	TreeDepth = 3
	TreeBranth = 5
	Dim = TreeDepth * TreeBranth
	data_x = []
	data_y = []
	for i in range(number):
		weights = np.random.rand(Dim)
		labels = solve(weights, TreeDepth, TreeBranth)
		data_x.append(weights)
		data_y.append(labels)
	np.save('./data/Multi/syn_train_w.npy', np.array(data_x))
	np.save('./data/Multi/syn_train_x.npy', np.array(data_y))

