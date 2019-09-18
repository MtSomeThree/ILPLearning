import numpy as np
import torch
import gurobipy as grb
from scipy.linalg import null_space

if __name__ == '__main__':

	train_num = 9000

	data = np.array(torch.load('labels.pt')).reshape(-1, 729)
	query = np.array(torch.load('features.pt')).reshape(-1, 729)

	arc = range(729)

	model = grb.Model('Sudoku')
	z = model.addVars(arc, vtype=grb.GRB.BINARY, name='z')

	model.setParam('MIPGap', 0.0)
	model.setParam('OutputFlag', 0)
	model.setParam('TimeLimit', 60.0)

	ones = np.ones(train_num)
	data_1 = np.c_[data[:train_num, :], ones]
	kernel = null_space(data_1)

	M = kernel.shape[1]

	for i in range(M):
		w = dict()
		for j in range(729):
			w[j] = kernel[j][i]
		tmp = -kernel[729][i]
		model.addConstr(sum(z[k] * w[k] for k in arc) >= tmp - 1e-3, "eq%d_lower"%(i))
		model.addConstr(sum(z[k] * w[k] for k in arc) <= tmp + 1e-3, "eq%d_lower"%(i))

	correct = 0
	for i in range(train_num, 10000):
		model.setObjective(-sum(z[j] * query[i][j] for j in arc))
		model.optimize()

		solution = np.zeros(729)
		for v in model.getVars():
			if v.varName[0] == 'z':
				index = eval(v.varName[1:])
				solution[index] = v.x

		flag = True
		for j in range(729):
			if solution[j] != data[i, j]:
				flag = False
				break
		if flag:
			correct += 1
		if i % 20 == 0:
			print ("%d Test Cases Processed!, now acc is %f (%d/%d)."%(i - train_num + 1, float(correct) / float(i - train_num + 1), correct, i - train_num + 1))
		#print (query[i].sum(), solution.sum(), data[i].sum())
	print (correct)


