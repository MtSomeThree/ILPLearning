import gurobipy as grb
import numpy as np
from scipy.linalg import null_space

def init_GRB(N, arc):
	m = grb.Model('mst')
	z = m.addVars(arc, vtype=grb.GRB.BINARY, name='z')

	m.addConstrs((z[i, j] == z[j, i] for i, j in arc), "sym")

	return m, z

def add_data_point(model, z, w, x, arc, cnt):
	tmp = sum([w[i, j] * x[i, j] for (i, j) in arc])
	#print (tmp)
	model.addConstr(sum(z[i, j] * w[i, j] for i, j in arc) >= tmp, "data%d"%(cnt))

def add_equation_const(model, z, zero_space, one_to_two, dim):
	theta = 1e-2
	num_const = zero_space.shape[1]
	for const_idx in range(num_const):
		w = dict()
		arc = set()
		for j in range(dim):
			idx1, idx2 = one_to_two[j]
			arc.add((idx1, idx2))
			w[(idx1, idx2)] = zero_space[j][const_idx]
		tmp = -zero_space[dim][const_idx]
		model.addConstr(sum(z[i, j] * w[i, j] for i, j in arc) >= tmp - theta, "eq%d_lower"%(i))
		model.addConstr(sum(z[i, j] * w[i, j] for i, j in arc) <= tmp + theta, "eq%d_upper"%(i))

def get_weight_matrix(data_file, N):
	cnt = 0
	weight = []
	for line in data_file:
		tmp = line.strip().split()
		tmp = [float(x) for x in tmp]
		weight.append(tmp)
		cnt += 1
		if cnt == N + 1:
			break
	return np.array(weight)

def get_solution_matrix(data_file, N):
	cnt = 0
	sol = []
	for line in data_file:
		tmp = line.strip().split()
		tmp = [int(x) for x in tmp]
		sol.append(tmp)
		cnt += 1
		if cnt == N + 1:
			break
	return np.array(sol)

def get_mapping(N):
	two_to_one = dict()
	one_to_two = dict()
	cnt = 0
	for i in range(N + 1):
		for j in range(i + 1, N + 1):
			two_to_one[i, j] = cnt
			one_to_two[cnt] = (i, j)
			cnt += 1
	return two_to_one, one_to_two, cnt


if __name__ == '__main__':	

	N = 6
	two_to_one, one_to_two, dim = get_mapping(N)

	train_file = open('train.txt', 'r')
	test_file = open('test.txt', 'r')
	out_file = open('result.txt', 'w')

	arc = set()
	for i in range(N + 1):
		for j in range(N + 1):
			arc.add((i, j))

	model, z = init_GRB(N, arc)
	model.setParam("MIPGap", 0.0)
	model.setParam("OutputFlag", 0)
	'''
	model.setParam("OptimalityTol", 1e-8)
	model.setParam("FeasibilityTol", 1e-8)
	model.setParam("IntFeasTol", 1e-8)
	print (model.Params.OptimalityTol)
	print (model.Params.FeasibilityTol)
	print (model.Params.IntFeasTol)
	print (model.Params.MarkowitzTol)
	'''
	
	matrix_w = []
	matrix_x = []

	cnt = 0
	while(True):
		w = get_weight_matrix(train_file, N)
		x = get_solution_matrix(train_file, N)
		if w.size == 0:
			break
		add_data_point(model, z, w, x, arc, cnt)
		cnt += 1
		vector_w = []
		vector_x = []
		for i in range(dim):
			idx1, idx2 = one_to_two[i]
			vector_w.append(w[idx1, idx2])
			vector_x.append(x[idx1, idx2])
		matrix_w.append(vector_w)
		matrix_x.append(vector_x)

	#set_size_evaluate(data_w, data_x)
	matrix_w = np.array(matrix_w)
	matrix_x = np.array(matrix_x)
	ones = np.ones(cnt)
	matrix_x = np.c_[matrix_x, ones]
	zero_space = null_space(matrix_x)
	add_equation_const(model, z, zero_space, one_to_two, dim)

	cnt = 0
	correct = 0
	while(True):
		w = get_weight_matrix(test_file, N)
		x = get_solution_matrix(test_file, N)
		if w.size == 0:
			break

		model.setObjective(sum(z[i, j] * w[i, j] for i, j in arc))
		model.optimize()

		solution = np.zeros((N + 1, N + 1))
		for v in model.getVars():
			if v.varName[0] == 'z':
				index = eval(v.varName[1:])
				solution[index[0], index[1]] = v.x

		loss = 0
		gold_obj = 0.0
		my_obj = 0.0
		for i in range(N + 1):
			for j in range(N + 1):
				gold_obj += x[i, j] * w[i, j]
				my_obj += solution[i, j] * w[i, j]
				if x[i, j] != solution[i, j]:
					loss += 1
					out_file.write("![%d/%d]! "%(x[i, j], solution[i, j]))
				else:
					out_file.write(" (%d/%d)  "%(x[i, j], solution[i, j]))
			out_file.write("\n")
		out_file.write("Test Case #%d, wrong variable: %d, gold_obj: %f, my_obj: %f\n"%(cnt, loss, gold_obj, my_obj))
		cnt += 1
		if loss == 0:
			correct += 1
		print ("Test Case #%d, wrong variable: %d"%(cnt, int(loss / 2)))
	print ("Exact Match Accuracy: %d%%(%d/%d)"%(int(correct * 100 / cnt), correct, cnt))
