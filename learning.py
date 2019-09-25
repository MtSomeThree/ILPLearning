import gurobipy as grb
import numpy as np
import argparse
import os
from scipy.linalg import null_space

def init_GRB(arc, task):
	m = grb.Model('mst')
	m.setParam("MIPGap", 0.0)
	m.setParam("OutputFlag", 0)
	z = m.addVars(arc, vtype=grb.GRB.BINARY, name='z')

	if task == 'MST':
		m.addConstrs((z[i, j] == z[j, i] for i, j in arc), "sym")

	return m, z

def add_data_point(model, z, w, x, arc, cnt, slack=0):
	tmp = sum([w[idx] * x[idx] for idx  in arc])
	#print (tmp)
	model.addConstr(sum(z[idx] * w[idx] for idx in arc) >= tmp - slack, "data%d"%(cnt))

def add_equation_const(model, z, zero_space, one_to_two, dim):
	theta = 1e-2
	num_const = zero_space.shape[1]
	for const_idx in range(num_const):
		w = dict()
		arc = set()
		for j in range(dim):
			idx = one_to_two[j]
			arc.add(idx)
			w[idx] = zero_space[j][const_idx]
		tmp = -zero_space[dim][const_idx]
		model.addConstr(sum(z[idx] * w[idx] for idx in arc) >= tmp - theta, "eq%d_lower"%(i))
		model.addConstr(sum(z[idx] * w[idx] for idx in arc) <= tmp + theta, "eq%d_upper"%(i))

def get_weight_matrix(data_file, N):
	cnt = 0
	weight = []
	for line in data_file:
		tmp = line.strip().split()
		tmp = [float(x) for x in tmp]
		weight.append(tmp)
		cnt += 1
		if cnt == N:
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
		if cnt == N:
			break
	return np.array(sol)

def get_mapping(N, task):
	two_to_one = dict()
	one_to_two = dict()
	if task == 'MST':
		cnt = 0
		for i in range(N):
			for j in range(i + 1, N):
				two_to_one[i, j] = cnt
				one_to_two[cnt] = (i, j)
				cnt += 1
	else:
		cnt = 0
		for i in range(N):
			for j in range(N):
				two_to_one[i, j] = cnt
				one_to_two[cnt] = (i, j)
				cnt += 1

	return two_to_one, one_to_two, cnt

def get_variable_arc(N, task):
	arc = set()
	for i in range(N):
		for j in range(N):
			arc.add((i, j))
	return arc

def get_loss(solution, w, x, out_file):
	loss = 0
	gold_obj = 0.0
	my_obj = 0.0
	for i in range(N):
		for j in range(N):
			gold_obj += x[i, j] * w[i, j]
			my_obj += solution[i, j] * w[i, j]
			if x[i, j] != solution[i, j]:
				loss += 1
				out_file.write("![%d/%d]! "%(x[i, j], solution[i, j]))
			else:
				out_file.write(" (%d/%d)  "%(x[i, j], solution[i, j]))
		out_file.write("\n")
	out_file.write("Test Case #%d, wrong variable: %d, gold_obj: %f, my_obj: %f\n"%(cnt, loss, gold_obj, my_obj))
	return loss / 2

def upper_bound_solve(model, z, w, x, arc, out_file):
	model.setObjective(sum(z[idx] * w[idx] for idx in arc))
	model.optimize()

	solution = np.zeros((N, N))
	for v in model.getVars():
		if v.varName[0] == 'z':
			index = eval(v.varName[1:])
			solution[index[0], index[1]] = v.x

	return get_loss(solution, w, x, out_file)

def lower_bound_solve(data_x, w, x, out_file):
	M = len(data_x)
	opt_obj = 1e+8
	opt_idx = -1
	for i in range(M):
		tmp = (w * data_x[i]).sum()
		if tmp < opt_obj:
			opt_obj = tmp
			opt_idx = i
	solution = data_x[opt_idx]

	return get_loss(solution, w, x, out_file)

if __name__ == '__main__':	
	parser = argparse.ArgumentParser()
	parser.add_argument("--N", type=int, default=7)
	parser.add_argument("--task", type=str, default='MST')
	parser.add_argument("--equation", action='store_true')
	parser.add_argument("--out_file", type=str, default='result.txt')
	parser.add_argument("--train_file", type=str, default='train.txt')
	parser.add_argument("--test_file", type=str, default='test.txt')
	args = parser.parse_args()

	N = args.N

	train_file = os.path.join('data', args.task, args.train_file)
	test_file = os.path.join('data', args.task, args.test_file)
	out_file = os.path.join('data', args.task, args.out_file)

	train_file = open(train_file, 'r')
	test_file = open(test_file, 'r')
	out_file = open(args.out_file, 'w')

	arc = get_variable_arc(N, args.task)
	two_to_one, one_to_two, dim = get_mapping(N, args.task)

	model, z = init_GRB(arc, args.task)
	'''
	model.setParam("OptimalityTol", 1e-8)
	model.setParam("FeasibilityTol", 1e-8)
	model.setParam("IntFeasTol", 1e-8)
	print (model.Params.OptimalityTol)
	'''
	
	matrix_w = []
	matrix_x = []
	data_x = []

	cnt = 0
	while(True):
		w = get_weight_matrix(train_file, N)
		x = get_solution_matrix(train_file, N)
		if w.size == 0:
			break
		add_data_point(model, z, w, x, arc, cnt)
		data_x.append(x)
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
	zero_space = null_space(np.c_[matrix_x, ones])
	if args.equation:
		add_equation_const(model, z, zero_space, one_to_two, dim)

	cnt = 0
	correct_u = 0
	correct_l = 0
	error_u = 0.0
	error_l = 0.0
	while(True):
		w = get_weight_matrix(test_file, N)
		x = get_solution_matrix(test_file, N)
		if w.size == 0:
			break
		loss_u = upper_bound_solve(model, z, w, x, arc, out_file)
		loss_l = lower_bound_solve(data_x, w, x, out_file)
		cnt += 1
		error_u += loss_u
		error_l += loss_l
		if loss_u == 0:
			correct_u += 1
		if loss_l == 0:
			correct_l += 1
		print ("Test Case #%d, upper, wrong variable: %d"%(cnt, int(loss_u / 2)))
		print ("Test Case #%d, lower, wrong variable: %d"%(cnt, int(loss_l / 2)))

	print ("Upper Bound Exact Match Accuracy: %d%%(%d/%d), average wrong: %f"%(int(correct_u * 100 / cnt), correct_u, cnt, error_u / float(cnt)))
	print ("Lower Bound Exact Match Accuracy: %d%%(%d/%d), average wrong: %f"%(int(correct_l * 100 / cnt), correct_l, cnt, error_l / float(cnt)))
