import torch
from torch.autograd import Variable
import numpy as np
import random
import csv
from scipy.linalg import null_space
from NN_mst import myModel
from LatentLearning import *

def get_data(csv_file):
	data = csv.reader(csv_file)
	data_x = []
	data_y = []
	head = True
	for line in data:
		if head:
			head = False
			continue
		x = [float(i) for i in line[:-14]]
		y = [5 - len(i) for i in line[-14:]]
		data_x.append(x)
		data_y.append(y)
	return np.array(data_x), np.array(data_y)

def loader(data_x, data_y, batch_size):
	N = data_x.shape[0]
	ids = np.array(range(N))
	np.random.shuffle(ids)

	data_x = data_x[ids]
	data_y = data_y[ids]

	num_batch = int(N / batch_size)

	for i in range(num_batch):
		yield torch.Tensor(data_x[i * batch_size: (i + 1) * batch_size]), torch.Tensor(data_y[i * batch_size: (i + 1) * batch_size])

def ILP_solve_upper(model, z, w, x, arc, N, out_file):
	model.setObjective(sum(z[idx] * w[idx] for idx in arc))
	model.optimize()
	obj = model.getObjective()

	solution = np.zeros(N)
	for v in model.getVars():
		if v.varName[0] == 'z':
			index = eval(v.varName[1:])
			solution[index[0]] = v.x

	wrong = 0
	for i in range(N):
		if abs(solution[i] - x[i]) > 1e-6:
			wrong += 1

	return wrong, obj.getValue()

def ILP_solve_lower(data_x, w, x, N, out_file):
	M = len(data_x)
	opt_obj = 1e+8
	opt_idx = -1
	for i in range(M):
		tmp = (w * data_x[i]).sum()
		if tmp < opt_obj:
			opt_obj = tmp
			opt_idx = i
	solution = data_x[opt_idx]

	wrong = 0
	for i in range(N):
		if abs(solution[i] - x[i]) > 1e-6:
			wrong += 1

	return wrong, opt_obj

def label_hash(y):
	hash_value = 0
	for i in range(N):
		hash_value += y[i] * (1 << i)
	return hash_value

if __name__ == '__main__':
	DataDir = './data/Multi/yeast.csv'
	N = 96
	TrainSize = 10000
	out_file = open('./data/Multi/result_sync.txt', 'w')
	trainFlag = False
	latentFlag = True
	#data_x, data_y = get_data(open(DataDir, 'r'))
	data_x = np.load('./data/Multi/ImCLEF07A_x.npy')
	data_y = np.load('./data/Multi/ImCLEF07A_y.npy')

	label_set = set()
	indices_set = set()
	for i in range(TrainSize):
		label_set.add(label_hash(data_y[i]))
		for j in range(96):
			if data_y[i][j] < 0.5:
				continue
			for k in range(j + 1, 96):
				if data_y[i][k] > 0.5:
					indices_set.add((j, k)) 
	print (len(label_set))
	for i in range(TrainSize, data_x.shape[0]):
		label_set.add(label_hash(data_y[i]))
	print (len(label_set))

	arc = set()
	for i in range(N):
		arc.add(i)

	grb_model, z = init_GRB(arc, task='Multi')

	if latentFlag:
		latent_indices, order_to_idx, idx_to_order, latent_dim = get_latent_indices(N, task='Multi', indices_set=indices_set)
		print ("Latent Dimension=%d"%(latent_dim))
		print (latent_indices)
		latent_variables = add_latent_variables(grb_model, latent_indices)
		add_latent_predefined(grb_model, z, latent_variables, indices_set)

	model = myModel(80, 96)
	model.cuda()

	if trainFlag == True:

		criterion = torch.nn.BCELoss() 

		optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

		lossLog = []
		for epoch in range(100):
			total_loss = 0
			cnt = 0
			train_loader = loader(data_x[:TrainSize], data_y[:TrainSize], batch_size=50)
			for x, y in train_loader:
				x = Variable(x).cuda()
				y = Variable(y.float()).cuda()

				optimizer.zero_grad()
				outputs = model(x)
				loss = criterion(outputs, y)
				loss.backward()
				optimizer.step()
				cnt += 1
				total_loss += loss
			print ('epoch %d, loss %.8f'%(epoch, total_loss / cnt))
			lossLog.append(total_loss)
		torch.save(model.state_dict(), './model/Multi/3layers_ImCLEF07A.pt')
	else:
		model.load_state_dict(torch.load('./model/Multi/3layers_ImCLEF07A.pt'))

	model.eval()
	
	for i in range(TrainSize):
		x = Variable(torch.Tensor(data_x[i].reshape(1, -1))).cuda()
		y = data_y[i]
		w = model(x).cpu().detach().numpy().reshape(-1)
		w = 0.5 - w
		#w = 0.5 - data_x[i]

		#add_data_point(grb_model, z, w, y, arc, i)

	grb_model.setObjective(z[0])
	grb_model.optimize()
	constrs = grb_model.getConstrs()
	print (len(constrs))

	ones = np.ones(TrainSize)
	print (np.c_[data_y[:TrainSize], ones].shape)
	zero_space = null_space(np.c_[data_y[:TrainSize], ones])
	print (zero_space.shape)

	if latentFlag:
		data_latent = predefined_h_from_x(data_y[:TrainSize], indices_set, latent_dim)
		latent_zero_space = null_space(np.c_[data_latent, ones])
		add_equation_const(grb_model, latent_variables, latent_zero_space, order_to_idx, latent_dim)
		print (latent_zero_space.shape)
	
	cnt = 0
	correct = 0
	error = 0
	test_loader = loader(data_x[TrainSize:], data_y[TrainSize:], batch_size=1)
	for x, y in test_loader:
		wrong = 0

		x = Variable(x).cuda()
		y = y.long().squeeze()

		output = model(x).squeeze()
		prediction = np.where(output.cpu().detach().numpy() > 0.5, 1, 0) + 1e-3
		prediction = torch.LongTensor(prediction)
		wrong += (prediction != y).sum()

		#print ("Test Case %d, wrong variable: %d"%(cnt, wrong))
		cnt += 1
		if wrong == 0:
			correct += 1
		error += wrong
	print ("Exact Match Accuracy: %d%%(%d/%d), average wrong: %f(%d/%d)"%(int(correct * 100 / cnt), correct, cnt, float(error) / float(cnt), error, cnt))
	'''
	cnt = 0
	correct = 0
	error = 0
	test_loader = loader(data_x[:TrainSize], data_y[:TrainSize], batch_size=1)
	for x, y in test_loader:
		wrong = 0

		x = Variable(x).cuda()
		y = y.long().squeeze()

		output = model(x).squeeze()
		prediction = np.where(output.cpu().detach().numpy() > 0.5, 1, 0) + 1e-3
		prediction = torch.LongTensor(prediction)
		wrong += (prediction != y).sum()

		#print ("Test Case %d, wrong variable: %d"%(cnt, wrong))
		cnt += 1
		if wrong == 0:
			correct += 1
		error += wrong
	print ("Exact Match Accuacy: %d%%(%d/%d), average wrong: %f(%d/%d)"%(int(correct * 100 / cnt), correct, cnt, float(error) / float(cnt), error, cnt))
	'''

	cnt = 0
	correct_u = correct_l = correct_m = 0
	error_u = error_l = error_m = 0
	test_loader = loader(data_x[TrainSize:], data_y[TrainSize:], batch_size=1)
	for x, y in test_loader:
		wrong = 0

		x = Variable(x).cuda()
		w = model(x).cpu().detach().numpy().reshape(-1)
		w = 0.5 - w
		y = y.long().squeeze()
		y = y.cpu().detach().numpy()
		
		#w = 0.5 - x.cpu().detach().numpy().reshape(-1)

		wrong_u, obj_u = ILP_solve_upper(grb_model, z, w, y, arc, N, out_file)
		wrong_l, obj_l = ILP_solve_lower(data_y[:TrainSize], w, y, N, out_file)

		if obj_u > obj_l:
			wrong_m = wrong_u
		else:
			wrong_m = wrong_l

		#print ("Test Case %d, wrong variable: %d"%(cnt, wrong))
		cnt += 1
		if wrong_u == 0:
			correct_u += 1
		if wrong_l == 0:
			correct_l += 1
		if wrong_m == 0:
			correct_m += 1
		error_u += wrong_u
		error_l += wrong_l
		error_m += wrong_m

		if cnt % 10 == 0:
			print ("%d cases processed!"%(cnt))
	print ("Upper Bound Exact Match Accuracy: %d%%(%d/%d), average wrong: %f"%(int(correct_u * 100 / cnt), correct_u, cnt, error_u / float(cnt)))
	print ("Lower Bound Exact Match Accuracy: %d%%(%d/%d), average wrong: %f"%(int(correct_l * 100 / cnt), correct_l, cnt, error_l / float(cnt)))
	print ("Mixed Solve Exact Match Accuracy: %d%%(%d/%d), average wrong: %f"%(int(correct_m * 100 / cnt), correct_m, cnt, error_m / float(cnt)))
