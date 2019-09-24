import torch
from torch.autograd import Variable
import numpy as np
import random
import csv
from scipy.linalg import null_space
from NN_mst import myModel
from learning import *


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

def ILP_solve(model, z, w, x, arc, N, out_file):
	model.setObjective(sum(z[idx] * w[idx] for idx in arc))
	model.optimize()

	solution = np.zeros(N)
	for v in model.getVars():
		if v.varName[0] == 'z':
			index = eval(v.varName[1:])
			solution[index[0]] = v.x

	wrong = 0
	for i in range(N):
		if solution[i] != x[i]:
			wrong += 1

	return wrong

if __name__ == '__main__':
	DataDir = './data/Multi/yeast.csv'
	TrainSize = 2200
	out_file = open('./data/Multi/result.txt', 'w')
	data_x, data_y = get_data(open(DataDir, 'r'))

	arc = set()
	for i in range(14):
		arc.add(i)
	grb_model, z = init_GRB(arc, task='multi')

	model = myModel(103, 14)
	model.cuda()

	criterion = torch.nn.BCELoss() 

	optimizer = torch.optim.Adam(model.parameters(), lr=0.003, weight_decay=0.00001)

	lossLog = []
	for epoch in range(200):
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

	for i in range(TrainSize):
		x = Variable(torch.Tensor(data_x[i].reshape(1, -1))).cuda()
		y = data_y[i]
		w = model(x).cpu().detach().numpy().reshape(-1)
		w = 0.5 - w

		add_data_point(grb_model, z, w, y, arc, i)

	ones = np.ones(TrainSize)
	print (np.c_[data_y[:TrainSize], ones].shape)
	zero_space = null_space(np.c_[data_y[:TrainSize], ones])
	print (zero_space)

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

	cnt = 0
	correct = 0
	error = 0
	test_loader = loader(data_x[TrainSize:], data_y[TrainSize:], batch_size=1)
	for x, y in test_loader:
		wrong = 0

		x = Variable(x).cuda()
		y = y.long().squeeze()

		w = model(x).cpu().detach().numpy().reshape(-1)
		y = y.cpu().detach().numpy()
		w = 0.5 - w

		wrong = ILP_solve(grb_model, z, w, y, arc, 14, out_file)

		#print ("Test Case %d, wrong variable: %d"%(cnt, wrong))
		cnt += 1
		if wrong == 0:
			correct += 1
		error += wrong
		if cnt % 10 == 0:
			print ("%d:%d"%(cnt, wrong))
	print ("Exact Match Accuracy: %d%%(%d/%d), average wrong: %f(%d/%d)"%(int(correct * 100 / cnt), correct, cnt, float(error) / float(cnt), error, cnt))

	constrs = grb_model.getConstrs()
	print (len(constrs))
