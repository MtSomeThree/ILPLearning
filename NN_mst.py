import torch
from torch.autograd import Variable
import numpy as np
import random

class myModel(torch.nn.Module):
	def __init__(self, input_dim, output_dim):
		super(myModel, self).__init__()
		hidden_dim = input_dim
		self.linears = torch.nn.Sequential(
			torch.nn.Linear(input_dim, hidden_dim),
			torch.nn.ReLU(),
			torch.nn.Linear(hidden_dim, hidden_dim),
			torch.nn.ReLU(),
			torch.nn.Linear(hidden_dim, hidden_dim),
			torch.nn.Sigmoid()
			)

	def forward(self, x):
		return self.linears(x)

def get_data(data_file, N):
	cnt = 0
	data = []
	while(True):
		cnt = 0
		weight = []
		for line in data_file:
			tmp = line.strip().split()
			tmp = [float(x) for x in tmp]
			weight.append(tmp)
			cnt += 1
			if cnt == N + 1:
				break
		if cnt < N + 1:
			break
		w = []
		for i in range(N + 1):
			for j in range(i + 1, N + 1):
				w.append(weight[i][j])

		cnt = 0
		sol = []
		for line in data_file:
			tmp = line.strip().split()
			tmp = [int(x) for x in tmp]
			sol.append(tmp)
			cnt += 1
			if cnt == N + 1:
				break
		if cnt < N + 1:
			break
		x = []
		for i in range(N + 1):
			for j in range(i + 1, N + 1):
				x.append(sol[i][j])
		data.append((w, x))
	return data

def loader(data, batch_size=50):
	random.shuffle(data)
	length = len(data)
	for b in range(int(length / batch_size)):
		weights = []
		sols = []
		for j in range(batch_size):
			i = b * batch_size + j
			w, x = data[i]
			weights.append(w)
			sols.append(x)
		weights = torch.Tensor(np.array(weights))
		sols = torch.Tensor(np.array(sols))
		yield weights, sols

if __name__ == '__main__':
	N = 6
	train_data = get_data(open('train.txt', 'r'), N)
	train_loader = loader(train_data)
	test_data = get_data(open('test.txt', 'r'), N)
	test_loader = loader(test_data, batch_size=1)

	dim = int(N * (N + 1) / 2)

	model = myModel(dim, dim)
	model.cuda()
	criterion = torch.nn.MSELoss() 

	optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

	lossLog = []
	for epoch in range(200):
		total_loss = 0
		cnt = 0
		train_loader = loader(train_data)
		for w, x in train_loader:
			w = Variable(w).cuda()
			x = Variable(x.float()).cuda()

			optimizer.zero_grad()
			outputs = model(w)
			loss = criterion(outputs, x)
			loss.backward()
			optimizer.step()
			cnt += 1
			total_loss += loss
		print ('epoch %d, loss %.8f'%(epoch, total_loss / cnt))
		lossLog.append(total_loss)

	cnt = 0
	correct = 0
	error = 0
	for w, x in test_loader:
		wrong = 0

		w = Variable(w).cuda()
		x = x.long().squeeze()

		output = model(w).squeeze()
		prediction = np.where(output.cpu().detach().numpy() > 0.5, 1, 0) + 1e-3
		prediction = torch.LongTensor(prediction)
		wrong += (prediction != x).sum()

		print ("Test Case %d, wrong variable: %d"%(cnt, wrong))
		cnt += 1
		if wrong == 0:
			correct += 1
		error += wrong
	print ("Exact Match Accuracy: %d%%(%d/%d), average wrong: %f"%(int(correct * 100 / cnt), correct, cnt, error / cnt))

