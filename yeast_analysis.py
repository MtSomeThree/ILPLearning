from NN_ILP_multi import *

if __name__ == '__main__':
	DataDir = './data/Multi/yeast.csv'
	TrainSize = 1500
	out_file = open('./data/Multi/result.txt', 'w')
	trainFlag = False
	data_x, data_y = get_data(open(DataDir, 'r'))

	print (data_y.sum(axis=0))
	print (data_y[:30, :])

	d = dict()
	for i in range(13):
		for j in range(i + 1, 13):
			d[(i, j)] = dict()
			for k in range(2417):
				l1 = data_y[k][i]
				l2 = data_y[k][j]
				if (l1, l2) in d[(i, j)]:
					d[(i, j)][(l1, l2)] += 1
				else:
					d[(i, j)][(l1, l2)] = 1
	for i in range(13):
		for j in range(i + 1, 13):
			for k in range(2):
				for l in range(2):
					if k + l != 1:
						continue
					if (k, l) not in d[(i, j)]:
						print (i, j, k, l, 0)
						continue
					if d[(i, j)][(k, l)] < 200:
						print (i, j, k, l, d[(i, j)][(k, l)])