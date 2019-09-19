import numpy as np

def MST(N, dist, arc):
	edgeList = []
	for i, j in arc:
		edgeList.append(((i, j), dist[i, j]))
	edgeList.sort(key = lambda x : x[1])
	father = [-1] * (N + 1)
	
	def find(x):
		if father[x] == -1:
			return x
		temp = find(father[x])
		father[x] = temp
		return temp

	def union(x, y):
		a = find(x)
		b = find(y)
		if a == b:
			return True
		else:
			father[a] = b
			return False

	cnt = 0
	ans = 0
	solution = np.zeros((N + 1, N + 1))
	for e in edgeList:
		(i, j), d = e
		if not union(i, j):
			ans += d
			cnt += 1
			solution[i, j] = 1
		if cnt == N:
			break
	return ans, solution

if __name__ == '__main__':	

	N = 6

	data_file = open('./data/MST/train.txt', 'w')
	#data_file.write("%d\n"%(N))
	for T in range(15000):
		dist = np.zeros((N + 1, N + 1))
		arc = set()
		dist_set = set()

		for i in range(N + 1):
			for j in range(i + 1, N + 1):
				tmp = np.random.rand()
				while (tmp in dist_set):
					tmp = np.random.rand()
				dist_set.add(tmp)
				dist[i, j] = tmp
				dist[j, i] = dist[i, j]
				arc.add((i, j))
				arc.add((j, i))
		
		for i in range(N + 1):
			for j in range(N + 1):
				data_file.write("%f "%(dist[i, j]))
			data_file.write("\n")
	
		ans, sol = MST(N, dist, arc)

		for i in range(N + 1):
			for j in range(N + 1):
				if sol[i, j] == 1:
					sol[j, i] = 1

		for i in range(N + 1):
			for j in range(N + 1):
				data_file.write("%d "%(sol[i, j]))
			data_file.write("\n")
		
