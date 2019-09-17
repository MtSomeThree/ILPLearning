ans = 0

def DFS(N, now, columns, points, ans):
	if now == N:
		return ans + 1
	for i in range(N):
		if i in columns:
			continue
		flag = False
		for x, y in points:
			if abs(x - now) == abs(y - i):
				flag = True
				break
		if flag:
			continue
		columns.add(i)
		points.add((now, i))
		ans = DFS(N, now + 1, columns, points, ans)
		points.remove((now, i))
		columns.remove(i)
	return ans

N = 12
print (DFS(N, 0, set(), set(), 0))
