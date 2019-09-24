import sys
from queue import PriorityQueue

M = 5
K = 3
q = PriorityQueue()
q.put((8,1,5,5,1,0,(-1,-1,-1)))
cycle = [(5,5,1)]
res = []

cnt = 1
while not q.empty():
	p, curr, m, c, b, step, prev = q.get()
	for i in range(step):
		print("\t",end="")
	# print(m,c,b,"({}{}[{}])".format(prev,p,curr))
	print(m,c,b,"({})".format(p))
	# print("i",m,c,b,"({})".format(p))
	res.append((p,curr,m,c,b,step))
	if m == 0 and c == 0 and b == 0:
		print(step)
		# res.sort(key=lambda x:x[-1])
		# for item in res:
		# 	p, curr, m, c, b, step = item
		# 	for i in range(step):
		# 		print("\t",end="")
		# 	print(m,c,b,"({}[{}])".format(p,curr))
		sys.exit()
	for pair in [(0,1),(0,2),(0,3),(1,0),(1,1),(2,0),(2,1),(3,0)]: # boat ensure
		if pair[0] > m or pair[1] > c:
			continue
		if b == 0:
			newM = m + pair[0]
			newC = c + pair[1]
			newB = 1
		else:
			newM = m - pair[0]
			newC = c - pair[1]
			newB = 0
		if (newM != 0 and newM < newC) or (5 - newM != 0 and 5 - newM < 5 - newC):
			continue
		priority = step + 1 + newM + newC - 2 * newB
		cnt += 1
		if (newM,newC,newB) not in cycle:
			# for i in range(step+1):
			# 	print("\t",end="")
			# print(newM,newC,newB,"({}[{}])".format(priority,cnt))
			cycle.append((newM,newC,newB))
			q.put((priority,cnt,newM,newC,newB,step+1,(m,c,b)))
		else:
			for i in range(step+1):
				print("\t",end="")
			# print(newM,newC,newB,"({}d[{}])".format((m,c,b),cnt))
			print(newM,newC,newB,"(d)")
			res.append((0,cnt,newM,newC,newB,step+1))