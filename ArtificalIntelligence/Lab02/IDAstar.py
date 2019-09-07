import sys
from copy import deepcopy
from queue import PriorityQueue

MAX_INT = 0x3f3f3f3f

def print_arr(arr):
	if arr == None:
		print(arr)
	else:
		for i in range(4):
			print(*(arr[i]))
	print()

def get_target_pos(num):
	return ((num-1) // 4, (num-1) % 4) if num != 0 else (3,3)

def heuristics(a):
	res = 0
	for i in range(4):
		for j in range(4):
			(ti, tj) = get_target_pos(a[i][j])
			res += abs(i - ti) + abs(j - tj)
	return res

def find_space(a):
	for i in range(4):
		try:
			j = a[i].index(0)
			return (i,j)
		except:
			pass

def move(a,i,j,d):
	res = deepcopy(a)
	if d == 'U':
		if i+1 < 4:
			res[i][j], res[i+1][j] = res[i+1][j], res[i][j]
			return res
		else:
			return None
	elif d == 'L':
		if j+1 < 4:
			res[i][j], res[i][j+1] = res[i][j+1], res[i][j]
			return res
		else:
			return None
	elif d == 'D':
		if i-1 >= 0:
			res[i][j], res[i-1][j] = res[i-1][j], res[i][j]
			return res
		else:
			return None
	elif d == 'R':
		if j-1 >= 0:
			res[i][j], res[i][j-1] = res[i][j-1], res[i][j]
			return res
		else:
			return None
	else:
		raise RuntimeError
cnt = 0
def search(path,cost,bound):
	global cnt
	# q = PriorityQueue()
	# q.put((0,0,arr.copy()))
	# minPriority = []
	# found = False
	# while not q.empty():
	# 	_, cost, a = q.get()
	# 	if a == [[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,-1]]:
	# 		found = True
	# 		break
	# 	(si,sj) = find_space(a)
	# 	for d in ['U','L','D','R']:
	# 		new_state = move(a,si,sj,d)
	# 		priority = cost + 1 + heuristics(new_state)
	# 		if new_state != None:
	# 			if priority > bound:
	# 				minPriority.append(priority)
	# 			else:
	# 				q.put((priority,cost + 1,new_state))
	# return min(minPriority), found
	arr = path[-1]
	f = cost + heuristics(arr)
	if f > bound:
		return f, False
	if arr == [[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,0]]:
		return f, True # h = 0
	(si,sj) = find_space(arr)

	# find the successors
	q = PriorityQueue()
	for d in ['U','L','D','R']:
		new_state = move(arr,si,sj,d) # not share
		# print_arr(new_state)
		if new_state in path:
			continue
		if new_state != None:
			priority = cost + 1 + heuristics(new_state)
			q.put((priority,new_state))
	# cnt += 1
	# if cnt == 3000:
	# 	print(new_state)
	# 	sys.exit()

	minF = MAX_INT
	while not q.empty():
		_, state = q.get()
		# print(state)
		path = path + [state]
		f, found = search(path,cost+1,bound)
		if found == True:
			# print(state, f)
			return f, True
		else:
			minF = min(f,minF)
		del path[-1]
	return minF, False

def ida_star(src):
	bound = heuristics(arr)
	path = [src]
	while True:
		bound, found = search(path,0,bound)
		if found == True:
			break
		if bound == MAX_INT:
			print("NOT FOUND!")
			return
	print(bound)

arr = []
for i in range(4):
	arr.append(list(map(int,input().split())))

ida_star(arr)