import sys, time
from copy import deepcopy
from queue import Queue

SIZE = int(sys.argv[2])

class Grid(object):
	def __init__(self,x,y,size):
		self.x = x
		self.y = y
		self.size = size
		self.value = -1
		self.valid = [i for i in range(1,size+1)]
		self.constraints = []
		self.tmp_value = self.value
		self.tmp_valid = self.valid.copy()

	def __str__(self):
		return str(self.value)

	def set_value(self,val):
		self.value = val
		# update current grid's domain
		self.valid = [val]

	def get_valid_num(self):
		return self.valid

	def get_valid_len(self):
		return len(self.valid)

	def add_constraint(self,c):
		self.constraints.append(c)

	def get_constraints(self):
		return self.constraints

	def del_num(self,num):
		if num == self.value:
			return
		try:
			index = self.valid.index(num)
			self.valid.pop(index)
		except:
			pass

	def prune_smaller_than(self,val):
		for i in range(val+1):
			self.del_num(i)

	def prune_bigger_than(self,val):
		for i in range(self.size,val-1,-1):
			self.del_num(i)

	def store_state(self):
		self.tmp_value = self.value
		self.tmp_valid = self.valid.copy()

	def restore_state(self):
		self.value = self.tmp_value
		self.valid = self.tmp_valid.copy()

	def print(self):
		print("Grid({},{})-{}:{}".format(self.x,self.y,self.value,self.valid),flush=True)

class Futoshiki(object):
	def __init__(self,size):
		self.size = size
		self.arr = []
		for i in range(size):
			self.arr.append([])
			for j in range(size):
				self.arr[i].append(Grid(i,j,size))

	def add(self,x,y,val):
		self.set_value(x,y,val)

	def add_constraint(self,grid1,op,grid2):
		if type(grid1) == type(""):
			grid1 = eval(grid1)
			grid2 = eval(grid2)
			x1, y1 = grid1[0]-1, grid1[1]-1
			x2, y2 = grid2[0]-1, grid2[1]-1
		else:
			x1, y1 = grid1[0], grid1[1]
			x2, y2 = grid2[0], grid2[1]
		self.arr[x1][y1].add_constraint(((x1,y1),op,(x2,y2)))
		if op == "!=":
			self.arr[x2][y2].add_constraint(((x2,y2),op,(x1,y1)))
		else:
			self.arr[x2][y2].add_constraint(((x2,y2),'<' if op == '>' else '>',(x1,y1)))

	def ini_unassigned(self):
		self.unassigned = []
		for i in range(self.size):
			for j in range(self.size):
				if self.arr[i][j].value == -1:
					self.unassigned.append((i,j))

	def set_unassigned(self,x,y):
		self.unassigned.append((x,y))

	def get_unassigned(self): # at the same time pop out
		# Minimum Remaining Values Heuristics (MRV)
		self.unassigned.sort(key=lambda x: self.arr[x[0]][x[1]].get_valid_len())
		res = self.unassigned[0]
		self.unassigned = self.unassigned[1:]
		return self.arr[res[0]][res[1]]

	def get_unassigned_len(self):
		return len(self.unassigned)

	def get_valid_len(self,x,y):
		return self.arr[x][y].get_valid_len()

	def set_value(self,x,y,val,propagation=True):
		self.arr[x][y].set_value(val)
		if not propagation:
			return
		# update constraints domain
		for c in self.arr[x][y].get_constraints():
			u, v = c[2][0], c[2][1]
			if c[1] == '<':
				self.arr[u][v].prune_smaller_than(val)
			elif c[1] == '>':
				self.arr[u][v].prune_bigger_than(val)
			elif c[1] == "!=":
				self.arr[u][v].del_num(val)

	def store(self):
		for i in range(self.size):
			for j in range(self.size):
				self.arr[i][j].store_state()

	def restore(self):
		for i in range(self.size):
			for j in range(self.size):
				self.arr[i][j].restore_state()

	def print(self):
		for i in range(self.size):
			print(*(self.arr[i]))
		print()

	def print_valid(self):
		for i in range(self.size):
			for j in range(self.size):
				print(self.arr[i][j].get_valid_num(),end=" ")
			print()
		print(flush=True)

def FC(board,level):
	if board.get_unassigned_len() == 0:
		board.print()
		print("Time: {:.2f}s".format(time.time()-start))
		sys.exit()
	grid = board.get_unassigned()
	for d in grid.get_valid_num():
		# avoid restoring the pruned values
		currBoard = deepcopy(board)
		# while set, do constraints propagation
		currBoard.set_value(grid.x,grid.y,d)
		# FCCheck
		DWO = False
		for i in range(SIZE):
			for j in range(SIZE):
				if currBoard.get_valid_len(i,j) == 0:
					# print(i,j,currBoard.arr[i][j].get_valid_num())
					DWO = True
					break
		if not DWO:
			FC(currBoard,level+1)
	board.set_unassigned(grid.x,grid.y)

def GAC_Enforce(board,q):
	while not q.empty():
		v1, op, v2 = q.get()
		v1 = board.arr[v1[0]][v1[1]]
		v2 = board.arr[v2[0]][v2[1]]
		for (v,v_other) in [(v1,v2),(v2,v1)]:
			valid_num = v.get_valid_num().copy()
			if op != "!=" and (v.x,v.y) != (v1.x,v1.y):
				op = "<" if op == ">" else ">"
			for d in valid_num:
				flag_assignment = False
				for d_other in v_other.get_valid_num():
					if eval("d " + op + " d_other"):
						flag_assignment = True
						break
				if not flag_assignment:
					v.del_num(d)
					if v.get_valid_len() == 0:
						q = Queue()
						return "DWO"
					else:
						for c_other in v.get_constraints():
							if c_other not in list(q.queue):
								q.put(c_other)
	return True

def GAC(board,level):
	if board.get_unassigned_len() == 0:
		board.print()
		print("Time: {:.2f}s".format(time.time()-start))
		sys.exit()
	grid = board.get_unassigned()
	valid_num = grid.get_valid_num()
	for d in valid_num:
		# avoid restoring the pruned values
		currBoard = deepcopy(board)
		# need not do propagation
		currBoard.set_value(grid.x,grid.y,d,False)
		GACQueue = Queue()
		for c in currBoard.arr[grid.x][grid.y].get_constraints():
			GACQueue.put(c)
		if GAC_Enforce(currBoard,GACQueue) != "DWO":
			GAC(currBoard,level+1)
	board.set_unassigned(grid.x,grid.y)

board = Futoshiki(SIZE)
toBeAdded = []
with open(sys.argv[1],"r") as file:
	for (i,line) in enumerate(file):
		if i < SIZE:
			lst = list(map(int,line.split()))
			for (j,val) in enumerate(lst):
				if val != 0:
					toBeAdded.append((i,j,val))
		else:
			grid1, op, grid2 = line.split()
			board.add_constraint(grid1,op,grid2)

for i in range(SIZE):
	for j in range(SIZE):
		# column contraints
		for k in range(i+1,SIZE):
			board.add_constraint((i,j),"!=",(k,j))
		# row constraints
		for k in range(j+1,SIZE):
			board.add_constraint((i,j),"!=",(i,k))

# after all the constraints are read in
# add the known values (ensure pruning at first)
for (i,j,val) in toBeAdded:
	board.add(i,j,val)
board.ini_unassigned()

start = time.time()
# FC(board,0)
GAC(board,0)