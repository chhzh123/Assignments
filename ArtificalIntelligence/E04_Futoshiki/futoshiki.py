import sys, time
from copy import deepcopy

SIZE = int(sys.argv[1])

class Grid(object):
	def __init__(self,x,y,size):
		self.x = x
		self.y = y
		self.size = size
		self.value = -1
		self.valid = [i for i in range(1,size+1)]
		self.constraints = []

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
		grid1 = eval(grid1)
		grid2 = eval(grid2)
		x1, y1 = grid1[0]-1, grid1[1]-1
		x2, y2 = grid2[0]-1, grid2[1]-1
		self.arr[x1][y1].add_constraint((op,(x2,y2)))
		self.arr[x2][y2].add_constraint(('<' if op == '>' else '>',(x1,y1)))

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

	def set_value(self,x,y,val):
		self.arr[x][y].set_value(val)
		# update row domain
		for j in range(self.size):
			if j != y:
				self.arr[x][j].del_num(val)
		# update column domain
		for i in range(self.size):
			if i != x:
				self.arr[i][y].del_num(val)
		# update constraints domain
		for c in self.arr[x][y].get_constraints():
			u, v = c[1][0], c[1][1]
			if c[0] == '<':
				self.arr[u][v].prune_smaller_than(val)
			elif c[0] == '>':
				self.arr[u][v].prune_bigger_than(val)

	def print(self):
		for i in range(self.size):
			print(*(self.arr[i]))
		print()

	def print_valid(self):
		for i in range(self.size):
			for j in range(self.size):
				print(self.arr[i][j].get_valid_num(),end=" ")
			print()
		print()

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

board = Futoshiki(SIZE)
toBeAdded = []
with open("in.txt","r") as file:
	for (i,line) in enumerate(file):
		if i < SIZE:
			lst = list(map(int,line.split()))
			for (j,val) in enumerate(lst):
				if val != 0:
					toBeAdded.append((i,j,val))
		else:
			grid1, op, grid2 = line.split()
			board.add_constraint(grid1,op,grid2)

# after all the constraints are read in
# add the known values (ensure pruning at first)
for (i,j,val) in toBeAdded:
	board.add(i,j,val)
board.ini_unassigned()

start = time.time()
FC(board,0)

''' Test case
0 0 0 0
0 0 0 0
0 0 0 0
0 0 0 3
(1,1) > (1,2)
(1,3) > (1,4)
(1,4) > (2,4)
(3,3) > (2,3)
3 1 4 2
4 3 2 1
1 2 3 4
2 4 1 3

0 0 0 7 3 8 0 5 0
0 0 7 0 0 2 0 0 0
0 0 0 0 0 9 0 0 0
0 0 0 4 0 0 0 0 0
0 0 1 0 0 0 6 4 0
0 0 0 0 0 0 2 0 0
0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 6
(1,1) < (1,2)
(1,3) > (1,4)
(2,4) < (2,5)
(2,7) < (2,8)
(2,7) > (3,7)
(3,1) > (3,2)
(3,3) < (3,4)
(3,4) < (4,4)
(4,2) > (5,2)
(4,3) > (4,4)
(4,5) > (4,6)
(4,6) < (4,7)
(4,6) > (5,6)
(4,8) > (4,9)
(5,1) < (5,2)
(5,5) > (6,5)
(5,9) > (6,9)
(6,2) < (6,3)
(6,2) < (7,2)
(6,5) < (6,6)
(6,7) > (6,8)
(6,7) > (7,7)
(6,9) > (7,9)
(7,4) < (7,5)
(7,8) > (8,8)
(8,2) < (9,2)
(8,3) > (9,3)
(8,6) < (9,6)
(8,9) > (9,9)
(9,6) < (9,7)
'''