from copy import deepcopy
from queue import Queue
from queue import PriorityQueue
import sys

doms = [[1,2,3,4],[1,2,3,4],[1,2,3,4],[1,2,3,4],[1,2,3,4]]
sol = [0,0,0,0,0]
unassigned = [0,1,2,3,4]
constraints = {0:["a <= b","a > d","a != c"],
    1:["b >= a","b != c"],
    2:["c != a","c != b","c != d","c - 1 != d","c > e"],
    3:["d < a","d > e","d != c","d + 1 != c"],
    4:["e < c","e < d"]}

def get_index(alp):
    return ord(alp) - ord('a')

def FCCheck(c,curr_doms):
    x = get_index(c[0])
    y = get_index(c[-1])
    if y not in unassigned:
        return True
    curr_dom = curr_doms[y].copy()
    for z in curr_dom:
        if not eval("sol[x]" + c[1:-1] + "z"):
            curr_doms[y].remove(z)
    # print(c,x,y,curr_doms[y])
    if len(curr_doms[y]) == 0:
        return "DWO"
    else:
        return True

def FC(level,doms):
    global unassigned
    if 0 not in sol:
        print("Sol:",*sol)
        return
    mrv = PriorityQueue()
    for v in unassigned:
        mrv.put((len(doms[v]),v))
    _, var = mrv.get()
    dom = doms[var]
    unassigned.remove(var)
    for v in dom:
        tmp_doms = deepcopy(doms)
        sol[var] = v # assignment
        tmp_doms[var] = v
        DWOOccured = False
        for cst in constraints[var]:
            if FCCheck(cst,tmp_doms) == "DWO":
                DWOOccured = True
                break
        # print(level,*tmp_doms)
        print(level,end=" & ")
        print(*tmp_doms,sep=" & ",end="\\\ \\hline\n")
        if not DWOOccured:
            FC(level+1,tmp_doms)
    unassigned.append(var)
    sol[var] = 0
    return

def GAC_Enforce(queue,curr_doms):
    while not queue.empty():
        c = queue.get()
        # scope of c
        x = get_index(c[0])
        y = get_index(c[-1])
        for v in [x,y]:
            tmp_curr_doms = curr_doms[v].copy()
            for d1 in tmp_curr_doms:
                if sol[v] != 0 and sol[v] != d1:
                    continue
                other_dom = curr_doms[y] if v == x else curr_doms[x]
                flag = False
                for d2 in other_dom:
                    (a,b) = ("d1","d2") if v == x else ("d2","d1")
                    if eval(a + c[1:-1] + b):
                        flag = True
                        break
                if not flag:
                    curr_doms[v].remove(d1)
                    print(c,*curr_doms,sep=" & ",end="\\\ \\hline\n")
                    if len(curr_doms[v]) == 0:
                        queue = Queue()
                        return "DWO"
                    else:
                        for cnew in constraints[v]:
                            if cnew not in list(queue.queue):
                                queue.put(cnew)
    return True

def GAC(level,doms):
    global unassigned
    if 0 not in sol:
        print("Sol:",*sol)
        sys.exit()
        # return
    mrv = PriorityQueue()
    for v in unassigned:
        mrv.put((len(doms[v]),v))
    _, var = mrv.get()
    dom = doms[var]
    unassigned.remove(var)
    for v in dom:
        tmp_doms = deepcopy(doms)
        sol[var] = v # assignment
        tmp_doms[var] = [v]
        GACQueue = Queue()
        for cst in constraints[var]:
            GACQueue.put(cst)
        print(level,*tmp_doms,sep=" & ",end="\\\ \\hline\n")
        if (GAC_Enforce(GACQueue,tmp_doms) != "DWO"):
            # sys.exit()
            GAC(level+1,tmp_doms)
    unassigned.append(var)
    sol[var] = 0
    return

# FC(1,doms)
GAC(1,doms)