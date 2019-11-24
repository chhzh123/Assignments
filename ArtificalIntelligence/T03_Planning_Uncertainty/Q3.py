from copy import deepcopy
import BayesianNetwork as BN

# create nodes for Bayes Net
A = BN.Node("A",["A"],["0","1"]) # metastatic cancer
B = BN.Node("B",["B","A"],["0","1"]) # increased total serum calcium
C = BN.Node("C",["C","A"],["0","1"]) # brain tumor
D = BN.Node("D",["D","B","C"],["0","1"]) # occasional coma
E = BN.Node("E",["E","C"],["0","1"]) # severe headaches

# Generate cpt for each node
A.setCpt({"1":0.2,"0":0.8})
B.setCpt({("1","1"):0.8,
          ("1","0"):0.2,
          ("0","1"):0.2,
          ("0","0"):0.8
          })
C.setCpt({("1","1"):0.2,
          ("1","0"):0.05,
          ("0","1"):0.8,
          ("0","0"):0.95
          })
D.setCpt({("1","1","1"):0.8,
          ("1","1","0"):0.8,
          ("1","0","1"):0.8,
          ("1","0","0"):0.05,
          ("0","1","1"):0.2,
          ("0","1","0"):0.2,
          ("0","0","1"):0.2,
          ("0","0","0"):0.95
          })
E.setCpt({("1","1"):0.8,
          ("1","0"):0.6,
          ("0","1"):0.2,
          ("0","0"):0.4
          })

factorList = [A,B,C,D,E]

print("P(A,B,C,~D,E)")
ve = BN.VariableElimination(deepcopy(factorList))
ve.inference(["A","B","C","D","E"], [], {})

# e,d,b,c,a
#    key: ('1', '0', '0', '0', '0') val : 0.34656
#    key: ('1', '0', '0', '0', '1') val : 0.01824
#    key: ('1', '0', '0', '1', '0') val : 0.005120000000000001
#    key: ('1', '0', '0', '1', '1') val : 0.0012800000000000003
#    key: ('1', '0', '1', '0', '0') val : 0.01824
#    key: ('1', '0', '1', '0', '1') val : 0.01536
#    key: ('1', '0', '1', '1', '0') val : 0.0012800000000000003
#    key: ('1', '0', '1', '1', '1') val : 0.005120000000000001

print("P(A|~D,E)")
ve = BN.VariableElimination(deepcopy(factorList))
ve.inference(["A"], ["B","C"], {"D":"0","E":"1"})