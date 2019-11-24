import BayesianNetwork as BN

# create nodes for Bayes Net
A = BN.Node("A",["A"],["0","1"])
B = BN.Node("B",["B"],["0","1"])
C = BN.Node("C",["C","A","B"],["0","1"])
D = BN.Node("D",["D","B"],["0","1"])
E = BN.Node("E",["E","C"],["0","1"])
F = BN.Node("F",["F","C"],["0","1"])

# Generate cpt for each node
A.setCpt({"1":0.9,"0":0.1})
B.setCpt({"1":0.2,"0":0.8})
C.setCpt({("1","1","1"):0.1,
          ("1","1","0"):0.8,
          ("1","0","1"):0.7,
          ("1","0","0"):0.4,
          ("0","1","1"):0.9,
          ("0","1","0"):0.2,
          ("0","0","1"):0.3,
          ("0","0","0"):0.6
          })
D.setCpt({("1","1"):0.1,
          ("1","0"):0.8,
          ("0","1"):0.9,
          ("0","0"):0.2
          })
E.setCpt({("1","1"):0.7,
          ("1","0"):0.2,
          ("0","1"):0.3,
          ("0","0"):0.8
          })
F.setCpt({("1","1"):0.2,
          ("1","0"):0.9,
          ("0","1"):0.8,
          ("0","0"):0.1
          })

factorList = [A,B,C,D,E,F]

print("P(E)")
ve = BN.VariableElimination(factorList)
ve.inference(["E"], ["A","B","C","D","F"], {})

print("P(E|~F)")
ve = BN.VariableElimination(factorList)
ve.inference(["E"], ["D","A","B","C"], {"F":"0"})