from pomegranate import *

burglary = DiscreteDistribution( {'T':0.001, 'F':0.999} )
earthquake = DiscreteDistribution( {'T':0.002, 'F':0.998} )
alarm = ConditionalProbabilityTable(
	[['T','T','T',0.95],
	 ['T','F','T',0.94],
	 ['F','T','T',0.29],
	 ['F','F','T',0.001],
	 ['T','T','F',0.05],
	 ['T','F','F',0.06],
	 ['F','T','F',0.71],
	 ['F','F','F',0.999]], [burglary, earthquake])
johncalls = ConditionalProbabilityTable(
	[['T','T',0.90],
	 ['F','T',0.05],
	 ['T','F',0.10],
	 ['F','F',0.95]], [alarm])
marycalls = ConditionalProbabilityTable(
	[['T','T',0.70],
	 ['F','T',0.01],
	 ['T','F',0.30],
	 ['F','F',0.99]], [alarm])

s1 = State(burglary, name="burglary")
s2 = State(earthquake, name="earthquake")
s3 = State(alarm, name="alarm")
s4 = State(johncalls, name="johncalls")
s5 = State(marycalls, name="marycalls")

model = BayesianNetwork("Burglary")

model.add_states(s1,s2,s3,s4,s5)

model.add_transition(s1,s3)
model.add_transition(s2,s3)
model.add_transition(s3,s4)
model.add_transition(s3,s5)
model.bake()

marginals = model.predict_proba({})
# P(A)
print("P(A) = {}".format(marginals[2].parameters[0]["T"]))
# P(J&&~M) = P(J|~M)P(~M)
j_nm = model.predict_proba({'marycalls':'F'})[3].parameters[0]["T"] * marginals[4].parameters[0]["F"]
print("P(J && ~M) = {}".format(j_nm))
# P(A|J&&~M)
print("P(A | J && ~M) = {}".format(model.predict_proba({'johncalls':'T','marycalls':'F'})[2].parameters[0]["T"]))
# P(B|A)
print("P(B | A) = {}".format(model.predict_proba({'alarm':'T'})[0].parameters[0]["T"]))
# P(B|J&&~M)
b_c_j_nm = model.predict_proba({'johncalls':'T','marycalls':'F'})[0].parameters[0]["T"]
print("P(B | J && ~M) = {}".format(b_c_j_nm))
# P(J&&~M|~B) = P(~B && J && ~M) / P(~B)
#             = P(~B | J && ~M) P(J && ~M) / P(~B)
#             = (1- P(B | J && ~M)) P(J && ~M) / P(~B)
print("P(J && ~M | ~B) = {}".format((1-b_c_j_nm) * j_nm / marginals[0].parameters[0]["F"]))