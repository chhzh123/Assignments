import itertools
from copy import deepcopy

class VariableElimination:
    def inference(factorList, queryVariables, orderedListOfHiddenVariables, evidenceList):
        for ev in evidenceList:
            for i,node in enumerate(factorList):
                if ev in node.varList:
                    factorList[i] = node.restrict(ev,evidenceList[ev])
        for var in orderedListOfHiddenVariables:
            # for node in factorList:
            #     print(node.name,end=" ")
            # print("\n")
            newFactorList = []
            for node in factorList:
                if var in node.varList:
                    newFactorList.append(node)
            res = newFactorList[0]
            factorList.remove(res)
            for factor in newFactorList[1:]:
                res = res.multiply(factor)
                factorList.remove(factor)
            res = res.sumout(var)
            factorList.append(res)
        print("RESULT:")
        res = factorList[0]
        for factor in factorList[1:]:
            res = res.multiply(factor)
        total = sum(res.cpt.values())
        res.cpt = {k: v/total for k, v in res.cpt.items()}
        res.printInf()

    def printFactors(factorList):
        for factor in factorList:
            factor.printInf()

def get_new_cpt_var(new_var_list):
    valList = ""
    for var in new_var_list:
        valList += "valDict['{}'],".format(var)
    cpt_var = list(eval("itertools.product({})".format(valList)))
    return cpt_var

class Node:
    def __init__(self, name, var_list, val_list=[]):
        self.name = name
        # the first var is itself, others are dependency
        self.varList = var_list
        self.valList = val_list
        self.cpt = {}

    def setCpt(self, cpt):
        if type(list(cpt.keys())[0]) == type(""):
            for item in cpt:
                self.cpt[tuple([item])] = cpt[item]
        else:
            self.cpt = cpt

    def printInf(self):
        print("Name = " + self.name)
        print(" vars " + str(self.varList))
        for key in self.cpt:
            print("   key: " + str(key) + " val : " + str(self.cpt[key]))
        print()

    def multiply(self, factor):
        """function that multiplies with another factor"""
        var_list_1 = self.varList.copy()
        var_list_2 = factor.varList.copy()
        new_var_list = list(set(var_list_1 + var_list_2)) # take a union
        new_cpt = {}
        cpt_var = get_new_cpt_var(new_var_list)
        for var in cpt_var:
            var_dict = {}
            for i,v in enumerate(new_var_list):
                var_dict[v] = var[i]
            item = []
            for var1 in self.varList:
                item += [var_dict[var1]]
            f1 = self.cpt[tuple(item)]
            item = []
            for var2 in factor.varList:
                item += [var_dict[var2]]
            f2 = factor.cpt[tuple(item)]
            new_cpt[var] = f1 * f2
        new_node = Node("f" + str(new_var_list), new_var_list)
        new_node.setCpt(new_cpt)
        # print("{} multiply {} -> {}".format(self.name,factor.name,new_node.name))
        return new_node

    def sumout(self, variable):
        """function that sums out a variable given a factor"""
        index = self.varList.index(variable)
        new_var_list = self.varList.copy()
        new_var_list.remove(variable)
        cpt_var = get_new_cpt_var(new_var_list)
        new_cpt = {}
        for var in cpt_var:
            sumup = 0
            varLst = list(var)
            for curr in valDict[variable]:
                origin_var = tuple(varLst[:index] + [curr] + varLst[index:])
                sumup += self.cpt[origin_var]
            new_cpt[var] = sumup
        new_node = Node("f" + str(new_var_list), new_var_list)
        new_node.setCpt(new_cpt)
        # print("{} sumout {} -> {}".format(self.name,variable,new_node.name))
        return new_node

    def restrict(self, variable, value):
        """function that restricts a variable to some value
        in a given factor"""
        index = self.varList.index(variable)
        new_var_list = self.varList.copy()
        new_var_list.remove(variable)
        cpt_var = get_new_cpt_var(new_var_list)
        new_cpt = {}
        for var in cpt_var:
            varLst = list(var)
            origin_var = tuple(varLst[:index] + [value] + varLst[index:])
            new_cpt[var] = self.cpt[origin_var]
        new_node = Node("f" + str(new_var_list), new_var_list)
        new_node.setCpt(new_cpt)
        # print("{} restricts {} to {} -> {}".format(self.name,variable,value,new_node.name))
        return new_node

# create nodes for Bayes Net
PatientAge = Node("PatientAge",["PatientAge"],['0-30','31-65','65+'])
CTScanResult = Node("CTScanResult",["CTScanResult"],['Ischemic Stroke','Hemmorraghic Stroke'])
MRIScanResult = Node("MRIScanResult",["MRIScanResult"],['Ischemic Stroke','Hemmorraghic Stroke'])
Anticoagulants = Node("Anticoagulants",["Anticoagulants"],['Used','Not used'])

StrokeType = Node("StrokeType",["StrokeType","CTScanResult","MRIScanResult"],['Ischemic Stroke','Hemmorraghic Stroke','Stroke Mimic'])
Mortality = Node("Mortality",["Mortality",'StrokeType',"Anticoagulants"],['False','True'])
Disability = Node("Disability",["Disability","StrokeType","PatientAge"],['Negligible','Moderate','Severe'])

# Generate cpt for each node
PatientAge.setCpt({'0-30':0.10, '31-65':0.30, '65+':0.60})
CTScanResult.setCpt({'Ischemic Stroke':0.7, 'Hemmorraghic Stroke':0.3})
MRIScanResult.setCpt({'Ischemic Stroke':0.7, 'Hemmorraghic Stroke':0.3})
Anticoagulants.setCpt({'Used':0.5 ,'Not used':0.5})

StrokeType.setCpt({('Ischemic Stroke','Ischemic Stroke','Ischemic Stroke'):0.8,
	('Ischemic Stroke','Ischemic Stroke','Hemmorraghic Stroke'):0.5,
	('Ischemic Stroke','Hemmorraghic Stroke','Ischemic Stroke'):0.5,
	('Ischemic Stroke','Hemmorraghic Stroke','Hemmorraghic Stroke'):0,
	('Hemmorraghic Stroke','Ischemic Stroke','Ischemic Stroke'):0,
	('Hemmorraghic Stroke','Ischemic Stroke','Hemmorraghic Stroke'):0.4,
	('Hemmorraghic Stroke','Hemmorraghic Stroke','Ischemic Stroke'):0.4,
	('Hemmorraghic Stroke','Hemmorraghic Stroke','Hemmorraghic Stroke'):0.9,
	('Stroke Mimic','Ischemic Stroke','Ischemic Stroke'):0.2,
	('Stroke Mimic','Ischemic Stroke','Hemmorraghic Stroke'):0.1,
	('Stroke Mimic','Hemmorraghic Stroke','Ischemic Stroke'):0.1,
	('Stroke Mimic','Hemmorraghic Stroke','Hemmorraghic Stroke'):0.1})
Mortality.setCpt({('False','Ischemic Stroke','Used'):0.28,
	('False','Hemmorraghic Stroke','Used'):0.99,
	('False','Stroke Mimic','Used'):0.1,
	('False','Ischemic Stroke','Not used'):0.56,
	('False','Hemmorraghic Stroke','Not used'):0.58,
	('False','Stroke Mimic','Not used'):0.05,
	('True','Ischemic Stroke','Used'):0.72,
	('True','Hemmorraghic Stroke','Used'):0.01,
	('True','Stroke Mimic','Used'):0.9,
	('True','Ischemic Stroke','Not used'):0.44,
	('True','Hemmorraghic Stroke','Not used'):0.42,
	('True','Stroke Mimic','Not used'):0.95})
Disability.setCpt({('Negligible','Ischemic Stroke','0-30'):0.80,
	('Negligible','Hemmorraghic Stroke','0-30'):0.70,
	('Negligible','Stroke Mimic','0-30'):0.9,
	('Negligible','Ischemic Stroke','31-65'):0.60,
	('Negligible','Hemmorraghic Stroke','31-65'):0.50,
	('Negligible','Stroke Mimic','31-65'):0.4,
	('Negligible','Ischemic Stroke','65+'):0.30,
	('Negligible','Hemmorraghic Stroke','65+'):0.20,
	('Negligible','Stroke Mimic','65+'):0.1,
	('Moderate','Ischemic Stroke','0-30'):0.1,
	('Moderate','Hemmorraghic Stroke','0-30'):0.2,
	('Moderate','Stroke Mimic','0-30'):0.05,
	('Moderate','Ischemic Stroke','31-65'):0.3,
	('Moderate','Hemmorraghic Stroke','31-65'):0.4,
	('Moderate','Stroke Mimic','31-65'):0.3,
	('Moderate','Ischemic Stroke','65+'):0.4,
	('Moderate','Hemmorraghic Stroke','65+'):0.2,
	('Moderate','Stroke Mimic','65+'):0.1,
	('Severe','Ischemic Stroke','0-30'):0.1,
	('Severe','Hemmorraghic Stroke','0-30'):0.1,
	('Severe','Stroke Mimic','0-30'):0.05,
	('Severe','Ischemic Stroke','31-65'):0.1,
	('Severe','Hemmorraghic Stroke','31-65'):0.1,
	('Severe','Stroke Mimic','31-65'):0.3,
	('Severe','Ischemic Stroke','65+'):0.3,
	('Severe','Hemmorraghic Stroke','65+'):0.6,
	('Severe','Stroke Mimic','65+'):0.8})

factorList = [PatientAge,CTScanResult,MRIScanResult,Anticoagulants,StrokeType,Mortality,Disability]

valDict = {}
for factor in factorList:
    valDict[factor.name] = factor.valList

print("p1 = P(Mortality='True' && CTScanResult='Ischemic Stroke' | PatientAge='31-65')")
VariableElimination.inference(deepcopy(factorList), ["Mortality","CTScanResult"], ["MRIScanResult","Anticoagulants","StrokeType","Disability"], {"PatientAge":"31-65"})

print("p2 = P(Disability='Moderate' && CTScanResult='Hemmorraghic Stroke' | PatientAge='65+' &&  MRIScanResult='Hemmorraghic Stroke')")
VariableElimination.inference(deepcopy(factorList), ["Disability","CTScanResult"], ["Anticoagulants","StrokeType","Mortality"], {"PatientAge":"65+","MRIScanResult":"Hemmorraghic Stroke"})

print("p3 = P(StrokeType='Hemmorraghic Stroke' | PatientAge='65+' && CTScanResult='Hemmorraghic Stroke' && MRIScanResult='Ischemic Stroke')")
VariableElimination.inference(deepcopy(factorList), ["StrokeType"], ["Mortality","Disability","Anticoagulants"], {"PatientAge":"65+","CTScanResult":"Hemmorraghic Stroke","MRIScanResult":"Ischemic Stroke"})

print("p4 = P(Anticoagulants='Used' | PatientAge='31-65')")
VariableElimination.inference(deepcopy(factorList), ["Anticoagulants"], ["CTScanResult","MRIScanResult","StrokeType","Mortality","Disability"], {"PatientAge":"31-65"})

print("p5 = P(Disability='Negligible')")
VariableElimination.inference(deepcopy(factorList), ["Disability"], ["PatientAge","CTScanResult","MRIScanResult","Anticoagulants","StrokeType","Mortality"], {})