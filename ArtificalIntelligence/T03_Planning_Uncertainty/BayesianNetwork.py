import itertools
from copy import deepcopy

class VariableElimination:
    def __init__(self,factorList):
        self.factorList = deepcopy(factorList)
        self.valDict = {}
        for factor in self.factorList:
            self.valDict[factor.name] = factor.valList

    def inference(self, queryVariables, orderedListOfHiddenVariables, evidenceList):
        for ev in evidenceList:
            for i,node in enumerate(self.factorList):
                if ev in node.varList:
                    self.factorList[i] = node.restrict(ev,evidenceList[ev],self.valDict)
        for var in orderedListOfHiddenVariables:
            # for node in self.factorList:
            #     print(node.name,end=" ")
            # print("\n")
            newFactorList = []
            for node in self.factorList:
                if var in node.varList:
                    newFactorList.append(node)
            res = newFactorList[0]
            self.factorList.remove(res)
            for factor in newFactorList[1:]:
                res = res.multiply(factor,self.valDict)
                self.factorList.remove(factor)
            res = res.sumout(var,self.valDict)
            self.factorList.append(res)
        print("RESULT:")
        res = self.factorList[0]
        for factor in self.factorList[1:]:
            res = res.multiply(factor,self.valDict)
        total = sum(res.cpt.values())
        res.cpt = {k: v/total for k, v in res.cpt.items()}
        res.printInf()

    def printFactors(self):
        for factor in self.factorList:
            factor.printInf()

def get_new_cpt_var(new_var_list,valDict):
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

    def multiply(self, factor, valDict, printFlag=True):
        """function that multiplies with another factor"""
        var_list_1 = self.varList.copy()
        var_list_2 = factor.varList.copy()
        new_var_list = list(set(var_list_1 + var_list_2)) # take a union
        new_cpt = {}
        cpt_var = get_new_cpt_var(new_var_list,valDict)
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
        if printFlag:
            print("{} multiply {} -> {}".format(self.name,factor.name,new_node.name))
        return new_node

    def sumout(self, variable, valDict, printFlag=True):
        """function that sums out a variable given a factor"""
        index = self.varList.index(variable)
        new_var_list = self.varList.copy()
        new_var_list.remove(variable)
        cpt_var = get_new_cpt_var(new_var_list,valDict)
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
        if printFlag:
            print("{} sumout {} -> {}".format(self.name,variable,new_node.name))
        return new_node

    def restrict(self, variable, value, valDict, printFlag=True):
        """function that restricts a variable to some value
        in a given factor"""
        index = self.varList.index(variable)
        new_var_list = self.varList.copy()
        new_var_list.remove(variable)
        cpt_var = get_new_cpt_var(new_var_list,valDict)
        new_cpt = {}
        for var in cpt_var:
            varLst = list(var)
            origin_var = tuple(varLst[:index] + [value] + varLst[index:])
            new_cpt[var] = self.cpt[origin_var]
        new_node = Node("f" + str(new_var_list), new_var_list)
        new_node.setCpt(new_cpt)
        if printFlag:
            print("{} restricts {} to {} -> {}".format(self.name,variable,value,new_node.name))
        return new_node