#!/usr/bin/python

import Orange, os

def getLeafNodes(node):
    leafNodes = []
    # null node
    if not node:
        pass
    # internal node
    elif node.branch_selector:
        nodeDesc = node.branch_selector.class_var.name
        for branch in node.branches:
            leafNodes += getLeafNodes(branch)
    # leaf node
    else:
        leafNodes.append(node)
    return leafNodes
    
def getAllValues(leaf):
    allValues = None
    if len(leaf.instances) > 0:
        allValues = []
        for value in leaf.instances[0]:
            allValues.append([])
        for item in leaf.instances:
            for i in range(len(item)):
                allValues[i].append(item[i])
    return allValues
    
def getValueRanges(leaf):
    allValues = getAllValues(leaf)
    if allValues is not None:
        # build ranges from possible values
        valueRanges = []
        for eachList in allValues:
            # for continuous variables, store min and max data
            if eachList[0].var_type == Orange.feature.Type.Continuous:
                valueRanges.append({ \
                    'type': Orange.feature.Type.Continuous, \
                    'min': min(eachList), \
                    'max': max(eachList)})
            # else, store all values
            else:
                valueRanges.append({ \
                    'type': eachList[0].var_type, \
                    'values': [str(x) for x in eachList]})
        return valueRanges
        
def outputTreeToDotFile(treeClassifier, resultFileDirectory, filePrefix=''):
    if not os.path.isdir(resultFileDirectory):
        os.makedirs(resultFileDirectory)
    similarFiles = [x for x in os.listdir(resultFileDirectory) if filePrefix in x]
    fileNumber = len(similarFiles) + 1
    dotFileName = filePrefix + "_" + str(fileNumber) + ".dot"
    treeClassifier.dot(file_name=resultFileDirectory + dotFileName, leaf_shape="oval", node_shape="oval")
    
def getModelAccuracy(treeClassifier, dataTable):
    correct = 0
    for item in dataTable:
        if treeClassifier(item) == item.get_class():
            correct += 1
    return float(correct) / float(len(dataTable))
