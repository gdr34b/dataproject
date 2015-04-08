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
    
def getValueRanges(leaf):
    if len(leaf.instances) > 0:
        valueLists = []
        for value in leaf.instances[0]:
            valueLists.append([])
        # get possible values
        for item in leaf.instances:
            for i in range(len(item)):
                valueLists[i].append(item[i])
        # build ranges from possible values
        valueRanges = []
        for eachList in valueLists:
            # for continuous variables, store min and max data
            if eachList[0].var_type == Orange.feature.Type.Continuous:
                valueRanges.append({ \
                    'type': Orange.feature.Type.Continuous, \
                    'min': min(eachList), \
                    'max': max(eachList)})
            # else, store all possible values
            else:
                valueRanges.append({ \
                    'type': eachList[0].var_type, \
                    'values': set([str(x) for x in eachList])})
        return valueRanges
        
def outputTreeToDotFile(treeClassifier, resultFileDirectory, filePrefix=''):
    if not os.path.isdir(resultFileDirectory):
        os.makedirs(resultFileDirectory)
    similarFiles = [x for x in os.listdir(resultFileDirectory) if filePrefix in x]
    fileNumber = len(similarFiles) + 1
    dotFileName = filePrefix + "_" + str(fileNumber) + ".dot"
    treeClassifier.dot(file_name=resultFileDirectory + dotFileName, leaf_shape="oval", node_shape="oval")
    