#!/usr/bin/python

import math, Orange, os, random, sys, traceback

################################################################################
# Global Variables
################################################################################
NEW_DATA_PER_LEAF = 0.5
RESULT_FILE_DIRECTORY = "./results/"

################################################################################
# Function Definitions
################################################################################
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
    
def buildNewInstances(dataTable, leaf):
    newData = []
    domain = dataTable.domain
    valueRanges = getValueRanges(leaf)
    numOfNewInstances = int(math.floor(len(leaf.instances) * NEW_DATA_PER_LEAF))
    for i in range(numOfNewInstances):
        newInstance = []
        for eachRange in valueRanges:
            if eachRange['type'] == Orange.feature.Type.Continuous:
                newInstance.append(random.uniform(eachRange['min'], eachRange['max']))
            else:
                newInstance.append(random.sample(eachRange['values'], 1)[0])
        orangeInstance = Orange.data.Instance(domain, newInstance)
        newData.append(orangeInstance)
    return newData
    
def addNewDataToTable(dataTable, newData):
    for item in newData:
        dataTable.append(item)
        
def outputTreesToDotFiles(treeClassifier, newTreeClassifier):
    if not os.path.isdir(RESULT_FILE_DIRECTORY):
        os.makedirs(RESULT_FILE_DIRECTORY)
    contents = os.listdir(RESULT_FILE_DIRECTORY)
    fileNumber = len(contents) / 2
    oldTreeFileName = "old_tree_" + str(fileNumber) + ".dot"
    newTreeFileName = "new_tree_" + str(fileNumber) + ".dot"
    treeClassifier.dot(file_name=RESULT_FILE_DIRECTORY + oldTreeFileName, leaf_shape="oval", node_shape="oval")
    newTreeClassifier.dot(file_name=RESULT_FILE_DIRECTORY + newTreeFileName, leaf_shape="oval", node_shape="oval")

################################################################################
# Main Script
################################################################################
if __name__ == "__main__":
    try:
        # load data into table
        irisData = Orange.data.Table("iris")
        # create decision tree
        treeClassifier = Orange.classification.tree.TreeLearner(irisData, store_instances=True)
        # make new data and add it to the current table
        leafNodes = getLeafNodes(treeClassifier.tree)
        newData = []
        for leaf in leafNodes:
            newData += buildNewInstances(irisData, leaf)
        addNewDataToTable(irisData, newData)
        # build another decision tree including the new instances
        newTreeClassifier = Orange.classification.tree.TreeLearner(irisData, store_instances=True)
        # output decision trees to DOT files (used by Graphviz)
        outputTreesToDotFiles(treeClassifier, newTreeClassifier)
        # somehow compare the new data model to the old one
        print("*** Old Decision Tree ***\n" + str(treeClassifier) + \
            "\n*** New Decision Tree ***\n" + str(newTreeClassifier))
    except:
        traceback.print_exc(file=sys.stdout)
    