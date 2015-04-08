#!/usr/bin/python

import copy, math, Orange, random, sys, traceback
sys.path.append("./libraries")
import tree_functions as TREE

################################################################################
# Global Variables
################################################################################
NEW_DATA_PER_LEAF = 0.5
RESULT_FILE_DIRECTORY = "./results/"
OUTPUT_DOT_FILES = False
MAX_DEPTH = 3

################################################################################
# Function Definitions
################################################################################
def interpolateLeafData(dataTable, leaf):
    newData = []
    domain = dataTable.domain
    valueRanges = TREE.getValueRanges(leaf)
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
    
def compareTrees(treeClassifier, combinedTreeClassifier, irisData, combinedData):
    print( \
        "*** Original Data Model Accuracy ***" + \
        "\nOriginal data: " + str(TREE.getModelAccuracy(treeClassifier, irisData)) + \
        "\nCombined data: " + str(TREE.getModelAccuracy(treeClassifier, combinedData)) + \
        "\n\n*** Combined Data Model Accuracy ***" + \
        "\nOriginal data: " + str(TREE.getModelAccuracy(combinedTreeClassifier, irisData)) + \
        "\nCombined data: " + str(TREE.getModelAccuracy(combinedTreeClassifier, combinedData)) \
        )
        
################################################################################
# Main Script
################################################################################
if __name__ == "__main__":
    try:
        # load data into table
        irisData = Orange.data.Table("iris")
        # create decision tree
        treeClassifier = Orange.classification.tree.TreeLearner(irisData, store_instances=True, max_depth=MAX_DEPTH)
        # interpolate new data
        leafNodes = TREE.getLeafNodes(treeClassifier.tree)
        newData = []
        for leaf in leafNodes:
            newData += interpolateLeafData(irisData, leaf)
        # create data table which includes the new data
        combinedData = copy.copy(irisData)
        for item in newData:
            combinedData.append(item)
        # build another decision tree including the new instances
        combinedTreeClassifier = Orange.classification.tree.TreeLearner(combinedData, store_instances=True, max_depth=MAX_DEPTH)
        # output decision trees to DOT files (used by Graphviz)
        if OUTPUT_DOT_FILES:
            TREE.outputTreeToDotFile(treeClassifier, RESULT_FILE_DIRECTORY, filePrefix='X1_original')
            TREE.outputTreeToDotFile(combinedTreeClassifier, RESULT_FILE_DIRECTORY, filePrefix='X1_combined')
        # compare accuracy of decision trees
        compareTrees(treeClassifier, combinedTreeClassifier, irisData, combinedData)
    except:
        traceback.print_exc(file=sys.stdout)
    