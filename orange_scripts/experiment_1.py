#!/usr/bin/python

import math, Orange, random, sys, traceback
import numpy as np
sys.path.append("./libraries")
import tree_functions as TREE

################################################################################
# Global Variables
################################################################################
NEW_DATA_PER_LEAF = 0.5
RESULT_FILE_DIRECTORY = "./results/"
OUTPUT_DOT_FILES = False
MAX_DEPTH = 3
NUMBINS = 10

################################################################################
# Function Definitions
################################################################################
def interpolateLeafData(dataTable, leaf):
    newData = []
    domain = dataTable.domain
    valueRanges = TREE.getValueRanges(leaf)
    allValues = TREE.getAllValues(leaf)
    numOfNewInstances = int(math.floor(len(leaf.instances) * NEW_DATA_PER_LEAF))
    for i in range(numOfNewInstances):
        newInstance = []
        for j, eachRange in enumerate(valueRanges):
            # continuous data
            if eachRange['type'] == Orange.feature.Type.Continuous:
                # average = sum(allValues[i])/len(allValues[i])
                hist, bin_edges = np.histogram(allValues[j], bins=NUMBINS, density=True)
                y_hist = hist.cumsum()/hist.cumsum().max()
                mRand = random.random()
                counter = 1
                while mRand > y_hist[counter]:
                    counter += 1
                newInstance.append(random.uniform(bin_edges[counter-1], bin_edges[counter]))
            # discrete data
            else:
                #if it is discrete, the change I made in get value range will make it select from a distribution
                newInstance.append(random.sample(eachRange['values'], 1)[0])
        orangeInstance = Orange.data.Instance(domain, newInstance)
        newData.append(orangeInstance)
    return newData
    
def compareTrees(treeClassifier1, treeClassifier2, data1, data2, label1, label2):
    print( \
        "*** " + label1 + " Data Model Accuracy ***" + \
        "\n" + label1 + " data: " + str(TREE.getModelAccuracy(treeClassifier1, data1)) + \
        "\n" + label2 + " data: " + str(TREE.getModelAccuracy(treeClassifier1, data2)) + \
        "\n\n*** " + label2 + " Data Model Accuracy ***" + \
        "\n" + label1 + " data: " + str(TREE.getModelAccuracy(treeClassifier2, data1)) + \
        "\n" + label2 + " data: " + str(TREE.getModelAccuracy(treeClassifier2, data2)) \
        )

################################################################################
# Main Script
################################################################################
if __name__ == "__main__":
    try:
        # load data into table
        irisDataTable = Orange.data.Table("iris")
        # create decision tree
        treeClassifier = Orange.classification.tree.TreeLearner(irisDataTable, store_instances=True, max_depth=MAX_DEPTH)
        # interpolate new data
        leafNodes = TREE.getLeafNodes(treeClassifier.tree)
        newData = []
        for leaf in leafNodes:
            newData += interpolateLeafData(irisDataTable, leaf)
        # create data table with only the new data
        newDataTable = Orange.data.Table(irisDataTable.domain)
        for item in newData:
            newDataTable.append(item)
        # build another decision tree using only the new instances
        newTreeClassifier = Orange.classification.tree.TreeLearner(newDataTable, store_instances=True, max_depth=MAX_DEPTH)
        # output decision trees to DOT files (used by Graphviz)
        if OUTPUT_DOT_FILES:
            TREE.outputTreeToDotFile(treeClassifier, RESULT_FILE_DIRECTORY, filePrefix='X1_original')
            TREE.outputTreeToDotFile(newTreeClassifier, RESULT_FILE_DIRECTORY, filePrefix='X1_new')
        # compare accuracy of decision trees
        compareTrees(treeClassifier, newTreeClassifier, irisDataTable, newDataTable, "Original", "New")
    except:
        traceback.print_exc(file=sys.stdout)
