#!/usr/bin/python

import math, Orange, os, random, sys, traceback
import numpy as np
sys.path.append("./libraries")
import tree_functions as TREE

################################################################################
# Global Variables
################################################################################
NEW_DATA_PER_LEAF = 0.5
RESULT_FILE_DIRECTORY = "./results/"
OUTPUT_DOT_FILES = True
NUMBINS = 10
experimentNumber = 0

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
    accuracy11 = TREE.getModelAccuracy(treeClassifier1, data1)
    accuracy12 = TREE.getModelAccuracy(treeClassifier1, data2)
    accuracy21 = TREE.getModelAccuracy(treeClassifier2, data1)
    accuracy22 = TREE.getModelAccuracy(treeClassifier2, data2)
    print( \
        "-- " + label1 + " Data Model Accuracy" + \
        "\n  " + label1 + " data: " + str(accuracy11) + \
        "\n  " + label2 + " data: " + str(accuracy12) + \
        "\n-- " + label2 + " Data Model Accuracy" + \
        "\n  " + label1 + " data: " + str(accuracy21) + \
        "\n  " + label2 + " data: " + str(accuracy22) + \
        "\n" \
        )
    return {'tree1_data1': accuracy11, 'tree1_data2': accuracy12, 'tree2_data1': accuracy21, 'tree2_data2': accuracy22}
        
def runExperiment(datasetName, maxDepth):
    global experimentNumber
    experimentNumber += 1
    print( \
        "*** Experiment " + str(experimentNumber) + " ***" + \
        "\nDataset name: " + datasetName + \
        "\nMax depth: " + str(maxDepth) \
        )
    # load data into table
    irisDataTable = Orange.data.Table(datasetName)
    # create decision tree
    treeClassifier = Orange.classification.tree.TreeLearner(irisDataTable, store_instances=True, max_depth=maxDepth)
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
    newTreeClassifier = Orange.classification.tree.TreeLearner(newDataTable, store_instances=True, max_depth=maxDepth)
    # output decision trees to DOT files (used by Graphviz)
    if OUTPUT_DOT_FILES:
        prefix = datasetName + '_d' + str(maxDepth) + '_'
        TREE.outputTreeToDotFile(treeClassifier, RESULT_FILE_DIRECTORY, filePrefix=prefix + 'original')
        TREE.outputTreeToDotFile(newTreeClassifier, RESULT_FILE_DIRECTORY, filePrefix=prefix + 'new')
    # compare accuracy of decision trees
    return compareTrees(treeClassifier, newTreeClassifier, irisDataTable, newDataTable, "Original", "New")
    
def outputResultSummary(results):
    if len(results) > 0:
        # determine summary file name
        filePrefix = 'summary'
        if not os.path.isdir(RESULT_FILE_DIRECTORY):
            os.makedirs(RESULT_FILE_DIRECTORY)
        similarFiles = [x for x in os.listdir(RESULT_FILE_DIRECTORY) if filePrefix in x]
        fileNumber = len(similarFiles) + 1
        summaryFileName = filePrefix + "_" + str(fileNumber) + ".csv"
        # output results to file
        with open(RESULT_FILE_DIRECTORY + summaryFileName, 'w') as f:
            f.write(','.join(sorted(results[0].keys())) + '\n')
            for eachResult in results:
                orderedValues = []
                for key in sorted(eachResult.keys()):
                    orderedValues.append(eachResult[key])
                orderedStrings = [str(x) for x in orderedValues]
                f.write(','.join(orderedStrings) + '\n')
    
def runExperimentSet(allDatasetNames, maxDepthRange):
    results = []
    for datasetName in allDatasetNames:
        for maxDepth in range(maxDepthRange[0], maxDepthRange[1] + 1):
            treeComparison = runExperiment(datasetName, maxDepth)
            newResult = {'datasetName': datasetName, 'maxDepth': maxDepth}
            for key, value in treeComparison.iteritems():
                newResult[key] = value
            results.append(newResult)
    outputResultSummary(results)

################################################################################
# Main Script
################################################################################
if __name__ == "__main__":
    try:
        allDatasetNames = ['adult_sample', 'car', 'iris', 'lung-cancer', 'tic_tac_toe', 'voting']
        runExperimentSet(allDatasetNames, maxDepthRange=(1, 5))
        
    except:
        traceback.print_exc(file=sys.stdout)
