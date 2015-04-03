#!/usr/bin/python

import Orange, sys, traceback

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
    valueRanges = []
    
    return valueRanges
    
def buildNewInstances(valueRanges):
    newData = []
    
    return newData

################################################################################
# Main Script
################################################################################
if __name__ == "__main__":
    try:
        irisData = Orange.data.Table("iris")
        treeClassifier = Orange.classification.tree.TreeLearner(irisData)
        leafNodes = getLeafNodes(treeClassifier.tree)
        newData = []
        for leaf in leafNodes:
            valueRanges = getValueRanges(leaf)
            newData += buildNewInstances(valueRanges)
        # build another data model including the new instances
        # compare the new data model to the old one
    except:
        traceback.print_exc(file=sys.stdout)
    