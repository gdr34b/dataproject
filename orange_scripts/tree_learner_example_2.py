#!/usr/bin/python

import Orange, sys, traceback

################################################################################
# Function Definitions
################################################################################
def traverseTree(node, level=0):
    if not node:
        return
    if node.branch_selector:
        nodeDesc = node.branch_selector.class_var.name
        for i in range(len(node.branches)):
            print("|  "*level + "%s %s" % (nodeDesc, node.branch_descriptions[i]))
            traverseTree(node.branches[i], level + 1)
    else:
        majorClass = node.node_classifier.default_value
        print("|  "*level + "--> %s" % (majorClass))
            
# from http://docs.orange.biolab.si/reference/rst/Orange.classification.tree.html#tree-structure
def print_tree0(node, level):
    if not node:
        print " "*level + "<null node>"
        return
    if node.branch_selector:
        node_desc = node.branch_selector.class_var.name
        node_cont = node.distribution
        print "\n" + "   "*level + "%s (%s)" % (node_desc, node_cont),
        for i in range(len(node.branches)):
            print "\n" + "   "*level + ": %s" % node.branch_descriptions[i],
            print_tree0(node.branches[i], level+1)
    else:
        node_cont = node.distribution
        major_class = node.node_classifier.default_value
        print "--> %s (%s) " % (major_class, node_cont),

################################################################################
# Main Script
################################################################################
if __name__ == "__main__":
    try:
        irisData = Orange.data.Table("iris")
        treeClassifier = Orange.classification.tree.TreeLearner(irisData)
        # print(treeClassifier)
        traverseTree(treeClassifier.tree)
        # print_tree0(treeClassifier.tree, 0)
    except:
        traceback.print_exc(file=sys.stdout)
    