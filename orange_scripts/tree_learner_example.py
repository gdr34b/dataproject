#!/usr/bin/python

import Orange, sys, traceback

################################################################################
# Main Script
################################################################################
if __name__ == "__main__":
    try:
        irisData = Orange.data.Table("iris")
        treeClassifier = Orange.classification.tree.TreeLearner(irisData)
        print(treeClassifier)
    except:
        traceback.print_exc(file=sys.stdout)
    