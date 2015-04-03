#!/usr/bin/python

import Orange, sys, traceback

################################################################################
# Function Definitions
################################################################################


################################################################################
# Main Script
################################################################################
if __name__ == "__main__":
    try:
        irisData = Orange.data.Table("iris")
        treeClassifier = Orange.classification.tree.TreeLearner(irisData)
        # TO DO:
            # find leaf nodes and corresponding data instances
            # collect value ranges (or all possible discrete values) for all instance attributes per leaf node
            # build new instances with values in the available ranges (or discrete choices) for each leaf node
            # build another data model including the new instances
            # compare the new data model to the old one
    except:
        traceback.print_exc(file=sys.stdout)
    