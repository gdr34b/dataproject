#!/usr/bin/python

import Orange, sys, traceback

################################################################################
# Main Script
################################################################################
if __name__ == "__main__":
    try:
        irisData = Orange.data.Table("iris")
        discreteIrisData = Orange.data.discretization.DiscretizeTable(irisData, method=Orange.feature.discretization.EqualFreq(n=3))
        rules = Orange.associate.AssociationRulesInducer(discreteIrisData, support=0.3)
        for r in rules:
            print(r)
    except:
        traceback.print_exc(file=sys.stdout)
    