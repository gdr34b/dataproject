#!/usr/bin/python

import os, subprocess, sys, traceback

################################################################################
# Global Variables
################################################################################
RESULT_FILE_DIRECTORY = "./results/"

################################################################################
# Function Definitions
################################################################################
def convertDotFileToImage(dotFilePath):
    imageFilePath = dotFilePath.replace(".dot", ".png")
    if not os.path.isfile(imageFilePath):
        command = "dot -Tpng " + dotFilePath + " -o" + imageFilePath
        subprocess.call(command)

def convertAllDOTFiles(fileDirectory):
    contents = os.listdir(fileDirectory)
    for eachFileName in contents:
        if ".dot" in eachFileName:
            convertDotFileToImage(fileDirectory + eachFileName)

################################################################################
# Main Script
################################################################################
if __name__ == "__main__":
    try:
        convertAllDOTFiles(RESULT_FILE_DIRECTORY)
    except:
        traceback.print_exc(file=sys.stdout)
        