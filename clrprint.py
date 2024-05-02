"""
Library to print color output
"""

import numpy as np
import re


class bcolors:
    """ Definition of colors used to color terminal output. """
    HEADER = '\033[95m'     # Pink
    OKBLUE = '\033[94m'     # Blue
    OKCYAN = '\033[96m'     # Cyan
    OKGREEN = '\033[92m'    # Green
    WARNING = '\033[93m'    # Yellow
    FAIL = '\033[91m'       # Red
    ENDC = '\033[0m'        # White
    BOLD = '\033[1m'        # Bold
    UNDERLINE = '\033[4m'   # Underline


def bold(string:str):
    """ Format string to bold """
    return bcolors.BOLD+string+bcolors.ENDC

def green(string):
    """ Format green color """
    return bcolors.OKGREEN + string + bcolors.ENDC

def red(string):
    """ Format red color """
    return bcolors.FAIL + string + bcolors.ENDC

def yellow(string):
    """ Format yellow color """
    return bcolors.WARNING + string + bcolors.ENDC

def blue(string):
    """ Format blue color """
    return bcolors.OKBLUE + string + bcolors.ENDC

# Print functions
def printc(msg:str,fc:str=None,fw:str=None,tag:str=None,**kwargs):
    """
    Format string color and weight and print:
    fc = 'g','r','y','b'
    fw = 'b'
    tag = 'w', 'e'
    """

    if tag=='w' or tag=='warning':
        fc = 'y'
        printc("[WARNING]",fc='y',fw='b',end=' ')
    elif tag=='e' or tag=='error':
        fc = 'r'
        printc("[ERROR]",fc='r',fw='b',end=' ')
        
    if fc=='g' or fc=='green':
        msg = green(msg)
    elif fc=='r' or fc=='red':
        msg = red(msg)
    elif fc=='y' or fc=='yellow' or fc=='o' or fc=='orange':
        msg = yellow(msg)
    elif fc=='b' or fc=='blue':
        msg = blue(msg)
    
    if fw=='b' or fw=='bold':
        msg = bold(msg)


    print(msg,**kwargs)

def printW(msg):
    """ Print warning message """
    print(bcolors.WARNING + msg + bcolors.ENDC)

def printE(msg):
    """ Print error message """
    print(bcolors.FAIL + msg + bcolors.ENDC)

def printOK(msg):
    """ Print OK message """
    print(bcolors.OKGREEN + msg + bcolors.ENDC)

def printException(ex):
    template = "An exception of type {0} occurred. Arguments:\n{1!r}"
    message = template.format(type(ex).__name__, ex.args)
    print(message)

def backspace(n:int):
    """ Remove last `n` characters from console """
    for i in range(n):
        print("\b",end='',flush=True)

def clearNum(n:int):
    """ Clear integer number printed to console """
    backspace(1 + int(np.floor(np.log10(n if n>0 else 1))))

#     i = 555
# print("i =",i)
# print(i,end="")
# i += 1
# end = 1+int(np.floor(np.log10(i-1 if i>1 else 2)))
# for j in range(end): print("\b",end="",flush=True)
# print(i)
# print("end =",end)