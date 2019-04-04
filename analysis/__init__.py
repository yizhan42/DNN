from .calculate import *
from .figure import *
from .drawLossAcc import *
from .calculate import *
from .drawRoc import *
from .drawPR import *

# print and write log info
def write_and_print(string, writer, is_print=True, is_write=True):
    if(is_print):
        print(string)
    if(is_write):
        writer.write('{}\n'.format(string))