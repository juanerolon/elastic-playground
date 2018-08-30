#------------------------------------
#Elastic dump call using sys calls
#------------------------------------
#@Juan E. Rolon


import sys
import subprocess
import shlex
import argparse

#Test sys.argv

print("No. of arguments is {}".format(len(sys.argv)))

try:
    print("Input taken from within shell script is: {}".format(sys.argv[1]))
except:
    print("No argument provided")
try:
    print("Input taken from within shell script is:  {}".format(sys.argv[2]))
except:
    print("No argument provided ")
try:
    print("Input taken from within the console env is : {}".format(sys.argv[3]))
except:
    print("No argument provided")

#Execute a system command to list current directory; make sure tokens file exists

cmd = "ls"

#output below is a stream of bytes which needs to be decoded and converted to a string
output = subprocess.Popen(cmd,shell=True,stdout=subprocess.PIPE).stdout.read()
#output below the decoded byte stream using utf-8
output = output.decode("utf-8")
print(output)

#Execute a curl request to check whether the elastic cluster/node is alive

with open('tokens', 'r') as tokens_file:
    tokens = tokens_file.read().replace('\n', '')

cmd = 'curl -u "{}" http://localhost:9200 -vL'.format(tokens)

#output below is a stream of bytes which needs to be decoded and converted to a string
output = subprocess.Popen(cmd,shell=True,stdout=subprocess.PIPE).stdout.read()
#output below the decoded byte stream using utf-8
output = output.decode("utf-8")
print(output)


cmd = "elasticdump --input=http://{}@localhost:9200/product --output=/Users/juanerolon/Desktop/ingalls/curl_examples/scripted_dump.json".format(tokens)

#output below is a stream of bytes which needs to be decoded and converted to a string
output = subprocess.Popen(cmd,shell=True,stdout=subprocess.PIPE).stdout.read()
#output below the decoded byte stream using utf-8
output = output.decode("utf-8")
print(output)

