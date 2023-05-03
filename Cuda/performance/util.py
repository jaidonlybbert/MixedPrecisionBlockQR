import csv

def readFromLogFile():
    with open('./cuda/logFile.txt', newline='') as csvfile:
        spamreader = csv.reader(csvfile)
        for row in spamreader:
            print(row)
