import csv
from collections import defaultdict
import math

def getAverage(data, metrixIndex):
    result = defaultdict(list)
    for item in data:
        rows = int(float(item[0]))
        try:
            value = float(item[metrixIndex])
            if rows > 0 and not math.isinf(value):
                result[rows].append(value)
        except:
            print('{} is not a valid number'.format(item[metrixIndex]))
    result = { key: sum(result[key]) // len(result[key]) for key in result }
    return result

def readFromLogFile(filename):
    with open('./cuda/log/' + filename + '.txt', newline='') as csvfile:
        spamreader = csv.reader(csvfile)
        data = list(spamreader)
        data.pop(0)
        # filter invalid data

        runtime = getAverage(data, 2)
        flops = getAverage(data, 3)
        print(runtime)
        return runtime, flops

