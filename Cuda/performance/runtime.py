import matplotlib.pyplot as plt


rows = [100,200, 300, 400, 500, 600]
runtime_cpu = [10,20,30,40,50,60]
runtime_gpu = [5,10,15,20,25,30]
plt.plot(rows, runtime_cpu,color='g',label='cpu')
plt.plot(rows, runtime_gpu,color='b',label='gpu')
plt.xlabel('Matrix rows')
plt.ylabel('Runtime(s)')
plt.legend(loc = "best")
plt.show()