import matplotlib.pyplot as plt


rows = [100,200, 300, 400, 500, 600]
flops_cpu = [10,20,30,40,50,60]
flops_gpu = [50,100,150,200,250,300]
plt.plot(rows, flops_cpu,color='g',label='cpu')
plt.plot(rows, flops_gpu,color='b',label='gpu')
plt.xlabel('Matrix rows')
plt.ylabel('GFLOP/s')
plt.legend(loc = "best")
plt.show()