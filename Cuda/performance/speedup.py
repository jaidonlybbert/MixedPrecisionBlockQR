import matplotlib.pyplot as plt


rows = [100,200, 300, 400, 500, 600]
thread_1 = [10,20,30,40,50,60]
thread_2 = [5,10,15,20,25,30]
thread_4 = [1,2,3,4,5,6]

plt.plot(rows, thread_1,color='b',label='1 thread')
plt.plot(rows, thread_2,color='g',label='2 threads')
plt.plot(rows, thread_4,color='r',label='4 threads')

plt.xlabel('Matrix rows')
plt.ylabel('Speedup')
plt.legend(loc = "best")
plt.show()