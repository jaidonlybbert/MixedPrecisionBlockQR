import matplotlib.pyplot as plt
from util import readFromLogFile


cpu_runtime, cpu_flops, cpu_error = readFromLogFile('cpu_householder')
gpu_runtime, gpu_flops, gpu_error = readFromLogFile('gpu_block')

runtime_cpu = [item[1] for item in sorted(cpu_runtime.items())]
runtime_gpu =[item[1] for item in sorted(gpu_runtime.items())]

# run time plot
plt.plot(cpu_runtime.keys(), runtime_cpu,color='g',label='cpu')
plt.plot(gpu_runtime.keys(), runtime_gpu,color='b',label='gpu')
plt.xlabel('Matrix rows')
plt.ylabel('GFLOP/s')
plt.legend(loc = "best")

# flops plot
plt.figure()
flops_cpu = [item[1] for item in sorted(cpu_flops.items())]
flops_gpu = [item[1] for item in sorted(gpu_flops.items())]
plt.plot(cpu_flops.keys(), flops_cpu,color='g',label='cpu')
plt.plot(gpu_flops.keys(), flops_gpu,color='b',label='gpu')
plt.xlabel('Matrix rows')
plt.ylabel('GFLOP/s')
plt.legend(loc = "best")

# error plot
plt.figure()
error_cpu = [item[1] for item in sorted(cpu_error.items())]
error_gpu = [item[1] for item in sorted(gpu_error.items())]
plt.plot(cpu_error.keys(), error_cpu,color='g',label='cpu')
plt.plot(gpu_error.keys(), error_gpu,color='b',label='gpu')
plt.xlabel('Matrix rows')
plt.ylabel('Error(1e-8)')
plt.legend(loc = "best")
# gpu runtime
plt.figure()
plt.plot(gpu_runtime.keys(), runtime_gpu,color='b',label='gpu')
plt.xlabel('Matrix rows')
plt.ylabel('Runtime(s)')
plt.legend(loc = "best")

plt.show()

