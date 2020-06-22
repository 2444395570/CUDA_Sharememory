#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <memory>

using namespace std;

__global__ void gpu_shared_memory(float* d_a) {
	int i, index = threadIdx.x;
	float average, sum = 0.0f;
	//定义共享内存
	__shared__ float sh_arr[10];
	sh_arr[index] = d_a[index];

	//此条指令确保对于共享内存的所有写入操作已经完成。
	/*
	__syncthreads()是cuda的内建函数，用于块内线程通信.那些可以到达__syncthreads()的线程需要其他可以到达
	该点的线程，而不是等待块内所有其他线程。
	*/
	__syncthreads();

	for (int i = 0; i <= index; i++)
	{
		sum += sh_arr[i];
	}
	average = sum / (index + 1.0f);
	d_a[index] = average;


	sh_arr[index] = average;
	//这行代码是多余的，而且将会堆整个代码执行没有任何影响。最后一行代码将结果存放到了共享内存中。这行代码
	//对整体执行来说没有影响。因为共享内存的生存期到当前块执行完毕就结束了。
}

int main(int argc, char** argv) {
	float h_a[10];
	float* d_a;

	//初始化主机数组
	for (int i = 0; i < 10; i++)
	{
		h_a[i] = i;
	}

	//分配全局内存到设备上
	cudaMalloc((void**)&d_a, sizeof(float) * 10);
	//从主机内存到设备内存上复制数据
	cudaMemcpy((void*)d_a, (void*)h_a, sizeof(float) * 10,cudaMemcpyHostToDevice);
	gpu_shared_memory << <1, 10 >> > (d_a);
	//复制修改过的数组返回到主机
	cudaMemcpy((void*)h_a, (void*)d_a, sizeof(float) * 10, cudaMemcpyDeviceToHost);
	printf("Use of shared Memory on GPU:\n");
	for (int i = 0; i < 10; i++)
	{
		printf("The running average after %d element is %f\n", i, h_a[i]);
	}
	return 0;
}