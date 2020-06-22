#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <memory>

using namespace std;

__global__ void gpu_shared_memory(float* d_a) {
	int i, index = threadIdx.x;
	float average, sum = 0.0f;
	//���干���ڴ�
	__shared__ float sh_arr[10];
	sh_arr[index] = d_a[index];

	//����ָ��ȷ�����ڹ����ڴ������д������Ѿ���ɡ�
	/*
	__syncthreads()��cuda���ڽ����������ڿ����߳�ͨ��.��Щ���Ե���__syncthreads()���߳���Ҫ�������Ե���
	�õ���̣߳������ǵȴ��������������̡߳�
	*/
	__syncthreads();

	for (int i = 0; i <= index; i++)
	{
		sum += sh_arr[i];
	}
	average = sum / (index + 1.0f);
	d_a[index] = average;


	sh_arr[index] = average;
	//���д����Ƕ���ģ����ҽ������������ִ��û���κ�Ӱ�졣���һ�д��뽫�����ŵ��˹����ڴ��С����д���
	//������ִ����˵û��Ӱ�졣��Ϊ�����ڴ�������ڵ���ǰ��ִ����Ͼͽ����ˡ�
}

int main(int argc, char** argv) {
	float h_a[10];
	float* d_a;

	//��ʼ����������
	for (int i = 0; i < 10; i++)
	{
		h_a[i] = i;
	}

	//����ȫ���ڴ浽�豸��
	cudaMalloc((void**)&d_a, sizeof(float) * 10);
	//�������ڴ浽�豸�ڴ��ϸ�������
	cudaMemcpy((void*)d_a, (void*)h_a, sizeof(float) * 10,cudaMemcpyHostToDevice);
	gpu_shared_memory << <1, 10 >> > (d_a);
	//�����޸Ĺ������鷵�ص�����
	cudaMemcpy((void*)h_a, (void*)d_a, sizeof(float) * 10, cudaMemcpyDeviceToHost);
	printf("Use of shared Memory on GPU:\n");
	for (int i = 0; i < 10; i++)
	{
		printf("The running average after %d element is %f\n", i, h_a[i]);
	}
	return 0;
}