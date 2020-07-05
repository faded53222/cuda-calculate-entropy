import pycuda.driver as drv
import pycuda.tools
import pycuda.autoinit
import pycuda.gpuarray as gpuarray
import numpy as np
import numpy.linalg as la
import copy
from pycuda.compiler import SourceModule
from timeit import default_timer as timer
import warnings
warnings.filterwarnings('ignore')
N=10
mod = SourceModule("""
#define NSIZE 12
__global__ void function1(int*R,int *A){
	//由n*n的初始数组 得到16*(n+2)*(n+2)的数组 第一维对应各个数字
	//A[a,b]处的一个数字为x,则 R[x][a][b]处和此处的周围的所有位置加一
	//也就是说R[x][a][b]就是以此点为中心的搜索框中的数字x的数量
	__shared__ int shared[16*NSIZE*NSIZE];
	if(threadIdx.y==0 && threadIdx.x==0){
		for(int i=0;i<=15;i++)
			for(int j=0;j<NSIZE;j++)
				for(int k=0;k<NSIZE;k++)
					shared[NSIZE*NSIZE*i+NSIZE*j+k]=0;
	}
	__syncthreads();
	int row=threadIdx.y;
	int col=threadIdx.x;
	int map=NSIZE*NSIZE*A[(NSIZE-2)*row+col];
	++shared[map+NSIZE*row+col];
	__syncthreads();
	++shared[map+NSIZE*row+col+1];
	__syncthreads();
	++shared[map+NSIZE*row+col+2];
	__syncthreads();
	++shared[map+NSIZE*(row+1)+col];
	__syncthreads();
	++shared[map+NSIZE*(row+1)+col+1];
	__syncthreads();
	++shared[map+NSIZE*(row+1)+col+2];
	__syncthreads();
	++shared[map+NSIZE*(row+2)+col];
	__syncthreads();
	++shared[map+NSIZE*(row+2)+col+1];
	__syncthreads();
	++shared[map+NSIZE*(row+2)+col+2];
	__syncthreads();
	if(row==0 && col==0)
		for(int i=0;i<=15;i++)
			for(int j=0;j<NSIZE-2;j++)
				for(int k=0;k<NSIZE-2;k++)
					R[(NSIZE-2)*(NSIZE-2)*i+(NSIZE-2)*j+k]=shared[NSIZE*NSIZE*i+NSIZE*(j+1)+k+1];
}
__global__ void function2(float*A){
	//该函数将[x,a,b]处的 数量 映射为 -数量/9*log(数量/9)
	//也就是数量映射为熵
	int row=threadIdx.y;
	int col=threadIdx.x;
	if(A[row*(NSIZE-2)+col]>0)
		A[row*(NSIZE-2)+col]=-A[row*(NSIZE-2)+col]*log(float(A[row*(NSIZE-2)+col]/9))/9;
}
__global__ void function3(float*A,float*B){
	//该函数将两二维数组相加
	int row=threadIdx.y;
	int col=threadIdx.x;
	A[row*(NSIZE-2)+col]+=B[row*(NSIZE-2)+col];
}
""")
function1=mod.get_function("function1")
function2=mod.get_function("function2")
function3=mod.get_function("function3")
if __name__ == "__main__":
	a=np.random.randint(16,size=(N,N))
	print("target\n",a)
	r=np.empty([16,N,N],dtype=int)
	block=(N,N,1)
	grid =(1,1)
	start=timer()
	function1(drv.Out(r),drv.In(a),block=block,grid=grid)#得到不同点为中心的搜索框中的各数字的数量
	r=r.astype(np.float32)
	#List=[-1,0,0.3010,0.4771,0.6020,0.6989,0.7781,0.8450,0.903,0.9542]
	#List=np.array(List)
	#List.astype(np.float32)
	#List_gpu=drv.mem_alloc(List.nbytes)
	#drv.memcpy_htod(List_gpu,List)
	for i in range(len(r)):
		#print(r[i])
		#function2(drv.InOut(each),List_gpu,block=block,grid=grid)
		function2(drv.InOut(r[i]),block=block,grid=grid)#数量转化为熵
		#print(r[i])
	max_len=16;
	while 1:#通过对折的方法相加全部元素
		for i in range(int(max_len/2)):
			function3(drv.InOut(r[i]),drv.In(r[max_len-1-i]),block=block,grid=grid)
		max_len=int(max_len/2)
		if max_len==1:
			break;
	print("result\n")
	for i in range(1,N-1):#打印结果
		for j in range(1,N-1):
			print(r[0][i][j],"   ",end="")
		print("\n")
	run_time=timer()-start  
	print("gpu run time %f seconds "%run_time)#记录时间 
