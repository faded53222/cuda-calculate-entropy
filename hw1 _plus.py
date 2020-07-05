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
	__shared__ float shared2[(NSIZE-2)*(NSIZE-2)];
	if(A[row*(NSIZE-2)+col]>0){
		shared2[row*(NSIZE-2)+col]=-A[row*(NSIZE-2)+col]*log(float(A[row*(NSIZE-2)+col])/9)/9;
	}
	else{
		shared2[row*(NSIZE-2)+col]=0;
	}
	__syncthreads();
	for(int j=0;j<NSIZE-2;j++)
		for(int k=0;k<NSIZE-2;k++)
			A[(NSIZE-2)*j+k]=shared2[(NSIZE-2)*j+k];
}
__global__ void function3(float*A,float*B){
	//该函数将两二维数组相加
	const int row=threadIdx.y;
	const int col=threadIdx.x;
	const float temp_a = A[row*(NSIZE-2)+col];
	const float temp_b = B[row*(NSIZE-2)+col];
	__shared__ double shared3[(NSIZE-2)*(NSIZE-2)];
	shared3[row*(NSIZE-2)+col]=temp_a+temp_b;
	__syncthreads();
	for(int j=0;j<NSIZE-2;j++)
		for(int k=0;k<NSIZE-2;k++)
			A[(NSIZE-2)*j+k]=shared3[(NSIZE-2)*j+k];
}
__global__ void function4(float* R,float*A){
	//该函数将二维数组各项相加
	const int row=threadIdx.y;
	const int col=threadIdx.x;
	int h=threadIdx.z;
	__shared__ double shared4[16*(NSIZE-2)*(NSIZE-2)];
	if(h<8){
		shared4[h*(NSIZE-2)*(NSIZE-2)+row*(NSIZE-2)+col]=A[h*(NSIZE-2)*(NSIZE-2)+row*(NSIZE-2)+col]+A[(16-1-h)*(NSIZE-2)*(NSIZE-2)+row*(NSIZE-2)+col];
	}
	__syncthreads();
	if(h<4){
		shared4[h*(NSIZE-2)*(NSIZE-2)+row*(NSIZE-2)+col]=shared4[h*(NSIZE-2)*(NSIZE-2)+row*(NSIZE-2)+col]+shared4[(8-1-h)*(NSIZE-2)*(NSIZE-2)+row*(NSIZE-2)+col];
	}
	__syncthreads();
	if(h<2){
		shared4[h*(NSIZE-2)*(NSIZE-2)+row*(NSIZE-2)+col]=shared4[h*(NSIZE-2)*(NSIZE-2)+row*(NSIZE-2)+col]+shared4[(4-1-h)*(NSIZE-2)*(NSIZE-2)+row*(NSIZE-2)+col];
	}
	__syncthreads();
	if(h==0){
		shared4[0*(NSIZE-2)*(NSIZE-2)+row*(NSIZE-2)+col]=shared4[h*(NSIZE-2)*(NSIZE-2)+row*(NSIZE-2)+col]+shared4[(NSIZE-2)*(NSIZE-2)+row*(NSIZE-2)+col];
	}
	__syncthreads();
	for(int j=0;j<NSIZE-2;j++)
		for(int k=0;k<NSIZE-2;k++)
			R[(NSIZE-2)*j+k]=shared4[(NSIZE-2)*j+k];
}
""")
function1=mod.get_function("function1")
function2=mod.get_function("function2")
function3=mod.get_function("function3")
function4=mod.get_function("function4")
if __name__ == "__main__":
	a=np.random.randint(16,size=(N,N))
	print("target\n",a)
	r=np.empty([16,N,N],dtype=int)
	block=(N,N,1)
	grid =(1,1)
	start=timer()
	function1(drv.Out(r),drv.In(a),block=block,grid=grid)#得到不同点为中心的搜索框中的各数字的数量
	r=r.astype(np.float32)
	for i in range(len(r)):
		function2(drv.InOut(r[i]),block=block,grid=grid)#数量转化为熵
		#print(r[i])
		
	res=np.empty([N,N],dtype=int)
	res=res.astype(np.float32)
	function4(drv.Out(res),drv.In(r),block=(N,N,N),grid=(1,1))
	print("result\n")
	print(res)
	run_time=timer()-start  
	print("gpu run time %f seconds "%run_time)#记录时间 
