// OpenCL kernel. Each work item takes care of one element of c
#pragma OPENCL EXTENSION cl_intel_printf : enable

#define beta1 0.9
#define beta2 0.999
#define eps 0.0000001

__kernel void convolution(__global float *dest, int dest_n, int dest_m, int dest_depth,
							__global float *src, int src_n, int src_m, int src_depth,
							__global float *filter, int filter_n, int filter_m, int filter_depth, float lr,
							int isbias, __global float *bias)
{

	int i, j, de_dep, row, col, hfn, hfm;
	float s;
	int r;
	int id = get_global_id(0);
	if(id>=dest_n*dest_m*dest_depth) return;
	de_dep = id/(dest_m*dest_n);
	i = (id%(dest_m*dest_n))/dest_m;
	j = (id%(dest_m*dest_n))%dest_m;

	//printf("%d %d\n",src_n, src_m);
	//printf("%d %d %d\n",dest_depth, src_depth, filter_depth);
	s=0;
	for(row=0;row<filter_n;row++){
		for(col=0;col<filter_m;col++){
			for(r=0;r<src_depth;r++){
				if(lr==0){
					s += src[r*(src_n*src_m)+ (((src_n+i+row-1)%src_n)*src_m) + (src_m+j+col-1)%src_m]
					 * filter[de_dep*(filter_m*filter_n*src_depth) + r*(filter_m*filter_n)+((filter_n-1-row)*filter_m)+(filter_m-1-col)];
				}
				else{
					s += src[r*(src_n*src_m)+ (((src_n+i+row-1)%src_n)*src_m) + (src_m+j+col-1)%src_m]
					 * filter[r*(filter_m*filter_n*dest_depth) + de_dep*(filter_m*filter_n)+(row*filter_m)+col];
				}
			}
		}
	}
	if(isbias)
		dest[id] = s + bias[de_dep];
	else
		dest[id] = s;
}     

__kernel void convolution_backpropagation(__global float *dest, int dest_n, int dest_m, int dest_src_depth, int dest_filter_depth,
							__global float *src, int src_n, int src_m, int src_depth,
							__global float *filter, int filter_n, int filter_m, int filter_depth, float lr,
							int isbias, __global float *bias)
{

	int i, j, sr_dep, de_dep, row, col, hfn, hfm;
	float s;
	int r;
	int id = get_global_id(0);
	if(id>=dest_n*dest_m*dest_src_depth*dest_filter_depth) return;
	sr_dep = id/(dest_m*dest_n*dest_filter_depth);
	de_dep = (id%(dest_m*dest_n*dest_filter_depth))/(dest_m*dest_n);
	i = ((id%(dest_m*dest_n*dest_filter_depth))%(dest_m*dest_n))/dest_m;
	j = ((id%(dest_m*dest_n*dest_filter_depth))%(dest_m*dest_n))%dest_m;

	s=0;
	for(row=0;row<filter_n;row++){
		for(col=0;col<filter_m;col++){
			s += src[sr_dep*(src_n*src_m)+ (((src_n+i+row-1)%src_n)*src_m) + (src_m+j+col-1)%src_m]
			 * filter[de_dep*(filter_m*filter_n)+(row*filter_m)+col];
		}
	}
	if(isbias)
		dest[id] = s*(float)lr + bias[de_dep];
	else
		dest[id] = s*(float)lr;
}     

__kernel void matdotGPU(__global float* dest, __global float *op1, __global float *op2, int n, int m, int k)
{
	int i, j, l;
	int id = get_global_id(0);
	if(id>=n*m) return;
	i = id/m;
	j = id%m;
	dest[i*m+j]=0;
	for(l=0;l<k;l++){
		dest[i*m+j] += op1[i*k + l] * op2[l*m + j];
	}
}

__kernel void AdamGPU(__global float* d, __global float *dx, __global float *b, __global float*m, int t, float lr)
{
	int i, j, l;
	float bt, mt;
	int id = get_global_id(0);
	b[id] = beta1*b[id] - (float)(1.0-beta1)*dx[id];
	m[id] = beta2*m[id] + (float)(1.0-beta2)*dx[id]*dx[id];
	bt = b[id] / (1.0 - pow((float)beta1,(float)t));
	mt = m[id] / (1.0 - pow((float)beta2,(float)t));
	d[id] -= lr * bt / (float)((sqrt(mt))+eps);
}