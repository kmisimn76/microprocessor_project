
#include "Matrix.h"
#define PI 3.141592653589793238

#define CONV1 64
#define CONV2 32
#define FILTER_SIZE 3

#define CNN_LETTER 0
#define CNN_PICT 1

int IMAGE_SIZE; //28
int DATAS; //784
int DATASN; //28
int CONV1_SIZE;// 28
int CONV2_SIZE;// 14
int INPUTS; //CONV2*7*7
int HIDDENS; //100
int OUTPUTS; //10


//=====activation func

float ReLU(float x){
	return (x>0)?x:0.01*x;
}
void softmax(float* x, int n){
	int i;
	double exp_a[20], sum_exp_a = 0;
	float max = -0.00001;
	for(i=0;i<n;i++)
		if(max<x[i]) max=x[i];
    for(i=0;i<n;i++){
    	exp_a[i] = exp(x[i]-max);
    	sum_exp_a += exp_a[i];
    }
    for(i=0;i<n;i++){
    	x[i] = exp_a[i]/sum_exp_a;
    }
}

//=====activation func

//=====DeepNeuralNet

int launch_the_kernel(cl_context&       context,
					  cl_command_queue& queue,
					  cl_kernel&        kernel,
					  size_t            globalSize,
					  size_t            localSize,
					  cl_mem&           d_dest,
					  cl_mem&           d_src,
					  cl_mem&			d_filter,
					  cl_mem&			d_bias,
					  float*			dest,
					  int 				dest_n,
					  int 				dest_m,
					  int 				dest_depth,
					  float*			src,
					  int 				src_n,
					  int 				src_m,
					  int 				src_depth,
					  float*			filter,
					  int 				filter_n,
					  int 				filter_m,
					  int 				filter_depth,
					  float				lr,
					  int 				isbias,
					  float* 			bias
					  ){

	cl_int err;

	// Create the input and output arrays in device memory for our calculation
    d_dest = clCreateBuffer(context, CL_MEM_WRITE_ONLY, dest_n*dest_m*dest_depth*sizeof(float), NULL, &err);
	checkCL(err);
    d_src = clCreateBuffer(context, CL_MEM_READ_ONLY, src_n*src_m*src_depth*sizeof(float), NULL, &err);
	checkCL(err);
    d_filter = clCreateBuffer(context, CL_MEM_READ_ONLY, filter_n*filter_m*filter_depth*sizeof(float), NULL, &err);
	checkCL(err);
	d_bias = clCreateBuffer(context, CL_MEM_READ_ONLY, filter_depth*sizeof(float), NULL, &err);
	checkCL(err);

    // Write our data set into the input array in device memory
    checkCL(clEnqueueWriteBuffer(queue, d_src, CL_TRUE, 0,
                                   src_n*src_m*src_depth*sizeof(float), src, 0, NULL, NULL));
    checkCL(clEnqueueWriteBuffer(queue, d_filter, CL_TRUE, 0,
                                   filter_n*filter_m*filter_depth*sizeof(float), filter, 0, NULL, NULL));
    checkCL(clEnqueueWriteBuffer(queue, d_bias, CL_TRUE, 0,
    	                           filter_depth*sizeof(float), bias, 0, NULL, NULL));

    // Set the arguments to our compute kernel
    checkCL(clSetKernelArg(kernel,  0, sizeof(cl_mem), &d_dest));
    checkCL(clSetKernelArg(kernel,  1, sizeof(int), &dest_n));
    checkCL(clSetKernelArg(kernel,  2, sizeof(int), &dest_m));
    checkCL(clSetKernelArg(kernel,  3, sizeof(int), &dest_depth));
    checkCL(clSetKernelArg(kernel,  4, sizeof(cl_mem), &d_src));
    checkCL(clSetKernelArg(kernel,  5, sizeof(int), &src_n));
    checkCL(clSetKernelArg(kernel,  6, sizeof(int), &src_m));
    checkCL(clSetKernelArg(kernel,  7, sizeof(int), &src_depth));
    checkCL(clSetKernelArg(kernel,  8, sizeof(cl_mem), &d_filter));
    checkCL(clSetKernelArg(kernel,  9, sizeof(int), &filter_n));
    checkCL(clSetKernelArg(kernel, 10, sizeof(int), &filter_m));
    checkCL(clSetKernelArg(kernel, 11, sizeof(int), &filter_depth));
    checkCL(clSetKernelArg(kernel, 12, sizeof(float), &lr));
    checkCL(clSetKernelArg(kernel, 13, sizeof(int), &isbias));
    checkCL(clSetKernelArg(kernel, 14, sizeof(cl_mem), &d_bias));

    // Execute the kernel over the entire range of the data set
    checkCL(clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &globalSize, &localSize,
                                                              0, NULL, NULL));
     // Wait for the command queue to get serviced before reading back results
    checkCL(clFinish(queue));

    // Read the results from the device
    checkCL(clEnqueueReadBuffer(queue, d_dest, CL_TRUE, 0,
                                dest_n*dest_m*dest_depth*sizeof(float), dest, 0, NULL, NULL ));

	checkCL(clReleaseMemObject(d_dest));
	checkCL(clReleaseMemObject(d_src));
	checkCL(clReleaseMemObject(d_filter));
	checkCL(clReleaseMemObject(d_bias));

	return 0;
}

int launch_the_kernel_dot(cl_context&       context,
					  cl_command_queue& queue,
					  cl_kernel&        kernel,
					  size_t            globalSize,
					  size_t            localSize,
					  cl_mem&			d_d,
					  cl_mem&			d_a,
					  cl_mem&			d_b,
					  float*			dest,
					  float* 			a1,
					  float* 			a2,
					  int 				n,
					  int 				m,
					  int 				k
					  ){

	cl_int err;

	// Create the input and output arrays in device memory for our calculation
    d_d = clCreateBuffer(context, CL_MEM_WRITE_ONLY, n*m*sizeof(float), NULL, &err);
	checkCL(err);
    d_a = clCreateBuffer(context, CL_MEM_READ_ONLY, n*k*sizeof(float), NULL, &err);
	checkCL(err);
    d_b = clCreateBuffer(context, CL_MEM_READ_ONLY, k*m*sizeof(float), NULL, &err);
	checkCL(err);

    // Write our data set into the input array in device memory
    checkCL(clEnqueueWriteBuffer(queue, d_a, CL_TRUE, 0,
                                   n*k*sizeof(float), a1, 0, NULL, NULL));
    
    checkCL(clEnqueueWriteBuffer(queue, d_b, CL_TRUE, 0,
                                   k*m*sizeof(float), a2, 0, NULL, NULL));

    // Set the arguments to our compute kernel
    checkCL(clSetKernelArg(kernel,  0, sizeof(cl_mem), &d_d));
    checkCL(clSetKernelArg(kernel,  1, sizeof(cl_mem), &d_a));
    checkCL(clSetKernelArg(kernel,  2, sizeof(cl_mem), &d_b));
    checkCL(clSetKernelArg(kernel,  3, sizeof(int), &n));
    checkCL(clSetKernelArg(kernel,  4, sizeof(int), &m));
    checkCL(clSetKernelArg(kernel,  5, sizeof(int), &k));

    // Execute the kernel over the entire range of the data set
    checkCL(clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &globalSize, &localSize,
                                                              0, NULL, NULL));
     // Wait for the command queue to get serviced before reading back results
    checkCL(clFinish(queue));

    // Read the results from the device
    checkCL(clEnqueueReadBuffer(queue, d_d, CL_TRUE, 0,
                                n*m*sizeof(float), dest, 0, NULL, NULL ));

	checkCL(clReleaseMemObject(d_d));
	checkCL(clReleaseMemObject(d_a));
	checkCL(clReleaseMemObject(d_b));

	return 0;
}

void arr_convolution2D(cl_context& context, cl_command_queue& queue, cl_kernel& kernel,
						float* dest_mat, int dest_n, int dest_m, int dest_depth,
						float* src_mat, int src_n, int src_m, int src_depth,
						float* filter_mat, int filter_n, int filter_m, int filter_depth,
						float lr, int isbias, float *bias){

	cl_mem d_dest;
	cl_mem d_src;
	cl_mem d_filter;
	cl_mem d_bias;
    size_t globalSize, localSize, grid;
    if(!isbias)
    	bias = (float*)malloc(sizeof(float)*filter_depth);

    // Number of work items in each local work group
    localSize = 128; //TODO
	int n_pix = dest_n * dest_m * dest_depth;

    // Number of total work items - localSize must be devisor
	grid = (n_pix%localSize)? (n_pix/localSize)+1 : n_pix/localSize;
    globalSize = grid*localSize;

	launch_the_kernel(context, queue, kernel, globalSize, localSize,
					  d_dest, d_src, d_filter, d_bias,
					  dest_mat, dest_n, dest_m, dest_depth,
					  src_mat, src_n, src_m, src_depth,
					  filter_mat, filter_n,	filter_m, filter_depth,
					  lr, isbias, bias);

	if(!isbias)
		free(bias);

}
void dotGPU(cl_context& context, cl_command_queue& queue, cl_kernel& kernel,
			mat* result, mat* a, mat* b){

	cl_mem d_d;
	cl_mem d_a;
	cl_mem d_b;
    size_t globalSize, localSize, grid;

    int i, j;
	if(a->m!=b->n) return;

    // Number of work items in each local work group
    localSize = 128; //TODO
	int n_pix = a->n * b->m;
    // Number of total work items - localSize must be devisor
	grid = (n_pix%localSize)? (n_pix/localSize)+1 : n_pix/localSize;
    globalSize = grid*localSize;

	launch_the_kernel_dot(context, queue, kernel, globalSize, localSize, d_d, d_a, d_b,
					  result->ma, a->ma, b->ma, a->n, b->m, a->m);
}


void array_trans(float* array, int n, float (*func)(float)){
	int i;
	for(i=0;i<n;i++){
		array[i] = func(array[i]);
	}
}

float normalize(unsigned char x){
	//return ((float)x/256.0)*0.99+0.01;
	return (x>0)?0.994:0.01;
}

float wdc1[CONV1*9];
float wc1c2[CONV1*CONV2*9];
mat* wih;
mat* who;
float bc1[CONV1];
float bc2[CONV2];
mat* bih;
mat* bho;

//계산 tmp 변??
	//predict
	mat *input;
	mat *tmp_in_wih, *tmp_bias_hid, *tmp_l1;
	mat *tmp_hid_who, *tmp_l2;
////////

const char* matrix_data_file_name = "/data/local/tmp/matrixp.txt";
const char* pict_matrix_data_file_name = "data/local/tmp/pict_matrix.txt";

void cnn_init(int MODE){
	int i, j, k, l;
	FILE *f;

	if(MODE == CNN_LETTER){
		f = fopen(matrix_data_file_name,"r");
		IMAGE_SIZE = 28;
		DATAS = 784;
		DATASN = 28;
		CONV1_SIZE = 28;
		CONV2_SIZE = 14;
		INPUTS = CONV2*7*7;
		HIDDENS = 100;
		OUTPUTS = 10;
	}else if(MODE == CNN_PICT){
		f = fopen(pict_matrix_data_file_name,"r");
		IMAGE_SIZE = 36;
		DATAS = 1296;
		DATASN = 36;
		CONV1_SIZE = 36;
		CONV2_SIZE = 18;
		INPUTS = CONV2*9*9;
		HIDDENS = 120;
		OUTPUTS = 4;		
	}

	wih = MakeMat(INPUTS, HIDDENS);
	who = MakeMat(HIDDENS,OUTPUTS);
	bih = MakeMat(1, HIDDENS);
	bho = MakeMat(1, OUTPUTS);
	//calculate tmp
	input = MakeMat(1,INPUTS);
	tmp_in_wih = MakeMat(1,HIDDENS); tmp_bias_hid = MakeMat(1,HIDDENS); tmp_l1 = MakeMat(1,HIDDENS);
	tmp_hid_who = MakeMat(1,OUTPUTS); tmp_l2 = MakeMat(1,OUTPUTS);

	if(f){ //matrix data read
		//printf("\nMatrix file detect\n");
		for(i=0;i<INPUTS;i++){
			for(j=0;j<HIDDENS;j++){
				fscanf(f,"%e ",&wih->ma[i*wih->m+j]);
			}
		}
		for(i=0;i<HIDDENS;i++){
			for(j=0;j<OUTPUTS;j++){
				fscanf(f,"%e ",&who->ma[i*who->m+j]);
			}
		}
		for(k=0;k<CONV1;k++){
			for(i=0;i<3;i++){
				for(j=0;j<3;j++){
					fscanf(f,"%e ",&wdc1[k*9 + i*3 + j]);
				}
			}
		}
		for(l=0;l<CONV1;l++){
			for(k=0;k<CONV2;k++){
				for(i=0;i<3;i++){
					for(j=0;j<3;j++){
						fscanf(f,"%e ",&wc1c2[l*9*CONV2 + k*9 + i*3 + j]);
					}
				}
			}
		}

		for(i=0;i<CONV1;i++){ //bias initialation
			fscanf(f,"%e ",&bc1[i]);
		}
		for(i=0;i<CONV2;i++){
			fscanf(f,"%e ",&bc2[i]);
		}
		for(i=0;i<HIDDENS;i++){
			fscanf(f,"%e ",&bih->ma[i]);
		}
		for(i=0;i<OUTPUTS;i++){
			fscanf(f,"%e ",&bho->ma[i]);
		}
		fclose(f);
	}
}
void cnn_finite(){
	int i;
	close(&wih);
	close(&who);
	close(&input);
	close(&tmp_in_wih);	close(&tmp_bias_hid); close(&tmp_l1);
	close(&tmp_hid_who); close(&tmp_l2);
}

float data[36*36];
float conv1[36*36*CONV1];
float pool1[18*18*CONV1];
float conv2[18*18*CONV2];

int predict(cl_context& context, cl_command_queue& queue, cl_kernel& kernel_c, cl_kernel& kernel_d, unsigned char* in){
	int i, j, k, t, size = sqrt(DATAS), imgsize = DATASN*DATASN, imgsize2 = CONV2_SIZE*CONV2_SIZE;
	float max;
	for(i=0;i<size;i++){
		for(j=0;j<size;j++){
			data[i*DATASN+j] = normalize(in[(i*size+j)]); //index 0=indices
		}
	}

	//conv1
	arr_convolution2D(context, queue, kernel_c,
					 conv1, size, size, CONV1, data, size, size, 1, wdc1, 3, 3,1*CONV1, 1, 1, bc1);
	array_trans(conv1, imgsize*CONV1, ReLU);
	//max pooling 1
	for(i=0;i<CONV1;i++){
		for(j=0;j<CONV2_SIZE;j++){
			for(k=0;k<CONV2_SIZE;k++){
				max = conv1[i*imgsize+((j*2)*DATASN + (k*2))];
				if(max<conv1[i*imgsize+((j*2+1)*DATASN + (k*2))]){ max=conv1[i*imgsize+((j*2+1)*DATASN + (k*2))];} 
				if(max<conv1[i*imgsize+((j*2)*DATASN + (k*2+1))]){ max=conv1[i*imgsize+((j*2)*DATASN + (k*2+1))];}
				if(max<conv1[i*imgsize+((j*2+1)*DATASN + (k*2+1))]){ max=conv1[i*imgsize+((j*2+1)*DATASN + (k*2+1))];}
				pool1[(i*size*size/4)+(j*size/2+k)] = max;
			}
		}
	}
	//conv2
	arr_convolution2D(context, queue, kernel_c, 
					conv2, CONV2_SIZE, CONV2_SIZE, CONV2, pool1, CONV2_SIZE, CONV2_SIZE, CONV1, wc1c2, 3, 3, CONV1*CONV2, 1, 1, bc2);
	array_trans(conv2, imgsize2*CONV2, ReLU);
	for(i=0;i<CONV2;i++){
		for(j=0;j<CONV2_SIZE/2;j++){
			for(k=0;k<CONV2_SIZE/2;k++){
				max = conv2[i*imgsize2+((j*2)*CONV2_SIZE + (k*2))];
				if(max<conv2[i*imgsize2+((j*2+1)*CONV2_SIZE + (k*2))]){ max=conv2[i*imgsize2+((j*2+1)*CONV2_SIZE + (k*2))];} 
				if(max<conv2[i*imgsize2+((j*2)*CONV2_SIZE + (k*2+1))]){ max=conv2[i*imgsize2+((j*2)*CONV2_SIZE + (k*2+1))];}
				if(max<conv2[i*imgsize2+((j*2+1)*CONV2_SIZE + (k*2+1))]){ max=conv2[i*imgsize2+((j*2+1)*CONV2_SIZE + (k*2+1))];}
				input->ma[(i*CONV2_SIZE*CONV2_SIZE/4)+(j*CONV2_SIZE/2+k)] = max;
			}
		}
	}

	dotGPU(context, queue, kernel_d, tmp_in_wih, input, wih); msum(tmp_bias_hid, tmp_in_wih, bih); trans(tmp_l1, tmp_bias_hid, ReLU);
	dotGPU(context, queue, kernel_d, tmp_hid_who, tmp_l1, who); msum(tmp_l2, tmp_hid_who, bho);
    softmax(tmp_l2->ma, OUTPUTS);

    k = 0;
    for(i=0;i<OUTPUTS;i++){
    	if(tmp_l2->ma[k] < tmp_l2->ma[i]) k = i; //predict
    }
    return k;
}
