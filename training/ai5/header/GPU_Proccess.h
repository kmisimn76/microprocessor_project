#line 1
#define checkCL(expression) {                        \
	cl_int err = (expression);                       \
	if (err < 0 && err > -64) {                      \
		printf("Error on line %d. error code: %d\n", \
				__LINE__, err);                      \
		exit(0);                                     \
	}                                                \
}                                                    \

cl_platform_id cpPlatform;        // OpenCL platform
cl_device_id device_id;           // device ID
cl_context context;               // context
cl_command_queue queue;           // command queue
cl_program program;               // program
cl_kernel kernel;                 // kernel
cl_kernel kernel_b;
cl_kernel kernel_dot;
cl_kernel kernel_adam;
// Device input buffers
cl_mem d_src, d_filter;
// Device output buffer
cl_mem d_dest;
cl_mem d_bias;
cl_mem d_d, d_a, d_b;
cl_mem d_dx, d_m;

int opencl_infra_creation(cl_context&       context,
						  cl_platform_id&   cpPlatform,
						  cl_device_id&     device_id,
						  cl_command_queue& queue,
						  cl_program&       program,
						  cl_kernel&        kernel,
						  cl_kernel& 		kernel_b,
						  cl_kernel& 		kernel_dot,
						  cl_kernel& 		kernel_adam,
						  char*             kernel_file_buffer,
						  size_t            kernel_file_size,
						  const char*    kernel_name,
						  const char* 	 kernel_name_b,
						  const char*	 kernel_name_dot,
						  const char* 	 kernel_name_adam
						  ){
	cl_int err;

    // Bind to platform
    checkCL(clGetPlatformIDs(1, &cpPlatform, NULL));

    // Get ID for the device
    checkCL(clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_GPU, 1, &device_id, NULL));

    // Create a context
    context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);

    // Create a command queue
    queue = clCreateCommandQueue(context, device_id, 0, &err);
	checkCL(err);

    // Create the compute program from the source buffer
    program = clCreateProgramWithSource(context, 1,
                            (const char **) & kernel_file_buffer, &kernel_file_size, &err);
	checkCL(err);

    // Build the program executable
    checkCL(clBuildProgram(program, 0, NULL, NULL, NULL, NULL));

    // Create the compute kernel in the program we wish to run
    kernel = clCreateKernel(program, kernel_name, &err);
	checkCL(err);
    kernel_b = clCreateKernel(program, kernel_name_b, &err);
	checkCL(err);
    kernel_dot = clCreateKernel(program, kernel_name_dot, &err);
	checkCL(err);
    kernel_adam = clCreateKernel(program, kernel_name_adam, &err);
	checkCL(err);

	return 0;

}

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


int launch_the_kernel_b(cl_context&       context,
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
					  int 				dest_src_depth,
					  int 				dest_filter_depth,
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
    d_dest = clCreateBuffer(context, CL_MEM_WRITE_ONLY, dest_n*dest_m*dest_src_depth*dest_filter_depth*sizeof(float), NULL, &err);
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
    checkCL(clSetKernelArg(kernel,  3, sizeof(int), &dest_src_depth));
    checkCL(clSetKernelArg(kernel,  4, sizeof(int), &dest_filter_depth));
    checkCL(clSetKernelArg(kernel,  5, sizeof(cl_mem), &d_src));
    checkCL(clSetKernelArg(kernel,  6, sizeof(int), &src_n));
    checkCL(clSetKernelArg(kernel,  7, sizeof(int), &src_m));
    checkCL(clSetKernelArg(kernel,  8, sizeof(int), &src_depth));
    checkCL(clSetKernelArg(kernel,  9, sizeof(cl_mem), &d_filter));
    checkCL(clSetKernelArg(kernel, 10, sizeof(int), &filter_n));
    checkCL(clSetKernelArg(kernel, 11, sizeof(int), &filter_m));
    checkCL(clSetKernelArg(kernel, 12, sizeof(int), &filter_depth));
    checkCL(clSetKernelArg(kernel, 13, sizeof(float), &lr));
    checkCL(clSetKernelArg(kernel, 14, sizeof(int), &isbias));
    checkCL(clSetKernelArg(kernel, 15, sizeof(cl_mem), &d_bias));

    // Execute the kernel over the entire range of the data set
    checkCL(clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &globalSize, &localSize,
                                                              0, NULL, NULL));
     // Wait for the command queue to get serviced before reading back results
    checkCL(clFinish(queue));

    // Read the results from the device
    checkCL(clEnqueueReadBuffer(queue, d_dest, CL_TRUE, 0,
                                dest_n*dest_m*dest_src_depth*dest_filter_depth*sizeof(float), dest, 0, NULL, NULL ));

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


int launch_the_kernel_Adam(cl_context&       context,
					  cl_command_queue& queue,
					  cl_kernel&        kernel,
					  size_t            globalSize,
					  size_t            localSize,
					  float*			d,
					  float* 			dx,
					  float* 			b,
					  float* 			m,
					  int 				t,
					  float 			lr,
					  int 				n
					  ){

	cl_int err;

	// Create the input and output arrays in device memory for our calculation
    d_d = clCreateBuffer(context, CL_MEM_READ_WRITE, n*sizeof(float), NULL, &err);
	checkCL(err);
    d_dx = clCreateBuffer(context, CL_MEM_READ_ONLY, n*sizeof(float), NULL, &err);
	checkCL(err);
    d_b = clCreateBuffer(context, CL_MEM_READ_WRITE, n*sizeof(float), NULL, &err);
	checkCL(err);
    d_m = clCreateBuffer(context, CL_MEM_READ_WRITE, n*sizeof(float), NULL, &err);
	checkCL(err);

    // Write our data set into the input array in device memory
    checkCL(clEnqueueWriteBuffer(queue, d_d, CL_TRUE, 0,
                                   n*sizeof(float), d, 0, NULL, NULL));
    checkCL(clEnqueueWriteBuffer(queue, d_dx, CL_TRUE, 0,
                                   n*sizeof(float), dx, 0, NULL, NULL));
    checkCL(clEnqueueWriteBuffer(queue, d_b, CL_TRUE, 0,
                                   n*sizeof(float), b, 0, NULL, NULL));
    checkCL(clEnqueueWriteBuffer(queue, d_m, CL_TRUE, 0,
                                   n*sizeof(float), m, 0, NULL, NULL));

    // Set the arguments to our compute kernel
    checkCL(clSetKernelArg(kernel,  0, sizeof(cl_mem), &d_d));
    checkCL(clSetKernelArg(kernel,  1, sizeof(cl_mem), &d_dx));
    checkCL(clSetKernelArg(kernel,  2, sizeof(cl_mem), &d_b));
    checkCL(clSetKernelArg(kernel,  3, sizeof(cl_mem), &d_m));
    checkCL(clSetKernelArg(kernel,  4, sizeof(int), &t));
    checkCL(clSetKernelArg(kernel,  5, sizeof(float), &lr));

    // Execute the kernel over the entire range of the data set
    checkCL(clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &globalSize, &localSize,
                                                              0, NULL, NULL));
     // Wait for the command queue to get serviced before reading back results
    checkCL(clFinish(queue));

    // Read the results from the device
    checkCL(clEnqueueReadBuffer(queue, d_d, CL_TRUE, 0,
                                n*sizeof(float), d, 0, NULL, NULL ));
    checkCL(clEnqueueReadBuffer(queue, d_b, CL_TRUE, 0,
                                n*sizeof(float), b, 0, NULL, NULL ));
    checkCL(clEnqueueReadBuffer(queue, d_m, CL_TRUE, 0,
                                n*sizeof(float), m, 0, NULL, NULL ));


	checkCL(clReleaseMemObject(d_d));
	checkCL(clReleaseMemObject(d_dx));
	checkCL(clReleaseMemObject(d_b));
	checkCL(clReleaseMemObject(d_m));

	return 0;
}
