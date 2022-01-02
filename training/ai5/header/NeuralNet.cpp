
//=====activation func

int random(float percent){
	if(((float)(rand()%10000)/10000.0)<percent)
		return 1;
	else
		return 0;
}

float sigmoid(float x){
	return ((float)1)/((float)(1+exp(-x)));
}
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

void arr_convolution2D(float* dest_mat, int dest_n, int dest_m, int dest_depth,
						float* src_mat, int src_n, int src_m, int src_depth,
						float* filter_mat, int filter_n, int filter_m, int filter_depth,
						float lr, int isbias, float *bias){

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
void arr_convolution2D_backpropagation(float* dest_mat, int dest_n, int dest_m, int dest_src_depth, int dest_filter_depth,
						float* src_mat, int src_n, int src_m, int src_depth,
						float* filter_mat, int filter_n, int filter_m, int filter_depth,
						float lr, int isbias, float *bias){

    size_t globalSize, localSize, grid;
    if(!isbias)
    	bias = (float*)malloc(sizeof(float)*filter_depth);

    // Number of work items in each local work group
    localSize = 128; //TODO
	int n_pix = dest_n * dest_m *dest_src_depth * dest_filter_depth;

    // Number of total work items - localSize must be devisor
	grid = (n_pix%localSize)? (n_pix/localSize)+1 : n_pix/localSize;
    globalSize = grid*localSize;

	launch_the_kernel_b(context, queue, kernel_b, globalSize, localSize,
					  d_dest, d_src, d_filter, d_bias,
					  dest_mat, dest_n, dest_m, dest_src_depth, dest_filter_depth,
					  src_mat, src_n, src_m, src_depth,
					  filter_mat, filter_n,	filter_m, filter_depth,
					  lr, isbias, bias);

	if(!isbias)
		free(bias);

}

void dotGPU(mat* result, mat* a, mat* b){

    size_t globalSize, localSize, grid;

    int i, j;
	if(a->m!=b->n) return;

    // Number of work items in each local work group
    localSize = 128; //TODO
	int n_pix = a->n * b->m;
    // Number of total work items - localSize must be devisor
	grid = (n_pix%localSize)? (n_pix/localSize)+1 : n_pix/localSize;
    globalSize = grid*localSize;

	launch_the_kernel_dot(context, queue, kernel_dot, globalSize, localSize,
					  result->ma, a->ma, b->ma, a->n, b->m, a->m);
}


void array_trans(float* array, int n, float (*func)(float)){
	int i;
	for(i=0;i<n;i++){
		array[i] = func(array[i]);
	}
}

float normal(float mean, float dev){
	double r1, r2, z1, z2, result, dt, num;
	int i;
	result = 0.0;
	dt = 1.0/100.0;
	for(i=0;i<100;i++){
		r1 = (double)rand()/(double)RAND_MAX;
		r2 = (double)rand()/(double)RAND_MAX;
		z1=sqrt(-2*log(r1))*cos(2*PI*r2);
		z2=sqrt(-2*log(r1))*sin(2*PI*r2);
		num = (mean*dt)+(dev*z1*sqrt(dt));
		result += num;
	}
	if(!(-5<=result && result<=5)) result = 0.1;
	return (float)result;
}
float normalize(float x){
	//printf("%f %f\n",x,(x/255.0)*0.99+0.01);
	return (x/256.0)*0.99+0.01;
}

float training_data[DATAS+1];
float test_data[2001][DATAS+1];
int test_data_n = 0;
int training_start = 0;

float wdc1[CONV1*9];
float wc1c2[CONV1*CONV2*9];
float wdc1_tmp[CONV1*9];
float wc1c2_tmp[CONV1*CONV2*9];
mat* wih;
mat* who;
float bc1[CONV1];
float bc2[CONV2];
mat* bih;
mat* bho;

//Weight 변수
float bm_wdc1[2][CONV1*9] = {0};
float bm_wc1c2[2][CONV1*CONV2*9] = {0};
//float bm_wdc1_tmp[2][CONV1*9] = {0};
//float bm_wc1c2_tmp[2][CONV1*CONV2*9] = {0};
mat* b_wih, *m_wih;
mat* b_who, *m_who;
float bm_bc1[2][CONV1] = {0};
float bm_bc2[2][CONV2] = {0};
mat* b_bih, *m_bih;
mat* b_bho, *m_bho;
int process_t;
////

//계산 tmp 변수
	//predict
	mat *input;
	mat *tmp_in_wih, *tmp_bias_hid, *tmp_l1;
	mat *tmp_hid_who, *tmp_l2;


	mat *target;
	mat *tmp_pool2;
	mat *tmp_l2_e, *tmp_woh, *tmp_l1_e;
	mat *tmp_whi, *tmp_input_e;
	mat *tmp_l2_e_t, *tmp_l2_r_t, *tmp_l2_r;
	mat *tmp_l1_e_t, *tmp_l1_r_t, *tmp_l1_r;

////

const char* matrix_data_file_name = "data/matrix.txt";

void init(){
	int i, j, k, l;
	FILE *f = fopen(matrix_data_file_name,"r");
	wih = MakeMat(INPUTS, HIDDENS);
	who = MakeMat(HIDDENS,OUTPUTS);
	bih = MakeMat(1, HIDDENS);
	bho = MakeMat(1, OUTPUTS);
	//Adam
	b_wih = MakeMat(INPUTS, HIDDENS); InitZero(b_wih);
	b_who = MakeMat(HIDDENS,OUTPUTS); InitZero(b_who);
	b_bih = MakeMat(1, HIDDENS); InitZero(b_bih);
	b_bho = MakeMat(1, OUTPUTS); InitZero(b_bho);
	m_wih = MakeMat(INPUTS, HIDDENS); InitZero(m_wih);
	m_who = MakeMat(HIDDENS,OUTPUTS); InitZero(m_who);
	m_bih = MakeMat(1, HIDDENS); InitZero(m_wih);
	m_bho = MakeMat(1, OUTPUTS); InitZero(m_who);
	//calculate tmp
	input = MakeMat(1,INPUTS);
	tmp_in_wih = MakeMat(1,HIDDENS); tmp_bias_hid = MakeMat(1,HIDDENS); tmp_l1 = MakeMat(1,HIDDENS);
	tmp_hid_who = MakeMat(1,OUTPUTS); tmp_l2 = MakeMat(1,OUTPUTS);

	target = MakeMat(1,OUTPUTS);
	tmp_pool2 = MakeMat(1,INPUTS);
	tmp_l2_e = MakeMat(1,OUTPUTS); tmp_woh = MakeMat(OUTPUTS, HIDDENS);
	tmp_l1_e = MakeMat(1,HIDDENS); tmp_whi = MakeMat(HIDDENS, INPUTS);
	tmp_input_e = MakeMat(1, INPUTS);

	tmp_l2_e_t = MakeMat(OUTPUTS,1); tmp_l2_r_t = MakeMat(OUTPUTS,HIDDENS); tmp_l2_r = MakeMat(HIDDENS, OUTPUTS);
	tmp_l1_e_t = MakeMat(HIDDENS,1); tmp_l1_r_t = MakeMat(HIDDENS,INPUTS); tmp_l1_r = MakeMat(INPUTS, HIDDENS);

	printf("\n===init====\n");
	if(f==NULL){
		for(i=0;i<CONV1;i++){
			for(j=0;j<3;j++){
				for(k=0;k<3;k++){
					wdc1[i*9 + j*3 + k] = normal(0,1)/sqrt(1*DATAS/2);
				}
			}
		}
		for(l=0;l<CONV1;l++){
			for(i=0;i<CONV2;i++){
				for(j=0;j<3;j++){
					for(k=0;k<3;k++){
						wc1c2[l*9*CONV2 + i*9 + j*3 + k] = normal(0,1)/sqrt(CONV1*CONV2_SIZE*CONV2_SIZE/2);
					}
				}
			}
		}
		for(i=0;i<INPUTS;i++){
			for(j=0;j<HIDDENS;j++){
				wih->ma[i*wih->m+j] = normal(0,1)/sqrt(INPUTS/2);
			}
		}
		for(i=0;i<HIDDENS;i++){
			for(j=0;j<OUTPUTS;j++){
				who->ma[i*who->m+j] = normal(0,1)/sqrt(HIDDENS/2);
			}
		}
		for(i=0;i<CONV1;i++){ //bias initialation
			bc1[i] = 0;
		}
		for(i=0;i<CONV2;i++){
			bc2[i] = 0;
		}
		for(i=0;i<HIDDENS;i++){
			bih->ma[i] = 0;
		}
		for(i=0;i<OUTPUTS;i++){
			bho->ma[i] = 0;
		}
		process_t = 1;
		training_start = 0;
	}

	if(f){ //matrix data read
		printf("\nMatrix file detect\n");
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
		//Adam
		for(i=0;i<INPUTS;i++){
			for(j=0;j<HIDDENS;j++){
				fscanf(f,"%e %e ",&b_wih->ma[i*b_wih->m+j], &m_wih->ma[i*m_wih->m+j]);
			}
		}
		for(i=0;i<HIDDENS;i++){
			for(j=0;j<OUTPUTS;j++){
				fscanf(f,"%e %e ",&b_who->ma[i*b_who->m+j],&m_who->ma[i*m_who->m+j]);
			}
		}
		for(k=0;k<CONV1;k++){
			for(i=0;i<3;i++){
				for(j=0;j<3;j++){
					fscanf(f,"%e %e ",&bm_wdc1[0][k*9 + i*3 + j],&bm_wdc1[1][k*9 + i*3 + j]);
				}
			}
		}
		for(l=0;l<CONV1;l++){
			for(k=0;k<CONV2;k++){
				for(i=0;i<3;i++){
					for(j=0;j<3;j++){
						fscanf(f,"%e %e ",&bm_wc1c2[0][l*9*CONV2 + k*9 + i*3 + j],&bm_wc1c2[1][l*9*CONV2 + k*9 + i*3 + j]);
					}
				}
			}
		}
		for(i=0;i<CONV1;i++){ //bias initialation
			fscanf(f,"%e %e ",&bm_bc1[0][i], &bm_bc1[1][i]);
		}
		for(i=0;i<CONV2;i++){
			fscanf(f,"%e %e ",&bm_bc2[0][i], &bm_bc2[1][i]);
		}
		for(i=0;i<HIDDENS;i++){
			fscanf(f,"%e %e ",&b_bih->ma[i],&m_bih->ma[i]);
		}
		for(i=0;i<OUTPUTS;i++){
			fscanf(f,"%e %e ",&b_bho->ma[i],&m_bho->ma[i]);
		}
		fscanf(f, "%d\n", &training_start);
		if(0>training_start || training_start>60000) training_start=0;
		fscanf(f, "%d\n", &process_t);
		fclose(f);
	}
	printf("===finite init===\n");
}
void save(int p){
	int i, j, k, l;
	FILE* f = fopen(matrix_data_file_name,"w"); //matrix data output
	for(i=0;i<wih->n;i++){
		for(j=0;j<wih->m;j++){
			fprintf(f,"%e ",wih->ma[i*wih->m+j]);
		}
		fprintf(f,"\n");
	}
	fprintf(f,"\n");
	for(i=0;i<who->n;i++){
		for(j=0;j<who->m;j++){
			fprintf(f,"%e ",who->ma[i*who->m+j]);
		}
		fprintf(f,"\n");
	}
	fprintf(f,"\n");
	for(k=0;k<CONV1;k++){
		for(i=0;i<3;i++){
			for(j=0;j<3;j++){
				fprintf(f,"%e ",wdc1[k*9 + i*3 + j]);
			}
			fprintf(f,"\n");
		}
	}
	fprintf(f,"\n");
	for(l=0;l<CONV1;l++){
		for(k=0;k<CONV2;k++){
			for(i=0;i<3;i++){
				for(j=0;j<3;j++){
					fprintf(f,"%e ",wc1c2[l*9*CONV2 + k*9 + i*3 + j]);
				}
				fprintf(f,"\n");
			}
		}
	}
	for(i=0;i<CONV1;i++){ //bias initialation
		fprintf(f,"%e ",bc1[i]);
	}
	fprintf(f,"\n");
	for(i=0;i<CONV2;i++){
		fprintf(f,"%e ",bc2[i]);
	}
	fprintf(f,"\n");
	for(i=0;i<HIDDENS;i++){
		fprintf(f,"%e ",bih->ma[i]);
	}
	fprintf(f,"\n");
	for(i=0;i<OUTPUTS;i++){
		fprintf(f,"%e ",bho->ma[i]);
	}

	//Adam
	for(i=0;i<INPUTS;i++){
		for(j=0;j<HIDDENS;j++){
			fprintf(f,"%e %e ",b_wih->ma[i*b_wih->m+j], m_wih->ma[i*m_wih->m+j]);
		}
		fprintf(f,"\n");
	}
	for(i=0;i<HIDDENS;i++){
		for(j=0;j<OUTPUTS;j++){
			fprintf(f,"%e %e ",b_who->ma[i*b_who->m+j],m_who->ma[i*m_who->m+j]);
		}
		fprintf(f,"\n");
	}
	for(k=0;k<CONV1;k++){
		for(i=0;i<3;i++){
			for(j=0;j<3;j++){
				fprintf(f,"%e %e ",bm_wdc1[0][k*9 + i*3 + j],bm_wdc1[1][k*9 + i*3 + j]);
			}
		}
		fprintf(f,"\n");
	}

	for(l=0;l<CONV1;l++){
		for(k=0;k<CONV2;k++){
			for(i=0;i<3;i++){
				for(j=0;j<3;j++){
					fprintf(f,"%e %e ",bm_wc1c2[0][l*9*CONV2 + k*9 + i*3 + j],bm_wc1c2[1][l*9*CONV2 + k*9 + i*3 + j]);
				}
			}
			fprintf(f,"\n");
		}
	}
	for(i=0;i<CONV1;i++){ //bias initialation
		fprintf(f,"%e %e ",bm_bc1[0][i], bm_bc1[1][i]);
	}
	fprintf(f,"\n");
	for(i=0;i<CONV2;i++){
		fprintf(f,"%e %e ",bm_bc2[0][i], bm_bc2[1][i]);
	}
	fprintf(f,"\n");
	for(i=0;i<HIDDENS;i++){
		fprintf(f,"%e %e ",b_bih->ma[i],m_bih->ma[i]);
	}
	fprintf(f,"\n");
	for(i=0;i<OUTPUTS;i++){
		fprintf(f,"%e %e ",b_bho->ma[i],m_bho->ma[i]);
	}
	fprintf(f,"\n");
	fprintf(f,"\n");
	fprintf(f,"%d\n",p);
	fprintf(f,"%d\n",process_t);
	fclose(f);
}
void finite(){
	int i;
	save(0);
	close(&wih);
	close(&who);
	close(&input);
	close(&tmp_in_wih);	close(&tmp_bias_hid); close(&tmp_l1);
	close(&tmp_hid_who); close(&tmp_l2);
	close(&target);
	close(&tmp_pool2);
	close(&tmp_l2_e); close(&tmp_woh); close(&tmp_l1_e);
	close(&tmp_whi); close(&tmp_input_e);
	close(&tmp_l2_e_t); close(&tmp_l2_r_t); close(&tmp_l2_r);
	close(&tmp_l1_e_t); close(&tmp_l1_r_t); close(&tmp_l1_r);
}

float data[DATASN*DATASN];
float conv1[CONV1_SIZE*CONV1_SIZE*CONV1];
float pool1[CONV2_SIZE*CONV2_SIZE*CONV1];
float tmp_pool1[CONV2_SIZE*CONV2_SIZE*CONV1];
float conv2[CONV2_SIZE*CONV2_SIZE*CONV2];
float conv1_e[CONV1_SIZE*CONV1_SIZE*CONV1];
float conv2_e[CONV2_SIZE*CONV2_SIZE*CONV2];
float conv1_et[CONV2_SIZE*CONV2_SIZE*CONV1];
void predict(float* out, float* in){
	int i, j, k, t, size = sqrt(DATAS), imgsize = DATASN*DATASN, imgsize2 = CONV2_SIZE*CONV2_SIZE;
	float max;
	for(i=0;i<size;i++){
		for(j=0;j<size;j++){
			data[i*DATASN+j] = normalize(in[(i*size+j)+1]); //index 0=indices
		}
	}

	//conv1
	arr_convolution2D(conv1, size, size, CONV1, data, size, size, 1, wdc1, 3, 3,1*CONV1, 1, 1, bc1);
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
	arr_convolution2D(conv2, CONV2_SIZE, CONV2_SIZE, CONV2, pool1, CONV2_SIZE, CONV2_SIZE, CONV1, wc1c2, 3, 3, CONV1*CONV2, 1, 1, bc2);
	/*for(i=0;i<imgsize2*CONV2;i++){
		printf("%f ", conv2[i]);
	}
	printf("\n");*/
	array_trans(conv2, imgsize2*CONV2, ReLU);
	//max pooling2
	//input = MakeMat(1, INPUTS);
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

	//t1=dot(input,wih); t2=msum(t1, bih); l1=trans(t2,ReLU);    //layer_1 = (mdata.dot(self.wih)).trans(sigmoid)*****
	dot(tmp_in_wih, input, wih); msum(tmp_bias_hid, tmp_in_wih, bih); trans(tmp_l1, tmp_bias_hid, ReLU);
	/*for(i=0;i<HIDDENS;i++){
		printf("%f ", l1->ma[0][i]);
	}
	printf("\n");*/
    //t1=dot(l1,who); output=msum(t1, bho);    //output = (layer_1.dot(self.who)).trans(sigmoid)
	dot(tmp_hid_who, tmp_l1, who); msum(tmp_l2, tmp_hid_who, bho);
    softmax(tmp_l2->ma, OUTPUTS);

    for(i=0;i<OUTPUTS;i++){
    	out[i]=tmp_l2->ma[i];
    }
}

void train(float* in, float lr, float dropout, float dropout2){
	int i, j, k, l, t, row, col, size = sqrt(DATAS), imgsize = DATASN*DATASN, imgsize2 = CONV2_SIZE*CONV2_SIZE;
	float s, max;
	struct timeval start, end, timer;
	//data init
	for(i=0;i<size;i++){
		for(j=0;j<size;j++){
			data[i*DATASN+j] = normalize(in[(i*size+j)+1]); //index 0=indices
		}
	}
	for(i=0;i<OUTPUTS;i++){
		target->ma[i] = (i==in[0])?0.99:0.01;
	}

	//forward
	//conv1
	arr_convolution2D(conv1, size, size, CONV1, data, size, size, 1, wdc1, 3, 3, 1*CONV1, 1, 1, bc1);
	array_trans(conv1, imgsize*CONV1, ReLU);
	/*for(i=0;i<CONV1;i++){
		for(j=0;j<imgsize;j++){
			if(random((float)(1.0-dropout))){
				conv1[i*imgsize+j] = 0;	//dropout
			}
		}
	}*/

	//max pooling 1
	for(i=0;i<CONV1;i++){
		for(j=0;j<CONV2_SIZE;j++){
			for(k=0;k<CONV2_SIZE;k++){
				t = 0;
				max = conv1[i*imgsize+((j*2)*DATASN + (k*2))];
				if(max<conv1[i*imgsize+((j*2+1)*DATASN + (k*2))]){ max=conv1[i*imgsize+((j*2+1)*DATASN + (k*2))]; t=1;} 
				if(max<conv1[i*imgsize+((j*2)*DATASN + (k*2+1))]){ max=conv1[i*imgsize+((j*2)*DATASN + (k*2+1))]; t=2;}
				if(max<conv1[i*imgsize+((j*2+1)*DATASN + (k*2+1))]){ max=conv1[i*imgsize+((j*2+1)*DATASN + (k*2+1))]; t=3;}
				pool1[(i*size*size/4)+(j*size/2+k)] = max;
				tmp_pool1[(i*size*size/4)+(j*size/2+k)] = t;
			}
		}
	}

	//conv2
	arr_convolution2D(conv2, CONV2_SIZE, CONV2_SIZE, CONV2, pool1, CONV2_SIZE, CONV2_SIZE, CONV1, wc1c2, 3, 3, CONV1*CONV2, 1, 1, bc2);
	array_trans(conv2, imgsize2*CONV2, ReLU);
	/*for(i=0;i<CONV2;i++){
		for(j=0;j<imgsize2;j++){
			if(random((float)(1.0-dropout))){
				conv2[i*imgsize2+j] = 0;	//dropout
			}
		}
	}*/
	//max pooling2
	//input = MakeMat(1, INPUTS);
	//tmp = MakeMat(1, INPUTS);
	for(i=0;i<CONV2;i++){
		for(j=0;j<CONV2_SIZE/2;j++){
			for(k=0;k<CONV2_SIZE/2;k++){
				t = 0;
				max = conv2[i*imgsize2+((j*2)*CONV2_SIZE + (k*2))];
				if(max<conv2[i*imgsize2+((j*2+1)*CONV2_SIZE + (k*2))]){ max=conv2[i*imgsize2+((j*2+1)*CONV2_SIZE + (k*2))]; t=1;} 
				if(max<conv2[i*imgsize2+((j*2)*CONV2_SIZE + (k*2+1))]){ max=conv2[i*imgsize2+((j*2)*CONV2_SIZE + (k*2+1))]; t=2;}
				if(max<conv2[i*imgsize2+((j*2+1)*CONV2_SIZE + (k*2+1))]){ max=conv2[i*imgsize2+((j*2+1)*CONV2_SIZE + (k*2+1))]; t=3;}
				input->ma[(i*CONV2_SIZE*CONV2_SIZE/4)+(j*CONV2_SIZE/2+k)] = max;
				tmp_pool2->ma[(i*CONV2_SIZE*CONV2_SIZE/4)+(j*CONV2_SIZE/2+k)] = t;
			}
		}
	}
	for(i=0;i<INPUTS;i++){
		if(random((float)(1.0-dropout))){
			input->ma[i] = 0;	//dropout
		}
	}

	dotGPU(tmp_in_wih, input, wih); msum(tmp_bias_hid, tmp_in_wih, bih); trans(tmp_l1, tmp_bias_hid, ReLU);    //layer_1 = (mdata.dot(self.wih)).trans(sigmoid)*****
	for(i=0;i<HIDDENS;i++){
		if(random((float)(1.0-dropout2))){
			tmp_l1->ma[i] = 0;	//dropout
		}
	}
    dotGPU(tmp_hid_who, tmp_l1, who); msum(tmp_l2, tmp_hid_who, bho);    //output = (layer_1.dot(self.who)).trans(sigmoid)
    softmax(tmp_l2->ma, OUTPUTS);

    /*for(i=0;i<OUTPUTS;i++){
    	printf("%f ", tmp_l2->ma[i]);
    }
    printf("\n");*/

    //Backpropagration==============================
	csum(tmp_l2_e, tmp_l2,0,-1);	//softmax
    tmp_l2_e->ma[(int)(in[0])] += 1.0;
 
    T(tmp_woh, who); dotGPU(tmp_l1_e,tmp_l2_e,tmp_woh); //t3=csum(l1,-1,-1); t4=mmul(l1,t3);
    for(i=0;i<HIDDENS;i++){
    	tmp_l1_e->ma[i] = (tmp_l1->ma[i]>0)? tmp_l1_e->ma[i] : 0.01*tmp_l1_e->ma[i]; //Relu, Dropout
    }

    T(tmp_whi, wih);
    dotGPU(tmp_input_e, tmp_l1_e,tmp_whi);

    for(i=0;i<INPUTS;i++){
    	tmp_input_e->ma[i] = (input->ma[i]>0)? tmp_input_e->ma[i] : 0.01*tmp_input_e->ma[i]; //Dropout
    }

	for(i=0;i<CONV2;i++){
		for(j=0;j<CONV2_SIZE;j++){
			for(k=0;k<CONV2_SIZE;k++){
				t = tmp_pool2->ma[(i*CONV2_SIZE*CONV2_SIZE/4)+((j>>1)*CONV2_SIZE/2+(k>>1))];
				if(t==((j%2)*2+(k%2)) && /*input->ma[0][(i*CONV2_SIZE*CONV2_SIZE/4)+((j>>1)*CONV2_SIZE/2+(k>>1))]>0 && */
						conv2[i*imgsize2+(j*CONV2_SIZE + k)]>0) { //max, relu, dropout
					conv2_e[i*imgsize2+(j*CONV2_SIZE + k)] = tmp_input_e->ma[(i*CONV2_SIZE*CONV2_SIZE/4)+((j>>1)*CONV2_SIZE/2+(k>>1))];
				}else if(t!=((j%2)*2+(k%2)) || conv2[i*imgsize2+(j*CONV2_SIZE + k)]==0){ //not max
					conv2_e[i*imgsize2+(j*CONV2_SIZE + k)] = 0;
				}else{
					conv2_e[i*imgsize2+(j*CONV2_SIZE + k)] = 0.01*tmp_input_e->ma[(i*CONV2_SIZE*CONV2_SIZE/4)+((j>>1)*CONV2_SIZE/2+(k>>1))];
				}
				/*if(t==((j%2)*2+(k%2)) && conv2[i*imgsize2+(j*CONV2_SIZE + k)]>0) { //max, relu
					conv2_e[i*imgsize2+(j*CONV2_SIZE + k)] = tmp_input_e->ma[(i*CONV2_SIZE*CONV2_SIZE/4)+((j>>1)*CONV2_SIZE/2+(k>>1))];
				}
				else{
					conv2_e[i*imgsize2+(j*CONV2_SIZE + k)] = 0;
				}*/
			}
		}
	}

	arr_convolution2D(conv1_et, CONV2_SIZE, CONV2_SIZE, CONV1, conv2_e, CONV2_SIZE, CONV2_SIZE, CONV2, wc1c2, 3, 3, CONV1*CONV2, 0, 0, NULL);

	for(i=0;i<CONV1;i++){
		for(j=0;j<CONV1_SIZE;j++){
			for(k=0;k<CONV1_SIZE;k++){
				t = tmp_pool1[(i*CONV1_SIZE*CONV1_SIZE/4)+((j>>1)*CONV1_SIZE/2+(k>>1))];
				if(t==((j%2)*2+(k%2)) && /*conv2_e[(i*CONV1_SIZE*CONV1_SIZE/4)+((j>>1)*CONV1_SIZE/2+(k>>1))]>0 && */
						conv1[i*imgsize+(j*CONV1_SIZE + k)]>0) { //max, relu, dropout
					conv1_e[i*imgsize+(j*CONV1_SIZE + k)] = conv1_et[(i*CONV1_SIZE*CONV1_SIZE/4)+((j>>1)*CONV1_SIZE/2+(k>>1))];
				}else if(t!=((j%2)*2+(k%2)) || conv1[i*imgsize+(j*CONV1_SIZE + k)]==0){ //not max
					conv1_e[i*imgsize+(j*CONV1_SIZE + k)] = 0;
				}else{
					conv1_e[i*imgsize+(j*CONV1_SIZE + k)] = 0.01*conv1_et[(i*CONV1_SIZE*CONV1_SIZE/4)+((j>>1)*CONV1_SIZE/2+(k>>1))];
				}
				/*if(t==((j%2)*2+(k%2)) && conv1[i*imgsize+(j*CONV1_SIZE + k)]>0) { //max, relu
					conv1_e[i*imgsize+(j*CONV1_SIZE + k)] = conv1_et[(i*CONV1_SIZE*CONV1_SIZE/4)+((j>>1)*CONV1_SIZE/2+(k>>1))];
				}else{
					conv1_e[i*imgsize+(j*CONV1_SIZE + k)] = 0;
				}*/
			}
		}
	}


	//matrix propogation
    T(tmp_l2_e_t, tmp_l2_e); dotGPU(tmp_l2_r_t,tmp_l2_e_t,tmp_l1); T(tmp_l2_r,tmp_l2_r_t);
    /*for(i=0;i<HIDDENS;i++){
    	for(j=0;j<OUTPUTS;j++){
    		printf("%f ", tmp_l2_r->ma[i*OUTPUTS+j]);
    	}
    }
    printf("\n");*/
    /*for(i=0;i<HIDDENS;i++){
    	printf("%f ", tmp_l1->ma[i]);
    }
    printf("\n");*/
    Adam_matrix(who, tmp_l2_r, b_who, m_who, process_t, lr);
    T(tmp_l1_e_t, tmp_l1_e); dotGPU(tmp_l1_r_t,tmp_l1_e_t,input); T(tmp_l1_r, tmp_l1_r_t);
    Adam_matrix(wih, tmp_l1_r, b_wih, m_wih, process_t, lr);
	
	/*for(i=0;i<CONV2;i++){ //Dwc1c2구하기 위해 maxpooling 풀기
		for(j=0;j<CONV1_SIZE;j++){
			for(k=0;k<CONV1_SIZE;k++){
				conv2_et[i*imgsize+j*size+k]=conv2_e[i*(imgsize2)+(j/2)*CONV2_SIZE+(k/2)];
			}
		}
	}*/
	arr_convolution2D_backpropagation(wc1c2_tmp, 3, 3, CONV1, CONV2,
									  pool1, CONV2_SIZE, CONV2_SIZE, CONV1,
									  conv2_e, CONV2_SIZE, CONV2_SIZE, CONV2, 1, 0, NULL);
	AdamGPU(wc1c2, wc1c2_tmp, bm_wc1c2[0], bm_wc1c2[1], process_t, lr, CONV1*CONV2*9);

	arr_convolution2D_backpropagation(wdc1_tmp, 3, 3, 1, CONV1, data, 28, 28, 1, conv1_e, size, size, CONV1, 1, 0, NULL);
	for(i=0;i<CONV1*9;i++){
		Adam(wdc1[i], (wdc1_tmp[i]/*/784*/), bm_wdc1[0][i], bm_wdc1[1][i], process_t, lr);
	}

	//bias propogation
    Adam_matrix(bho, tmp_l2_e, b_bho, m_bho, process_t, lr);
    Adam_matrix(bih, tmp_l1_e, b_bih, m_bih, process_t, lr);
	for(i=0;i<CONV2;i++){
		s = 0;
		for(j=0;j<imgsize2;j++){
			s+=conv2_e[i*imgsize2+j];
		}
		Adam(bc2[i], s/*/3136*/, bm_bc2[0][i], bm_bc2[1][i], process_t, lr);
	}
	for(i=0;i<CONV1;i++){
		s = 0;
		for(j=0;j<imgsize;j++){
			s+=conv1_e[i*imgsize+j];
		}
		Adam(bc1[i], s/*/6272*/, bm_bc1[0][i], bm_bc1[1][i], process_t, lr);
	}
	process_t++;
}
