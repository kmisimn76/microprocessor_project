/**
** AI5. Convolution+FC Layer CNN
**	Update: Matrix GPU
**/
#define CL_FILE "header/ml.cl"

#define	timersub(tvp, uvp, vvp)						\
	do {								\
		(vvp)->tv_sec = (tvp)->tv_sec - (uvp)->tv_sec;		\
		(vvp)->tv_usec = (tvp)->tv_usec - (uvp)->tv_usec;	\
		if ((vvp)->tv_usec < 0) {				\
			(vvp)->tv_sec--;				\
			(vvp)->tv_usec += 1000000;			\
		}							\
	} while (0)

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>
#include <CL/opencl.h>
#include "header/ImageProcessing.h"
#include "header/Const.h"
#include "header/GPU_Proccess.h"
#include "header/Matrix.h"
#include "header/NeuralNet.cpp"


void print_acc(){
	int matched;
	int i,j,max;
	float out[OUTPUTS];
	float loss;
	int predicated;

	matched=0;
	loss = 0.0;
	for(i=0;i<test_data_n;i++){
		predict(out, test_data[i]);
		predicated=0;
		for(j=0;j<OUTPUTS;j++){
			if(out[predicated]<out[j])
				predicated = j;
		}
		/*if(i%((int)(test_data_n/10))==0){
			for(j=0;j<OUTPUTS;j++)
				printf("%f ",out[j]);
			printf("\n");
		}
		if(i%463==0){
		}*/
		if((int)(test_data[i][0])==predicated){
			matched++;
		}
		loss-=log(out[(int)(test_data[i][0])]);
	}
	printf("accuracy: %f\n",((float)matched/(float)test_data_n));
	printf("loss: %f\n", loss/(float)(test_data_n));
}
float p_data[DATAS];
void all_test(FILE *LOG){
	int matched;
	int i,j,max;
	float out[OUTPUTS];
	double loss;
	int predicated;
	int n = 10000;
	FILE *tf = fopen("data/mnist_test.csv","r");
	printf("alltest:\n");

	matched=0;
	loss = 0.0;
	for(i=0;i<n;i++){
		for(j=0;j<DATAS+1;j++){
			fscanf(tf,"%f, ",&p_data[j]);
		}
		predict(out, p_data);
		predicated=0;
		for(j=0;j<OUTPUTS;j++){
			if(out[predicated]<out[j])
				predicated = j;
		}
		if((int)(p_data[0])==predicated){
			matched++;
		}
		loss-=log(out[(int)(p_data[0])]);
	}
	printf("accuracy: %f\n",((float)matched/(float)n));
	printf("loss: %f\n", loss/(float)(n));
	fclose(tf);
	if(LOG!=NULL){
		fprintf(LOG, "accuracy: %f ", ((float)matched/(float)n));
		fprintf(LOG, "loss: %f\n", loss/(float)(n));
	}
}

//=====DeepNeuralNet

	BITMAPINFOHEADER bmpHeader;
	unsigned char *image;
	char image_name[] = "data/test.bmp";
	float user_test_data[DATAS+1];
void user_test(){
	int i, j, k;
	float out[OUTPUTS];
	//image reading
	printf("====UserTest===");
	image = read_bmp(image_name,&bmpHeader);
	printf("%d %d\n",bmpHeader.biSizeImage, bmpHeader.biHeight);
	if(image==NULL) printf("asdd\n");
	k=1;
	user_test_data[0] = -1; //<------make your self
	for(i=0;i<28;i++){
		for(j=0;j<28;j++){
			printf("%X ",image[((27-i)*28+j)*3]);
			user_test_data[k++] = image[((27-i)*28+j)*3];
		}
			printf("\n");
	}
	k = 0;
	predict(out, user_test_data);
	for(j=0;j<OUTPUTS;j++){
		printf("%f ",out[j]);
		if(out[k]<out[j]) k = j;
	}
	printf("\npredict: %d, original: %d\n", k, (int)(user_test_data[0]));
	printf("===UserTest End====");
	getchar();
}

FILE *LOG_FILE;
FILE *file_handle;
char *file_buffer, *file_log;
size_t file_size, log_size;
int main() {
	int ei;
	int epoch;
	int i,j,k;
	FILE* f;
	float out[OUTPUTS];

/* 상수정의 */

	epoch = 13;
	test_data_n = 2000;

/* ---- */

	srand(time(NULL));
	printf("\n\n******start******\n\n");
	printf("===read kernel===\n");
	file_handle=fopen(CL_FILE, "r");
	if(file_handle==NULL)
	{
		printf("Couldn't find the file");
		exit(1);
	}

	fseek(file_handle, 0, SEEK_END);
	file_size = ftell(file_handle);
	rewind(file_handle);
	file_buffer = (char*)malloc(file_size+1);
	fread(file_buffer, sizeof(char), file_size, file_handle);
	file_buffer[file_size] = '\0';
	fclose(file_handle);
	printf("==finite read kernel file==\n");
	//printf("%s\n%d\n",file_buffer,file_size);
	opencl_infra_creation(context, cpPlatform, device_id, queue, program,
						  kernel, kernel_b, kernel_dot, kernel_adam,
						  file_buffer, file_size,
						  "convolution", "convolution_backpropagation","matdotGPU","AdamGPU");
	printf("\n===finite opencl infra creation===\n");
	//conv test
	float img[7*7*2] = { 2, 3, 4, 2, 3, 4, 3,
						4, 5, 6, 4, 3, 5, 3,
						1, 3, 5, 3, 7, 2, 4,
						4, 7, 4, 5, 7, 3, 2,
						9, 5, 7, 5, 3, 7, 1,
						3, 6, 8, 3, 5, 2, 8,
						3, 5, 7, 4, 6, 4, 4,

						2, 3, 4, 2, 3, 4, 3,
						4, 5, 6, 4, 3, 5, 3,
						1, 3, 5, 3, 7, 2, 4,
						4, 7, 4, 5, 7, 3, 2,
						9, 5, 7, 5, 3, 7, 1,
						3, 6, 8, 3, 5, 2, 8,
						3, 5, 7, 4, 6, 4, 4,};
	float des[7*7*2];
	float filter[5*5*2*2] = {0.3, 0.3, 0.3,0.3, 0.3,
						 0.3,0.3, 0.3, 0.3, 0.3,
						 0.3,0.3, 0.3, 0.3, 0.3,
						 0.3,0.3, 0.3, 0.3, 0.3,
						 0.3,0.3, 0.3, 0.3, 0.3,

						 0.1,0.1, 0.1, 0.1, 0.1,
						 0.1,0.1, 0.1, 0.1, 0.1,
						 0.1,0.1, 0.1, 0.1, 0.1,
						 0.1,0.1, 0.1, 0.1, 0.1,
						 0.1,0.1, 0.1, 0.1, 0.1,

						0.5,0.5, 0.5, 0.5, 0.5,
						 0.5,0.5, 0.5, 0.5, 0.5,
						 0.5,0.5, 0.5, 0.5, 0.5,
						 0.5,0.5, 0.5, 0.5, 0.5,
						 0.5,0.5, 0.5, 0.5, 0.5,

						0.4,0.4, 0.4, 0.4, 0.4,
						 0.4,0.4, 0.4, 0.4, 0.4,
						 0.4,0.4, 0.4, 0.4, 0.4,
						 0.4,0.4, 0.4, 0.4, 0.4,
						 0.4,0.4, 0.4, 0.4, 0.4};
	float a[2] = {0,0};

	arr_convolution2D(des, 7, 7, 2, img, 7, 7, 2, filter, 5, 5, 4, 1, 1, a);
	for(k=0;k<2;k++){
		for(i=0;i<7;i++){
			for(j=0;j<7;j++){
				printf("%f ", des[k*49+i*7+j]);
			}
			printf("\n");
		}
		printf("\n\n");
	}
	float d[6] = {0.2, 0.46, 0.27, 0.53, 0.82, 0.24};
	float dx[6] = {0.01, 0.07, 0.03, 0.003, 0.04, 0.01};
	float b[6] = {0.5, 0.76, 0.27, 0.9, 0.032, 0.033};
	float m[6] = {0.3, 0.26, 0.3, 0.136, 0.064, 0.163};
	for(i=0;i<6;i++){
		Adam(d[i], dx[i], b[i], m[i], 16, 0.1);
		printf("%f %f %f ", d[i], b[i], m[i]);
	}
	printf("\n");
	float dd[6] = {0.2, 0.46, 0.27, 0.53, 0.82, 0.24};
	float ddx[6] = {0.01, 0.07, 0.03, 0.003, 0.04, 0.01};
	float bb[6] = {0.5, 0.76, 0.27, 0.9, 0.032, 0.033};
	float mm[6] = {0.3, 0.26, 0.3, 0.136, 0.064, 0.163};
	AdamGPU(dd, ddx, bb, mm, 16, 0.1, 6);
	for(i=0;i<6;i++){
		printf("%f %f %f ", dd[i], bb[i], mm[i]);
	}
	printf("\n");

	//mnist test loading
	printf("===mnist test loading===\n");
	f = fopen("data/mnist_test.csv","r");

	printf("%f %f %f \n", normal(0,1), normal(0,1), normal(0,1));
	for(j=0;j<test_data_n;j++){
		for(i=0;i<DATAS+1;i++){
			fscanf(f,"%f, ",&test_data[j][i]);
		//	if(i==0) printf("%f", test_data[j][i]);
		}
	}
	fclose(f);
	printf("===finite mnist test loading===\n");

	printf("\n----training-----\n");
	init();
	printf("%d\n", process_t);
//user_test();
    printf("\n");
	printf("WIH:%f WHO:%f\n", wih->ma[10*wih->m+10],who->ma[10*who->m+5]);
	/*for(i=0;i<training_start;i++){
		if(i%((int)(training_start/5))==0) printf("training loading\n");
		for(j=0;j<DATAS+1;j++){
			fscanf(f,"%f, ",&training_data[j]);
		}
	}*/
	printf("\n");
	LOG_FILE = fopen("data/log.txt", "w");
	fprintf(LOG_FILE, "Init: ");
	//all_test(LOG_FILE);
	fclose(LOG_FILE);
	for(ei = 0; ei < epoch; ei++){
		f = fopen("data/mnist_train.csv","r");		
		printf("====================Epoch %d ===================\n", ei+1);
		for(i = 0; i < 60000; i++){
			//printf("\n%d\n",i);
			for(j=0;j<DATAS+1;j++){
				fscanf(f,"%f, ",&training_data[j]);
			}
			if(i%20000==0){
				printf("=====check=====\n");
				print_acc();
				save(i);
				printf("=====check end=====\n");
			}
			if(i%10000==0){
				predict(out, training_data);
				k = 0;
				for(j=0;j<OUTPUTS;j++){
					if(out[j]>out[k]) k = j;
					printf("%0.6f ",out[j]);
				}
				printf(" >>> %d\n", k);
				for(j=0;j<OUTPUTS;j++){
					printf("%0.6f ",(training_data[0]==j)?0.99:0.00);
				}
				printf(" >>> %d", (int)(training_data[0]));
				if(k==(int)(training_data[0])) printf(" OOOOO\n");
				else printf(" XXXXXXXXXXXXX\n");
				printf("--\n");
			}
			//getchar();
			train(training_data, 0.001, 0.7, 0.5);
			/*printf("--\n");
			predict(out, training_data);
			if(i%1==0){
				for(j=0;j<OUTPUTS;j++)
					printf("%f ",out[j]);
				printf("\n");
			}*/
	    	//printf("======================================================\n");
	    	//getchar();
		}
		printf("====================Epoch Train END ==========================\n");
		printf("\n");
		printf("%d\n", i);
		printf("=====AllTest=====\n\n");
		LOG_FILE = fopen("data/log.txt", "a");
		if(LOG_FILE!=NULL){
			fprintf(LOG_FILE,"Epoch %d: ",ei+1);
		}
		all_test(LOG_FILE);
		save(i);
		printf("\n=====AllTest end=====\n\n");
		fclose(LOG_FILE);
		printf("====================Epoch END ================================\n");
		fclose(f);
	}
	finite();


	return 0;
}
