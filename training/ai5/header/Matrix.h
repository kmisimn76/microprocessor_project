

//=====Matrix
typedef struct Matrix {
	int n, m;
	float *ma;
}mat;

mat* MakeMat(int n, int m){
	int i;
	mat* ma = (mat*)malloc(sizeof(mat));
	ma->ma = (float*)malloc(sizeof(float)*n*m);
	ma->n = n;
	ma->m = m;
	return ma;
}
void InitZero(mat* m){
	int i, j;
	for(i=0;i<m->n*m->m; i++){
		m->ma[i] = 0;
	}
}

mat* dot(mat* result, mat* a, mat* b){
	int i, j, k;
	//mat* result = MakeMat(a->n, b->m);
	if(a->m!=b->n) return NULL;
	for(i = 0; i < a->n; i++){
		for(j=0;j<b->m;j++){
			result->ma[i*b->m+j] = 0;
			for(k=0;k<a->m;k++){
				result->ma[i*b->m+j] += a->ma[i*a->m+k]*b->ma[k*b->m+j];
			}
		}
	}
	return result;
}

mat* T(mat* result, mat* a){
	int i, j;
	//mat* result = MakeMat(a->m, a->n);
	for(i=0;i<a->m;i++){
		for(j=0;j<a->n;j++){
			result->ma[i*a->n+j] = a->ma[j*a->m+i];
		}
	}
	return result;
}

mat* mmul(mat* result, mat* a, mat* b){
	int i, j;
	//mat* result = MakeMat(a->n, a->m);
	if(a->n!=b->n || a->m!=b->m) return NULL;
	for(i=0;i<a->n;i++){
		for(j=0;j<a->m;j++){
			result->ma[i*a->m+j] = a->ma[i*a->m+j]*b->ma[i*b->m+j];
		}
	}
	return result;
}

mat* msum(mat* result, mat* a, mat* b){
	int i, j;
	//mat* result = MakeMat(a->n, a->m);
	if(a->n!=b->n || a->m!=b->m) return NULL;
	for(i=0;i<a->n;i++){
		for(j=0;j<a->m;j++){
			result->ma[i*a->m+j] = a->ma[i*a->m+j]+b->ma[i*b->m+j];
		}
	}
	return result;
}

mat * csum(mat* result, mat* a, float x, float inv){
    int i, j;
    //mat* result = MakeMat(a->n, a->m);
    for(i=0;i<a->n;i++){
    	for(j=0;j<a->m;j++){
    		result->ma[i*a->m+j] = (a->ma[i*a->m+j]+x)*inv;
    	}
    }
    return result;
}

mat * trans(mat* result, mat* a, float (*func)(float)){
	int i, j;
	//mat* result = MakeMat(a->n, a->m);
	for(i=0;i<a->n;i++){
		for(j=0;j<a->m;j++){
			result->ma[i*a->m+j] = func(a->ma[i*a->m+j]);
		}
	}
	return result;
}
//double beta1 = 0.4 , beta2 = 0.999, eps = 0.00000001; //momentum
double beta1 = 0.9, beta2 = 0.999, eps = 0.0000001; //ddam
void AdamGPU(float *d, float *dx, float *b, float *m, int t, float lr, int n_pix){

    size_t globalSize, localSize, grid;

    // Number of work items in each local work group
    localSize = 128; //TODO
    // Number of total work items - localSize must be devisor
	grid = (n_pix%localSize)? (n_pix/localSize)+1 : n_pix/localSize;
    globalSize = grid*localSize;

/*    for(int i=0;i<n_pix;i++){
    	printf("%f ", dx[i]);
    }
    printf("\n");*/
	launch_the_kernel_Adam(context, queue, kernel_adam, globalSize, localSize,
					  d, dx, b, m, t, lr, n_pix);
}

void Adam(float &d, float dx, float &b, float &m, int t, float lr){
	float bt, mt;
	b = beta1*b - (float)(1.0-beta1)*dx;
	m = beta2*m + (float)(1.0-beta2)*dx*dx;
	bt = b / (1.0 - pow(beta1,(double)t));
	mt = m / (1.0 - pow(beta2,(double)t));
	d -= lr * bt / (float)((double)(sqrt((double)mt))+eps); //learning rate 0.01
//	printf("%f %f\n", dx, lr * bt / (float)((double)(sqrt((double)mt))+eps));

//	d -= lr * dx; //learning rate = 0.0001

	//b = beta1 * b - lr*dx;
	//d += b;
	//if(abs(bt / (float)((double)(sqrt((double)mt))+eps))>3) printf("%f %f %f %f %f %f\n", dx, b, m, bt, mt, bt / (float)((double)(sqrt((double)mt))+eps));
}
void Adam_matrix(mat* d, mat *dx, mat* b, mat* m, int t, float lr){
	int i, j;
	int bt;
	int mt;
	AdamGPU(d->ma, dx->ma, b->ma, m->ma, t, lr, d->n*d->m);

	/*for(i=0;i<d->n;i++){
		for(j=0;j<d->m;j++){
			Adam(d->ma[i*d->m+j], dx->ma[i*d->m+j], b->ma[i*d->m+j], m->ma[i*d->m+j], t, lr);
			/*b->ma[i][j] = beta1*b->ma[i][j] - (float)(1.0-beta1)*dx->ma[i][j];
			m->ma[i][j] = beta2*m->ma[i][j] + (float)(1.0-beta2)*dx->ma[i][j]*dx->ma[i][j];
			bt = b->ma[i][j] / (1.0 - pow(beta1,(double)t));
			mt = m->ma[i][j]/ (1.0 - pow(beta2,(double)t));
			d->ma[i][j] += lr * bt / (float)((double)(sqrt((double)mt))+eps);*/

	//		d->ma[i][j] -= lr * dx->ma[i][j];

		//	b->ma[i][j] = beta1 * b->ma[i][j] - lr*dx->ma[i][j];
		//	d->ma[i][j] += b->ma[i][j];

			//printf("%f %f\n", dx->ma[i][j], b->ma[i][j]);

	/*	}
	}*/
}

int close(mat** m){
	int i;
	if(*m==NULL) return 0;
	free((*m)->ma);
	free(*m);
	*m=0;
	return 1;
}

//=====Matrix