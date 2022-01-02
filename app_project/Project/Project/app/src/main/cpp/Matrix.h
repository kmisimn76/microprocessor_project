

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

int close(mat** m){
	int i;
	if(*m==NULL) return 0;
	free((*m)->ma);
	free(*m);
	*m=0;
	return 1;
}

//=====Matrix