#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "net.h"


#if 0
int getMS(double *m,double *s)
{
	FILE *fp;
	int i = 0,j=0;
	
	fp = fopen("mean.txt","r");
	if (fp == NULL) {
		printf("Open File mean.txt error\n");
	}
	while(fscanf(fp,"%lf\n",&m[i]) != -1){
		printf("mean[%d]: %lf\n",i, m[i]);
		i++;
	}
	fclose(fp);

	fp = fopen("std.txt","r");
	while(fscanf(fp,"%lf\n",&s[j]) != -1){
		printf("std[%d]: %lf\n", j, s[j]);
		j++;
	}
	fclose(fp);
	return 0;
}
#endif

double *normalize(double* data,double* mean, double* std,int size)
{
	double* new_data;
	int i;
	new_data = (double*)malloc(size * sizeof(double));

	for(i=0;i<size;i++)
	{
		new_data[i] = (data[i]-mean[i])/std[i];
		printf("new_data[%d]: %lf = (%lf - %lf)/%lf;\n", i, new_data[i], data[i], mean[i], std[i]);
	}
	return new_data;
}

double activation(double x)
{
       return((2.0/(1.0+exp(-2.0*x)))-1);
}

#if 0
void comout_in2hidden(NET *net, double *p1,double *p2,double* c,int n1,int n2)
{   

	int i, j;
	double sum;
	p1[0] = 1.0;

	for (i=1; i<=n2; i++) {
		sum = 0.0;
		printf("hidden[%d]:\n", i);
		for (j=0; j<=n1; j++) {
			printf("%f * %f(%d, %d) = %f\n", p1[j], c[IN2HIDDEN_INDEX(j, i)],
					j, i, p1[j] * c[IN2HIDDEN_INDEX(j, i)]);
			sum += p1[j] * c[IN2HIDDEN_INDEX(j, i)];
		}
		p2[i] = activation(sum);
		printf("hidden[%d] is %f\n", i, p2[i]);
	}
}
#endif

void comout(int index)
{
	int i, j;
	double sum = 0.0;
	double *inunits = units_ptr[index];
	double *outunits = units_ptr[index+1];
	double *weight = weight_ptr[index];
	double *biasweight = bias_ptr[index];

	int indim = dim[index];
	int outdim = dim[index+1];

	for(i=0;i<outdim;i++){
		sum = 0.0;
		for(j=0;j<indim;j++){
			sum += inunits[j]*weight[i*indim+j];
		}
		sum += biasweight[i];
		outunits[i] = activation(sum);
	}
#if 0
	printf("indim: %d\n", indim);
	printf("outdim: %d\n", outdim);
	printf("in units:\n");
	for (i=0; i<indim; i++) {
		printf("%f, ", inunits[i]);
	}
	printf("\n");

	printf("weight:\n");
	for (i=0; i<outdim; i++) {
		for (j=0; j<indim; j++) {
			printf("%lf, ", weight[i*indim + j]);
		}
		printf ("\n");
	}

	printf("bias:\n");
	for (i=0; i<outdim; i++) {
		printf("%lf, ", biasweight[i]);
	}
	printf("\n");
#endif
}

double test_nn(double feature[])
{
	double *weight;
	int i;
	double *new_feature;
	double mean[] = {-1.70496957,0.9875274,-1.19082871,0.60613774,46.51459222,-1.97385195,-1.34022922};
	double std[] = {0.52533208,0.35070016,0.46897356,0.28186897,15.41503896,1.20168569,0.45616171};

	new_feature = malloc(sizeof(double) * input_dim);
	if (new_feature == NULL) {
		perror("malloc\n");
	}

	new_feature = normalize(feature, mean, std, input_dim);

	for (i=0; i<input_dim; i++) {
		input_units[i] = new_feature[i];
	}

	for (i=0; i<(hidden_count + 1); i++) {
		comout(i);
	}

	return output_units[0];
}

// For Test
double feature_test[] = {-1.30, 0.73, -0.84, 0.46, 33.24, -1.23, -0.90};
int main()
{
	double ret;

	ret = test_nn(feature_test);
	printf("result: %lf\n", ret);

#if 0
	FILE *fp;
	fp = fopen("data.txt", "r");
	
	while (fscanf(fp, "%lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf\n", &tmp,
			&feature_test[0],
			&feature_test[1],
			&feature_test[2],
			&feature_test[3],
			&feature_test[4],
			&feature_test[5],
			&feature_test[6]) != -1) {
		printf("count: %d\n", count);
		ret = test_nn(feature_test);
		printf("result is: %f  %f\n", ret, tmp);
		count += 1;
	}
#endif
}
