#include <stdio.h>
#include <stdlib.h>
#include <math.h>

typedef struct{
    int input_n;
    int hidden_n;
    int output_n;

    double *input_units;
    double *hidden_units;
    double *output_units;
    
    double *in2hidden_weights;
    double *hidden2out_weights;
    
}NET;


// x is number of input, y is number of hidden
#define IN2HIDDEN_INDEX(x, y)  (x * net->hidden_n + y)
#define HIDDEN2OUT_INDEX(x, y) (x * net->output_n + y)

NET *bp_create(int in_dim, int hidden_dim, int out_dim, double *weight)
{
	NET *net;
	int i, j;

	net = malloc(sizeof(NET));
	if (net == NULL) {
		printf("Error alloc\n");
		return NULL;
	}
	
	/* init Net */
	net->input_n = in_dim + 1;
	net->hidden_n = hidden_dim + 1;
	net->output_n = out_dim;

	net->input_units  = (double *)malloc(net->input_n * sizeof(double));
	net->hidden_units = (double *)malloc(net->hidden_n * sizeof(double));
	net->output_units = (double *)malloc(net->output_n * sizeof(double));
	
	net->in2hidden_weights  = (double *)malloc(((net->input_n) * (net->hidden_n)) * sizeof(double));
	net->hidden2out_weights = (double *)malloc(((net->hidden_n) * (net->output_n)) * sizeof(double));

	unsigned int weight_offset = 0;

	/* hidden's bias to output */
	for (i=0; i<1; i++) {
		for (j=0; j<net->output_n; j++) {
			net->hidden2out_weights[HIDDEN2OUT_INDEX(i, j)] = weight[weight_offset];
			weight_offset += 1;
		}
	}

	/* input's bias to hidden */
	for (i=0; i<1; i++) {
		for (j=1; j<net->hidden_n; j++) {
			net->in2hidden_weights[IN2HIDDEN_INDEX(i, j)] = weight[weight_offset];
			weight_offset += 1;
		}
	}

	/* input to hidden */
	for (i=1; i<net->input_n; i++) {
		for (j=1; j<net->hidden_n; j++) {
//			printf("in2hidden: %d %d %d\n", i, j, IN2HIDDEN_INDEX(i, j));
			fflush(stdout);
			printf("index: %d\n", weight_offset);
			printf("weight %lf\n", weight[weight_offset]);
			fflush(stdout);
			net->in2hidden_weights[IN2HIDDEN_INDEX(i, j)] = weight[weight_offset];
			weight_offset += 1;
		}
	}

	/* hidden to out */
	for (i=1; i<net->hidden_n; i++) {
		for (j=0; j<net->output_n; j++) {
			net->hidden2out_weights[HIDDEN2OUT_INDEX(i, j)] = weight[weight_offset];
			weight_offset += 1;
		}
	}

	return net;
}

void print_weight(NET *net)
{
	int i;
	int j;

	printf("input bias to hidden:  (0 is bias)\n");
	for (i=0; i<1; i++) {
		for (j=1; j<net->hidden_n; j++) {
			printf("[%d][%d]: %.20lf\n", i, j, net->in2hidden_weights[IN2HIDDEN_INDEX(i, j)]);
		}
	}
	printf("hidden bias to pit:  (0 is bias)\n");
	for (i=0; i<1; i++) {
		for (j=0; j<net->output_n; j++) {
			printf("[%d][%d]: %.20lf\n", i, j, net->hidden2out_weights[HIDDEN2OUT_INDEX(i, j)]);
		}
	}

	printf("input to hidden: \n");
	for (i=1; i<net->input_n; i++) {
		for (j=1; j<net->hidden_n; j++) {
			printf("[%d][%d]: %.20lf\n", i, j, net->in2hidden_weights[IN2HIDDEN_INDEX(i, j)]);
		}
	}

	printf("hidden to out: \n");
	for (i=1; i<net->hidden_n; i++) {
		for (j=0; j<net->output_n; j++) {
			printf("[%d][%d]: %.20lf\n", i, j, net->hidden2out_weights[HIDDEN2OUT_INDEX(i, j)]);
		}
	}

}

int getMS(double *m,double *s,int size)
{
	FILE *fp;
	int i = 0,j=0;
	int tmp;
	m = (double *)malloc(size * sizeof(double));
	s = (double *)malloc(size * sizeof(double));

	fp = fopen("mean.txt","r");
/*	if(fp == NULL)
	{
		printf("open file error!");
		return -1;		
	}	tmp = fgetc(fp);
	if(tmp == EOF)
	{
		printf("file empty!\n");
		return 0;
	}
	rewind(fp);
	while(!feof(fp))
	{
		fscanf(fp,"%lf",&m[i]);	
		i++;
		printf("%lf\n",m[i]);
	}
*/
	while(fscanf(fp,"%lf\n",&m[i]) != -1){
		printf("%lf\n",m[i]);
	}
	fclose(fp);
	fp = fopen("std.txt","r");
	while(fscanf(fp,"%lf\n",&s[j]) != -1){
		printf("%lf\n",s[j]);
	}
	fclose(fp);	
	return 0;
		
}
double* normalize(double* data,double* mean,double* std,int size)
{
	double* new_data;
        int i;
	new_data = (double*)malloc(size * sizeof(double*));
        for(i=0;i<size;i++)
        {
        //     new_data[i] = (data[i]-mean[i])/std[i];
	     printf("%lf\n",mean[i]);
	    
	//     printf("%lf ",mean[i])
        }
        return new_data;
 }

double activation(double x)
{
       return((2.0/(1.0+exp(-2.0*x)))-1);
}

void comout(double *p1,double *p2,double* c,int n1,int n2)
{   
      double sum;
      int j,k;
      p1[0] = 1;
      for(j = 0;j < n2;j++)
      {
          sum = 0.0;
          for(k = 0;k <= n1;k++)
          {
              sum += c[j*n1+k]*p1[k];
          }     
          p2[j] = activation(sum);
      }	
}

int test_nn(float feature[])
{
	NET *net;
	int in_dim, hidden_dim, out_dim;
	double *weight;
	int i;
	double *mean;
	double *std;

	FILE *fp;
	fp = fopen("net.txt", "r");
	
	fscanf(fp, "%d %d %d\n", &in_dim, &hidden_dim, &out_dim);

	unsigned int weight_number;
	weight_number = (in_dim+1) * (hidden_dim) + (hidden_dim + 1) * out_dim;
	printf("XXX weight_number: %d\n", weight_number);
	weight = malloc(weight_number * sizeof(double));

	for (i=0; i<weight_number; i++) {
		double tmp = 0.0;
		fscanf(fp, "%lf", &tmp);
		weight[i] = tmp;
	}

	net = bp_create(in_dim, hidden_dim, out_dim, weight);
	if (net == NULL) {
		printf("Create Net Error\n");
		return -1;
	}

//	print_weight(net);
	for(i=1;i<net->input_n;i++){
	
		net->input_units[i]=feature[i];
		
//		printf("net->input_units %lf\n",net->input_units[i]);
	}
	comout(net->input_units,net->hidden_units,net->in2hidden_weights,in_dim,hidden_dim);
	comout(net->hidden_units,net->output_units,net->hidden2out_weights,hidden_dim,out_dim);
}

// For Test
double feature_test[] = {-0.93, 0.68, -0.15, 0.12, 6.73, -0.55, -0.35};
int main()
{
	int f_size = 0,i;
	f_size =sizeof(feature_test)/sizeof(feature_test[0]); 
	double *mean,*std;
	double* new_feature;
	getMS(mean,std,f_size);
	printf("%lf\n",mean[1]);
//	new_feature = normalize(feature_test,mean,std,f_size);
	
//	test_nn(new_feature);	
}
