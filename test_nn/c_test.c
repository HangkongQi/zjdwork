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
// #define IN2HIDDEN_INDEX(x, y)  (x + y * net->input_n)
// #define HIDDEN2OUT_INDEX(x, y) (x + y * net->hidden_n)

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
			net->hidden2out_weights[HIDDEN2OUT_INDEX(j, i)] = weight[weight_offset];
			weight_offset += 1;
		}
	}
	// 0,0

	/* input's bias to hidden */
	for (i=0; i<1; i++) {
		for (j=1; j<net->hidden_n; j++) {
			net->in2hidden_weights[IN2HIDDEN_INDEX(i, j)] = weight[weight_offset];
			weight_offset += 1;
		}
	}
    // 0,1  0,2  0,3  0,4  0,5

	/* input to hidden */
	for (i=1; i<net->hidden_n; i++) {
		for (j=1; j<net->input_n; j++) {
			net->in2hidden_weights[IN2HIDDEN_INDEX(j, i)] = weight[weight_offset];
			weight_offset += 1;
		}
	}
	// 1,1   2,1   3,1    4,1   5,1    6,1   7,1
    // 1,2   2,2   3,2    4,2   5,2    6,2   7,2
    // ...

	/* hidden to out */
	for (i=0; i<net->output_n; i++) {
		for (j=1; j<net->hidden_n; j++) {
			net->hidden2out_weights[HIDDEN2OUT_INDEX(j, i)] = weight[weight_offset];
			weight_offset += 1;
		}
	}
	// 1,0  2,0  3,0   4,0  5,0

	return net;
}

void print_weight(NET *net)
{
	int i;
	int j;

	/* hidden's bias to output */
	for (i=0; i<1; i++) {
		for (j=0; j<net->output_n; j++) {
			printf("(%d, %d) %f  [%d]\n", j, i,
						net->hidden2out_weights[HIDDEN2OUT_INDEX(j, i)],
						HIDDEN2OUT_INDEX(j, i));
		}
	}
	// 0,0

	/* input's bias to hidden */
	for (i=0; i<1; i++) {
		for (j=1; j<net->hidden_n; j++) {
			printf("(%d, %d) %f   [%d]\n", i, j,
						net->in2hidden_weights[IN2HIDDEN_INDEX(i, j)],
						IN2HIDDEN_INDEX(i, j));
		}
	}
    // 0,1  0,2  0,3  0,4  0,5

	/* input to hidden */
	for (i=1; i<net->hidden_n; i++) {
		for (j=1; j<net->input_n; j++) {
			printf("(%d, %d) %f    [%d]\n", j, i,
						net->in2hidden_weights[IN2HIDDEN_INDEX(j, i)],
						IN2HIDDEN_INDEX(j, i));
		}
	}
	// 1,1   2,1   3,1    4,1   5,1    6,1   7,1
    // 1,2   2,2   3,2    4,2   5,2    6,2   7,2
    // ...

	/* hidden to out */
	for (i=0; i<net->output_n; i++) {
		for (j=1; j<net->hidden_n; j++) {
			printf("(%d, %d) %f    [%d]\n", j, i,
						net->hidden2out_weights[HIDDEN2OUT_INDEX(j, i)],
						HIDDEN2OUT_INDEX(j, i));
		}
	}
	// 1,0  2,0  3,0   4,0  5,0

}

int getMS(double *m,double *s)
{
	FILE *fp;
	int i = 0,j=0;
	
	fp = fopen("mean.txt","r");
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

void comout_hidden2out(NET *net, double *p1,double *p2,double* c,int n1,int n2)
{   
	int i, j;
	double sum;
	p1[0] = 1.0;

	for (i=0; i<n2; i++) {
		sum = 0.0;
		printf("out[%d]:\n", i);
		for (j=0; j<=n1; j++) {
			printf("%f * %f(%d, %d) = %f\n", p1[j], c[HIDDEN2OUT_INDEX(j, i)],
					j, i, p1[j] * c[HIDDEN2OUT_INDEX(j, i)]);
			sum += p1[j] * c[HIDDEN2OUT_INDEX(j, i)];
		}
		p2[i] = activation(sum);
		printf("out[%d]: %f\n", i, p2[i]);
	}
}

double test_nn(double feature[])
{
	NET *net;
	int in_dim, hidden_dim, out_dim;
	double *weight;
	int i;
	double *mean;
	double *std;
	double *new_feature;

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
	}

	print_weight(net);

	new_feature = malloc(sizeof(double) * in_dim);
	if (new_feature == NULL) {
		perror("malloc\n");
	}

	mean = (double *)malloc(in_dim * sizeof(double));
	std  = (double *)malloc(in_dim * sizeof(double));
	getMS(mean, std);

	new_feature = normalize(feature, mean, std, in_dim);	

	for(i=1;i<net->input_n;i++){
		net->input_units[i]=new_feature[i-1];
		printf("net->input_units %lf\n",net->input_units[i]);
	}

	comout_in2hidden(net, net->input_units,net->hidden_units,net->in2hidden_weights,in_dim,hidden_dim);
	comout_hidden2out(net, net->hidden_units,net->output_units,net->hidden2out_weights,hidden_dim,out_dim);

	return net->output_units[0];
}

// For Test
double feature_test[] = {};
int main()
{
	double ret;
	double tmp;

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
		ret = test_nn(feature_test);
		printf("result is: %f  %f\n", ret, tmp);
	}
}
