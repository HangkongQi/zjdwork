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
//#define IN2HIDDEN_INDEX(x, y)  (x + y * net->input_n)
//#define HIDDEN2OUT_INDEX(x, y) (x + y * net->hidden_n)

NET *bp_create(int in_dim, int hidden_dim, int out_dim, double *weight)
{
	NET *net;
	int i, j;
	int weight_offset = 0;

	net = (NET*)malloc(sizeof(NET));
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

    

	/* hidden's bias to output */
	for (i=0; i<1; i++) {
		for (j=0; j<net->output_n; j++) {
            printf(" bias to output: %d %d %d\n", i, j, IN2HIDDEN_INDEX(i, j));
            printf("weight_offset: %d\n", weight_offset);
            printf("weight %lf\n", weight[weight_offset]);

			net->hidden2out_weights[HIDDEN2OUT_INDEX(i, j)] = weight[weight_offset];
			weight_offset += 1;
		}
	}

	/* input's bias to hidden */
	for (i=0; i<1; i++) {
		for (j=1; j<net->hidden_n; j++) {
            printf(" bias to hidden: %d %d %d\n", i, j, IN2HIDDEN_INDEX(i, j));
            printf("weight_offset: %d\n", weight_offset);
            printf("weight %lf\n", weight[weight_offset]);

			net->in2hidden_weights[IN2HIDDEN_INDEX(i, j)] = weight[weight_offset];
			weight_offset += 1;
		}
	}

	/* input to hidden */
	for (i=1; i<net->input_n; i++) {
		for (j=1; j<net->hidden_n; j++) {
			printf("in2hidden: %d %d %d\n", i, j, IN2HIDDEN_INDEX(i, j));
			printf("index: %d\n", weight_offset);
			printf("weight %lf\n", weight[weight_offset]);
//			fflush(stdout);
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

	printf("hidden bias to out:  (0 is bias)\n");
	for (i=0; i<1; i++) {
		for (j=0; j<net->output_n; j++) {
			printf("[%d][%d]: %.20lf\n", i, j, net->hidden2out_weights[HIDDEN2OUT_INDEX(i, j)]);
            printf("hidden bias to out %d\n",HIDDEN2OUT_INDEX(i, j));
		}
	}


	printf("input bias to hidden:  (0 is bias)\n");
	for (i=0; i<1; i++) {
		for (j=1; j<net->hidden_n; j++) {
			printf("[%d][%d]: %.20lf\n", i, j, net->in2hidden_weights[IN2HIDDEN_INDEX(i, j)]);
            printf("input bias to hidden %d \n",IN2HIDDEN_INDEX(i, j));
		}
	}
	printf("input to hidden: \n");
	for (i=1; i<net->input_n; i++) {
		for (j=1; j<net->hidden_n; j++) {
			printf("input 2 hidden num %d %d %d\n",IN2HIDDEN_INDEX(i, j),i,j);
            printf("[%d][%d]: %.20lf\n", i, j, net->in2hidden_weights[IN2HIDDEN_INDEX(i, j)]);
		}
	}

	printf("hidden to out: \n");
	for (i=1; i<net->hidden_n; i++) {
		for (j=0; j<net->output_n; j++) {
			printf("[%d][%d]: %.20lf\n", i, j, net->hidden2out_weights[HIDDEN2OUT_INDEX(i, j)]);
            printf("hidden to out %d\n",HIDDEN2OUT_INDEX(i, j));
		}
	}
    printf("******** net->in2hidden_weights[11] = %.20lf\n", net->in2hidden_weights[11]);

}

int getMS(double *m,double *s)
{
	FILE *fp;
	int i = 0,j=0;
	
	fp = fopen("mean.txt","r");
	while(fscanf(fp,"%lf\n",&m[i]) != -1){
//		printf("mean[%d]: %lf\n",i, m[i]);
		i++;
	}
	fclose(fp);

	fp = fopen("std.txt","r");
	while(fscanf(fp,"%lf\n",&s[j]) != -1){
//		printf("std[%d]: %lf\n", j, s[j]);
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
//		printf("new_data[%d]: %lf = (%lf - %lf)/%lf;\n", i, new_data[i], data[i], mean[i], std[i]);
	}
	return new_data;
}

double activation(double x)
{
       return((2.0/(1.0+exp(-2.0*x)))-1);
}

/*void comout(double *p1,double *p2,double* c,int n1,int n2)
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
              printf("c[%d][%d]= %lf\n",k,j,c[j*n1+k]);
          }     
          p2[j] = activation(sum);
      }	
}*/
void comout(double *p1,double *p2,double* c,int n1,int n2)
{
    double sum;
    int j,k;
    p1[0] = 1;
    for(j = 0;j<=(n2-1);j++)
    {
        sum = 0.0;
        for(k=0;k<(n1);k++)
        {
            printf("c[%d][%d]= %lf\n",k,j,c[k*n2+j]);

            sum +=  c[k*n2+j]*p1[k];
            printf("p1[%d] = %lf\n",k,p1[k]);
        }
        printf("sum = %lf\n",sum);
        p2[j] = activation(sum);
        printf("j = %d,i = %d\n",j,k);
        printf("output  = %lf\n",p2[j]);
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
	int weight_number;
    double result;

	FILE *fp;
	fp = fopen("net.txt", "r");
	
	fscanf(fp, "%d %d %d\n", &in_dim, &hidden_dim, &out_dim);

    
	weight_number = (in_dim+1) * (hidden_dim) + (hidden_dim + 1) * out_dim;
//	printf("XXX weight_number: %d\n", weight_number);
	weight = (double *)malloc(weight_number * sizeof(double));

	for (i=0; i<weight_number; i++) {
		double tmp = 0.0;
		fscanf(fp, "%lf", &tmp);
		weight[i] = tmp;
	}

	net = bp_create(in_dim, hidden_dim, out_dim, weight);
	if (net == NULL) {
		printf("Create Net Error\n");
	}

	new_feature = (double *)malloc(sizeof(double) * in_dim);
	if (new_feature == NULL) {
		perror("malloc\n");
	}

	mean = (double *)malloc(in_dim * sizeof(double));
	std  = (double *)malloc(in_dim * sizeof(double));
	getMS(mean, std);

	new_feature = normalize(feature, mean, std, in_dim);	

	for(i=1;i<net->input_n;i++){
		net->input_units[i]=new_feature[i-1];
//		printf("net->input_units %lf\n",net->input_units[i]);
	}

	comout(net->input_units,net->hidden_units,net->in2hidden_weights,net->input_n,net->hidden_n);
	comout(net->hidden_units,net->output_units,net->hidden2out_weights,net->hidden_n,net->output_n);
//    printf("net->hidden_n = %d, ,net->output_n = %d\n",hidden_dim,out_dim);
   
//    print_weight(net);    
 
    result = net->output_units[1];
//    free(net->input_units);
//    free(net->hidden_units);
//    free(net->output_units);
//    free(net->hidden2out_weights);
//    free(net->in2hidden_weights);
    printf("result = %lf\n",net->output_units[0]);
	return result;
}

// For Test
//double feature_test[] = {-0.94,-0.51,24.54,2.78,-1.16,-0.24,31.62,3.25,-0.94,-0.41,14.65,3.12,12.64,-0.74,-0.49}; 
//0.989098
//double feature_test[] = {-2.65,-2.00,99.18,0.00,-1.98,-0.85,81.33,1.57,-2.30,-2.00,100.01,0.00,90.38,-2.08,-1.94};
// -0.999976 
//double feature_test[] = {-1.47,-0.85,38.56,3.12,-1.76,-0.61,11.85,3.61,-0.97,-0.33,16.72,3.77,14.80,-1.33,-0.77};
//0.970248
//double feature_test[] = {-0.91,-0.67,25.73,3.06,-0.94,-0.53,11.46,3.50,-0.40,-0.22,14.24,3.55,12.40,-0.68,-0.51};
//0.988004
//double feature_test[] = {-2.52,-2.00,99.50,0.00,-2.86,-2.00,99.63,0.00,-2.67,-2.00,100.23,0.00,100.00,-10.00,-2.59}; 
//-0.999966
//double feature_test[] = {-1.22,-1.24,45.05,2.73,-0.66,-0.86,16.62,3.32,-0.59,-0.10,10.35,3.90,16.07,-1.07,-0.62};
//0.966759
double feature_test[] = {-1.23,-1.00,23.54,3.33,-2.07,-0.47,32.04,3.90,-1.99,-0.75,56.33,3.08,37.11,-1.17,-1.19};
//-0.277606 
int main()
{
	double ret;

	ret = test_nn(feature_test);
    
//	printf("result is: %lf\n", ret);
	return 0;
}
