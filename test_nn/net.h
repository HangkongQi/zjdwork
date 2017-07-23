#ifndef __NET_H__
#define __NET_H__

unsigned int input_dim = 7;
double input_units[7] = {};
unsigned int hidden0_dim = 5;
double hidden0_units[5] = {};
unsigned int hidden1_dim = 5;
double hidden1_units[5] = {};
unsigned int output_dim = 1;
double output_units[1] = {};
unsigned int hidden_count = 2;

double hidden0_to_hidden1[5][5] = {
	0.936748,
	0.354030,
	0.477037,
	0.339265,
	0.294814,
	0.938711,
	0.281931,
	1.480601,
	0.384895,
	0.328219,
	-0.430867,
	1.179686,
	-0.182559,
	0.520106,
	0.041621,
	1.234053,
	1.077404,
	-0.250074,
	-1.042518,
	-0.257759,
	-1.000048,
	-0.428429,
	0.030733,
	0.638321,
	1.407963,
};

double in_to_hidden0[7][5] = {
	-0.374338,
	0.391419,
	-0.504277,
	0.993465,
	1.062580,
	-1.446146,
	0.863407,
	0.250449,
	0.191565,
	0.530802,
	-0.628967,
	1.227515,
	0.146415,
	-2.084276,
	-0.791533,
	-0.545902,
	1.570673,
	-1.374049,
	-0.454666,
	1.009607,
	1.334264,
	0.555928,
	0.871337,
	-1.202145,
	1.195579,
	-0.356582,
	0.272626,
	-0.310118,
	2.130409,
	-0.505908,
	-0.148633,
	-0.931741,
	-0.619046,
	1.074506,
	0.791803,
};

double hidden1_to_out[5][1] = {
	-0.928341,
	0.014517,
	-2.119419,
	-2.400458,
	0.234783,
};

double bias_to_hidden1[1][5] = {
	0.268517,
	0.760048,
	-0.895993,
	1.613549,
	-1.126294,
};

double bias_to_out[1][1] = {
	-1.624967,
};

double bias_to_hidden0[1][5] = {
	-0.990241,
	1.196292,
	-0.685626,
	-2.863642,
	-1.729075,
};

double *weight_ptr[] = {
	(double *)in_to_hidden0,
	(double *)hidden0_to_hidden1,
	(double *)hidden1_to_out,
};

double *bias_ptr[] = {
	(double *)bias_to_hidden0,
	(double *)bias_to_hidden1,
	(double *)bias_to_out,
};

unsigned int dim_ount = 4;
double dim[] = {7, 5, 5, 1, };
double *units_ptr[] = {
	(double *)input_units,
	(double *)hidden0_units,
	(double *)hidden1_units,
	(double *)output_units,
};

#endif