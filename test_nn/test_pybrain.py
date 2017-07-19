from pybrain.tools.customxml.networkreader import NetworkReader
import numpy as np
def getMS():
	mean = []
	std = []
	fid = open("mean.txt","r")
	for line in fid:
		tmp = line.split()
		mean.append(float(tmp[0]))
	fid.close()
	
	fid = open("std.txt","r")
	for line in fid:
		tmp = line.split()
		std.append(float(tmp[0]))
	fid.close()
	return mean,std

feat = [-1.51, 1.46, -0.56, 0.36, 20.33, -0.98, -0.80]
test_net = NetworkReader.readFrom("./nn.xml")
mean,std = getMS()
mean = np.array(mean)
std = np.array(std)
fea = np.array(feat)
f = (fea-mean)/std
result = test_net.activate(f)
print 'result',result
