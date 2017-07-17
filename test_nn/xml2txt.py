from pybrain.tools.customxml.networkreader import NetworkReader

net = NetworkReader.readFrom("./nn.xml")
input_n = net["in"].dim
hidden_n = net["hidden0"].dim
output_n = net["out"].dim

weight = net.params

fid = open("net.txt","w")
fid.write("%d %d %d\n" % (input_n,hidden_n,output_n))
for x in weight:
    fid.write("%s " % str(x))
fid.close()
