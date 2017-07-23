from pybrain.tools.customxml.networkreader import NetworkReader
import os

def NetforC(filename):
    os.system("rm -f net.h");
    file = open("net.h", "w");
    file.write("#ifndef __NET_H__\n");
    file.write("#define __NET_H__\n\n");
    net = NetworkReader.readFrom(filename)
    modules = net.modules
    l_modules = [x for x in modules]
#    print l_modules
    '''if bias = true else  hiddenLayer_n = len(l_modules) - 2'''
    hiddenLayer_n = len(l_modules) - 3 
    '''if bias'''
    bias_n = len(l_modules) - 1
    input_n = net['in'].dim
    output_n = net['out'].dim
    hidden_n = []
    dim = [];
    units = [];
    
    bias_w = [];
    connect_w = {};
    file.write("unsigned int input_dim = %d;\n" %input_n);
    dim.append(input_n);
    file.write("double input_units[%d] = {};\n" %(input_n));
    units.append("input_units");
    for i in range(hiddenLayer_n):
        hidden_name = 'hidden%d' % i 
        hidden_n.append(net[hidden_name].dim)
        file.write("unsigned int %s_dim = %d;\n" %(hidden_name, net[hidden_name].dim));
        dim.append(net[hidden_name].dim);
        file.write("double %s_units[%d] = {};\n" %(hidden_name, net[hidden_name].dim));
        units.append("%s_units" %hidden_name);
    file.write("unsigned int output_dim = %d;\n" %output_n);
    file.write("double output_units[%d] = {};\n" %output_n);
    dim.append(output_n);
    file.write("unsigned int hidden_count = %d;\n\n" %len(hidden_n));

    i = 0;
    k = []
    name = [];

    print net.connections;
    for key in net.connections:
        # print net.connections[key]
        if (key.name == "bias"):
            for bias_key in net.connections[key]:
                connect_w[key] = bias_key.params;
                #print "contains key is ", bias_key
                inmod = str(bias_key.inmod.name);
                outmod = str(bias_key.outmod.name);
                indim = bias_key.inmod.dim;
                outdim = bias_key.outmod.dim;
                file.write("double %s_to_%s[%d][%d] = {\n" %(inmod, outmod, indim, outdim));
                for x in connect_w[key]:
                    file.write("\t%lf,\n" %x);
                file.write("};\n\n");
                name.append("%s_to_%s" %(inmod, outmod));
                #print "%s to %s connect" %(inmod, outmod);
                #print connect_w[key]
        elif (net.connections[key]):
            connect_w[key] = net.connections[key][0].params;
            # print "contains key is ", net.connections[key][0];
            inmod = str(net.connections[key][0].inmod.name);
            outmod = str(net.connections[key][0].outmod.name);
            indim = net.connections[key][0].inmod.dim;
            outdim = net.connections[key][0].outmod.dim;
            # print "%s to %s connect" %(inmod, outmod);
            # print connect_w[key]
            file.write("double %s_to_%s[%d][%d] = {\n" %(inmod, outmod, indim, outdim));
            for x in connect_w[key]:
                file.write("\t%lf,\n" %x);
            file.write("};\n\n");
            name.append("%s_to_%s" % (inmod, outmod));
    file.write("double *weight_ptr[] = {\n");
    file.write("\t(double *)in_to_hidden0,\n");
    for x in range(len(hidden_n)-1):
        file.write("\t(double *)hidden%d_to_hidden%d,\n" %(x, x+1));
    file.write("\t(double *)hidden%d_to_out,\n" %(len(hidden_n)-1));
    file.write("};\n\n");

    file.write("double *bias_ptr[] = {\n");
    for x in range(len(hidden_n)):
        file.write("\t(double *)bias_to_hidden%d,\n" %(x));
    file.write("\t(double *)bias_to_out,\n");
    file.write("};\n\n");

    file.write("unsigned int dim_ount = %d;\n" %len(dim));
    file.write("double dim[] = {");
    for x in dim:
        file.write("%d, " %x);
    file.write("};\n");

    file.write("double *units_ptr[] = {\n");
    file.write("\t(double *)input_units,\n");
    for x in range(len(hidden_n)):
        file.write("\t(double *)hidden%d_units,\n" % x);
    file.write("\t(double *)output_units,\n");
    file.write("};\n");

    file.write("#endif");
    file.close();



NetforC("nn.xml")
