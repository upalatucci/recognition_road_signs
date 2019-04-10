import test_bench
import os
import numpy

min_ne = 15
Nmax_ne = 25

start_scale=1.01
scale_steps=0.01
Nmax_scale = 1.1

dir=os.getcwd()+"\XML_indicator\ind_1kpos_2kneg_20steps.xml"

TYPE = 0 # 0 -indicator  1 warnings 2 prohibitory
result_map = {}
max_fscore = 0
max_map = []
for i in range(min_ne,Nmax_ne):
    for j in numpy.arange(start_scale,Nmax_scale,scale_steps):

        result_map[(i,j)] =  test_bench.test(dir,i,j,TYPE)

        if(result_map[(i,j)]>max_fscore):
            max_fscore  =  result_map[(i,j)]
            tuple=(max_fscore,i,j)
            max_map.append(tuple)


print(max_fscore,max_map)
print("Massimo= "+str(max_map.pop()))
