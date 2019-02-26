#/bin/bash
/usr/local/cuda-8.0/bin/nvcc tf_sampling_g.cu -o tf_sampling_g.cu.o -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC
# in windows add the path of cl.exe to the system environment in oder to eliminate the nvcc compile error
# TF1.2
g++ -std=c++11 tf_sampling.cpp tf_sampling_g.cu.o -o tf_sampling_so.so -shared -fPIC
-I g:\Users\nedu_\Anaconda3\envs\tf_gpu\lib\site-packages\tensorflow\include
# -I /usr/local/cuda-8.0/include
-lcudart
-L /usr/local/cuda-8.0/lib64/ -O2 -D_GLIBCXX_USE_CXX11_ABI=0

#-I g:\Users\nedu_\Anaconda3\envs\tf_gpu\lib\site-packages\tensorflow\include/external/nsync/public
#-L g:\Users\nedu_\Anaconda3\envs\tf_gpu\lib\site-packages\tensorflow -ltensorflow_framework
# C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.0\include
# TF1.4#
#g++ -std=c++11 tf_sampling.cpp tf_sampling_g.cu.o -o tf_sampling_so.so -shared -fPIC
-I /usr/local/lib/python2.7/dist-packages/tensorflow/include -I /usr/local/cuda-8.0/include
-I /usr/local/lib/python2.7/dist-packages/tensorflow/include/external/nsync/public -lcudart
-L /usr/local/cuda-8.0/lib64/ -L/usr/local/lib/python2.7/dist-packages/tensorflow
-ltensorflow_framework
-O2
-D_GLIBCXX_USE_CXX11_ABI=0
