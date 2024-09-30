#!/bin/bash
source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1

#Command Line Arguments
arg=" -n 1024" # set matrix size
src="lab/"

echo ====================
echo mm_dpcpp_basic
icpx -fsycl ${src}mm_dpcpp_basic_full.cpp -o ${src}mm_dpcpp_basic -w -O3
./${src}mm_dpcpp_basic$arg


