#!/bin/bash
source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1

#Command Line Arguments
arg=" -n 1024 -m 16" # set matrix size
src="lab/"

echo ====================
echo mm_dpcpp_ndrange
icpx -fsycl ${src}mm_dpcpp_ndrange.cpp ${src}mm_dpcpp_common.cpp -o ${src}mm_dpcpp_ndrange -w -O3
./${src}mm_dpcpp_ndrange$arg
