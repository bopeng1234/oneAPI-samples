INCLUDES =
LDFLAGS  = -lm
CFLAGS   = -Wall -Wformat-security -Werror=format-security
CXXFLAGS = -Wall -Wformat-security -Werror=format-security
# Use icx from the DPC++ oneAPI toolkit to compile. Please source DPCPP's vars.sh before compilation.
CC       = mpiicx
CXX      = mpiicpx
example  = mpi3_onesided_jacobian

all: CFLAGS += -O2
all: CXXFLAGS += -O2
all: $(example)

debug: CFLAGS += -O0 -g
debug: CXXFLAGS += -O0 -g
debug: $(example)

%(example): ../include/common.h

% : %.c
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ $< $(LDFLAGS)

% : %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ $< $(LDFLAGS)

clean:
	-rm -f $(example).o $(example)
