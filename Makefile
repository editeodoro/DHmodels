CXX = g++
#HEADERS     = potential.h common.h
CXXFLAGS    = -O3 -fPIC -fopenmp 

all: libobj.so

libobj.so: $(HEADERS) DHmodels.o
	$(CXX) $(CXXFLAGS) -shared -Wl,-install_name,libobj.so -o libobj.so DHmodels.o

clean:
	@rm -f *.o libobj.so

