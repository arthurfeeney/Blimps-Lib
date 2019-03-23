
FLAGS = -std=c++17 -O3 -shared
EIGEN = -I/home/afeeney/cpplibs/Eigen/
PYBIND11 = -I/home/afeeney/cpplibs/pybind11/
GSL = -I/home/afeeney/cpplibs/GSL/include/
fPIC = -fPIC `python -m pybind11 --includes`
OUT = -o bind/nr_binding`python-config --extension-suffix`
FILE = bind/nr.cpp
OMP = -fopenmp

binding:
	g++ $(FLAGS) $(OMP) $(EIGEN) $(PYBIND11) $(GSL) $(fPIC) $(FILE) $(OUT)

clean:
	rm bind/nr_binding.so
