
STD = -std=c++17
EIGEN = -I/home/afeeney/cpplibs/Eigen/
PYBIND11 = -I/home/afeeney/cpplibs/pybind11/
GSL = -I/home/afeeney/cpplibs/GSL/include/
fPIC = -fPIC `python -m pybind11 --includes`
OUT = -o bind/nr`python-config --extension-suffix`
FILE = bind/nr.cpp
OMP = -fopenmp

binding:
	g++ $(STD) -O3 -shared $(OMP) $(EIGEN) $(PYBIND11) $(GSL) $(fPIC) $(FILE) $(OUT)

clean:
	rm bind/nr.so

