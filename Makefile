
FLAGS = -std=c++17 -O3 -shared
TEST = -I/home/afeeney/cpplibs/Catch2/
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

CASES = test/stats.cpp test/stat_tracker.cpp test/simple_lsh.cpp

catch:
	g++ -std=c++17 $(TEST) $(EIGEN) -c test/main.cpp  
	g++ -std=c++17 $(TEST) $(EIGEN) -o test/test main.o $(CASES) && ./test/test --success
