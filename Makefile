
FLAGS = -std=c++17 -Wall -pedantic -O3
EIGEN = -I/home/afeeney/cpplibs/Eigen/
PYBIND11 = -I/home/afeeney/cpplibs/pybind11/
fPIC = -fPIC `python -m pybind11 --includes`
OUT = -o bind/nr_binding`python-config --extension-suffix`
FILE = bind/nr.cpp
OMP = -fopenmp

binding:
	g++ $(FLAGS) -fPIC -c include/stats/stats.cpp
	g++ $(FLAGS) -shared $(OMP) $(EIGEN) $(PYBIND11) $(GSL) $(fPIC) stats.o $(FILE) $(OUT);
	cp bind/nr_* movielenstest/;
	cp bind/nr_* synthetic/

clean:
	rm bind/nr_binding.so

TEST = -I/home/afeeney/cpplibs/Catch2/
CASES = test/stats.cpp test/stat_tracker.cpp test/simple_lsh.cpp test/table.cpp test/index_builder.cpp test/tables.cpp

catch:
	g++ -std=c++17 $(TEST) $(EIGEN) -c test/main.cpp
	g++ -std=c++17 $(FLAGS) $(EIGEN) -c include/stats/stats.cpp
	g++ -std=c++17 $(TEST) $(EIGEN) -o test/test main.o stats.o $(CASES) && ./test/test --success

synth:
	g++ $(FLAGS) -fPIC -c include/stats/stats.cpp
	g++ $(FLAGS) $(OMP) $(EIGEN) -o synthetic/synth.o stats.o synthetic/synth.cpp;
	./synthetic/synth.o
