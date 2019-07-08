
FLAGS = -std=c++17 -Wall -pedantic -O3
EIGEN = -I/home/afeeney/cpplibs/Eigen/
PLOT = -I/home/afeeney/cpplibs/matplotlib-cpp -I/usr/include/python2.7 -lpython2.7
PYBIND11 = -I/home/afeeney/cpplibs/pybind11/
PLOT = -I/home/afeeney/cpplibs/matplotlib-cpp/ -I/usr/include/python2.7 -lpython2.7
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
CASES = test/stats.cpp test/stat_tracker.cpp test/simple_lsh.cpp test/table.cpp test/index_builder.cpp test/tables.cpp test/nr.cpp

catch:
	g++ -std=c++17 $(TEST) $(EIGEN) -c test/main.cpp
	g++ -std=c++17 $(FLAGS) $(EIGEN) -c include/stats/stats.cpp
	g++ -std=c++17 $(TEST) $(EIGEN) -o test/test main.o stats.o $(CASES) && ./test/test --success

k_probe_approx:
	g++ $(FLAGS) -fPIC -c include/stats/stats.cpp
	g++ $(FLAGS) $(OMP) $(EIGEN) $(PLOT) -o synthetic/k_probe_approx.o stats.o synthetic/synth_k_probe_approx.cpp;
	./synthetic/k_probe_approx.o

k_probe_approx_from_probs:
	g++ $(FLAGS) -fPIC -c include/stats/stats.cpp
	g++ $(FLAGS) $(OMP) $(EIGEN) $(PLOT) -o synthetic/synth_from_probs.o stats.o synthetic/synth_from_probs.cpp;
	./synthetic/synth_from_probs.o

k_probe:
	g++ $(FLAGS) -fPIC -c include/stats/stats.cpp
	g++ $(FLAGS) $(OMP) $(EIGEN) $(PLOT) -o synthetic/k_probe.o stats.o synthetic/synth_k_probe.cpp;
	./synthetic/k_probe.o
