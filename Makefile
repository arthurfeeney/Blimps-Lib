
FLAGS = -std=c++17 -Wall -pedantic -O3
EIGEN = -Iexternal/eigen/
PLOT = -Iexternal/matplotlib-cpp -I/usr/include/python2.7 -lpython2.7
PYBIND11 = -Iexternal/pybind11/
fPIC = -fPIC `python -m pybind11 --includes`
OUT = -o bind/nr_binding`python-config --extension-suffix`
FILE = bind/nr.cpp
OMP = -fopenmp

binding:
	g++ $(FLAGS) -shared $(OMP) $(EIGEN) $(PYBIND11) $(GSL) $(fPIC) $(FILE) $(OUT);
	cp bind/nr_* movielenstest/;
	cp bind/nr_* synthetic/
	cp bind/nr_* pyexamples/;

clean:
	rm bind/nr_binding.so

TEST = -Iexternal/Catch2/
CASES = test/stats.cpp test/stat_tracker.cpp test/simple_lsh.cpp test/table.cpp test/index_builder.cpp test/tables.cpp test/nr.cpp test/lsh.cpp test/p_stable_lsh.cpp test/lsh_multi.cpp test/fast_sim.cpp

catch:
	g++ -std=c++17 $(TEST) $(EIGEN) -o test/main.o -c test/main.cpp
	g++ -std=c++17 $(TEST) $(EIGEN) -o test/test test/main.o $(CASES) && ./test/test --success

k_probe_approx:
	g++ $(FLAGS) $(OMP) $(EIGEN) $(PLOT) -o synthetic/k_probe_approx.o synthetic/synth_k_probe_approx.cpp;
	./synthetic/k_probe_approx.o

k_probe_approx_from_probs:
	g++ $(FLAGS) $(OMP) $(EIGEN) $(PLOT) -o synthetic/synth_from_probs.o synthetic/synth_from_probs.cpp;
	./synthetic/synth_from_probs.o

k_probe:
	g++ $(FLAGS) $(OMP) $(EIGEN) $(PLOT) -o synthetic/k_probe.o synthetic/synth_k_probe.cpp;
	./synthetic/k_probe.o
