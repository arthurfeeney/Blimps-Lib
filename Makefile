
STD = -std=c++17
fPIC = -fPIC `python -m pybind11 --includes`
OUT = -o bind/nr`python-config --extension-suffix`
FILE = bind/nr.cpp
OMP = -fopenmp

binding:
	g++ $(STD) -O3 -shared $(OMP) $(fPIC) -I./ $(FILE) $(OUT)

clean:
	rm nr.o
	rm index_builder.o
	rm bind/nr.so

