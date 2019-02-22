
STD = -std=c++17
fPIC = -fpic `python -m pybind11 --includes`
OUT = -o bind/nr`python-config --extension-suffix`
FILE = bind/nr.cpp

binding: 
	g++ $(STD) -O3 -shared -Wall $(fPIC) -I . $(OUT) -fopenmp $(FILE)
