
#pragma once

/*
 * used to track the number of comparisons befored during probing.
 * separate class because I don't want it to be a returned value.
 * I want it to be an optional parameter.
 * This can also be passed in with a python binding.
 */

class CompCounter {
private:
  size_t count = 0;

public:
  CompCounter() {}
  void incr() { ++count; }
};
