
#include <Eigen/Core>
#include <algorithm>
#include <cmath>
#include <utility>
#include <chrono>

#include "include/index_builder.hpp"
#include "include/simple_lsh.hpp"
#include "include/tables.hpp"
#include "include/table.hpp"
#include "include/NR_multitable.hpp"
#include "include/NR_multiprobe.hpp"

/*
 * Contains "experimental" tests.
 * Does not contain anything important, just 
 * some basic checks. 
 */


int main() {

    std::vector<Eigen::VectorXd> data(1000000);
    int64_t i = 0;
    for(auto& vect : data) {
        vect = Eigen::VectorXd::Random(50);
        ++i;
    }

    //NR_MultiTable<Eigen::VectorXd> nr(5, 32, 16, 20);
    NR_MultiProbe<Eigen::VectorXd> nr(64, 60, 50, 50000);

    nr.fill(data);

    Eigen::VectorXd qnn = Eigen::VectorXd::Random(50);
    auto q = qnn / qnn.norm();


    using namespace std::chrono;
    auto start = high_resolution_clock::now();
    //auto p = nr.MIPS(q);//.query(q / q.norm(), .5);
    auto p = nr.probe(q, 500);
    auto end = high_resolution_clock::now();

    auto duration = duration_cast<milliseconds>(end - start);

    std::cout << duration.count() << " milliseconds" << '\n';

    if(p.first) {
        auto t = p.second;

        std::cout << "Inner found " << q.dot(t.first) << '\n';
        std::cout << t.second << '\n';

        //std::cout << t.first;
        //std::cout << data[t.second];

    }
    else {
        std::cout << "Seach failed" << '\n';
    }


    double big_dot = -1;
    int64_t id = 0;

    int rov = 0;
    for(auto& p2 : data) {
        if(q.dot(p2) > big_dot) {
            big_dot = q.dot(p2);
            id = rov;
        }
        ++rov;
    }

    std::cout << "real max: " << big_dot << '\n';
    std::cout << id;

    return 0;
}
