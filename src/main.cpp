#include <iostream>
#include <armadillo>

using namespace arma;

int main () {
     std::cout <<  "Starting program..." << std::endl;

     mat m1 = randu<mat>(40000,500);
     mat m2 = randu<mat>(500,2000);
     mat m3(40000,2000);
     mat m4(40000,2000);

     m3=m1*m2;

     std::cout << m3.row(0) << std::endl;


     return 0;
}
