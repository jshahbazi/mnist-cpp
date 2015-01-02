#include <iostream>
//#include "std_lib_facilities.h"
#include "Eigen/Dense"

//using namespace std;
using Eigen::MatrixXd;

int main () {
     std::cout <<  "Starting program..." << endl;

     MatrixXd m(10,5);
     m = MatrixXd::Random(10,5);

     MatrixXd m2(5,10);
     m2 = MatrixXd::Random(5,10);

     MatrixXd m3(10,10);

     m3 = m * m2;

     cout << m3;


     return 0;
}
