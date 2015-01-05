#include <iostream>
#include "Eigen/Dense"
#include <armadillo>

//using Eigen::MatrixXd;
//using namespace Eigen;

int main () {
     std::cout <<  "Starting program..." << std::endl;

     Eigen::MatrixXd m(10,5);
     m = Eigen::MatrixXd::Random(10,5);

     Eigen::MatrixXd m2(5,10);
     m2 = Eigen::MatrixXd::Random(5,10);

     Eigen::MatrixXd m3(10,10);

     m3 = m * m2;

     std::cout << m3;


     return 0;
}
