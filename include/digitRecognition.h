#include <armadillo.h>

using namespace arma;

void fmincg(double&, const int,mat&,const int,const int,const int,mat&,mat&,const double);
void costfunction(double&, mat&, mat&,const int,const int,const int, const mat&, const mat&,const double);
void predict(const mat&,const mat&,const mat&, mat&);
void sigmoid(const mat&, mat&);
void sigmoidGradient(const mat&, mat&);
//void pauseJNS();