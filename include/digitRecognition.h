#include <armadillo.h>

using namespace arma;

void costfunction(mat&, mat&, int, int, int, const mat&, const mat&, double);
void predict(const mat&,const mat&,const mat&, mat&);
void sigmoid(const mat&, mat&);
void sigmoidGradient(const mat&, mat&);