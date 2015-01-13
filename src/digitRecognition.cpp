#include <iostream>
#include <armadillo.h>
#include <digitRecognition.h>
#include <mkl.h>

using namespace arma;


//support functions
//cost = fmincg2(10000, nn_params2, k, n , num_labels, X, y, 0.01)
//real function fmincg2(length, nn_params, input_layer_size, hidden_layer_size, num_labels, inputdata, y, lambda)
//gradient1 is empty

//allocate(gradient1(1:size(nn_params), 1 : 1))   -----  
//real, allocatable, intent(inout) :: nn_params(:,:)  ----- combined_theta
//real, allocatable, intent(in) :: inputdata(:,:), y(:,:)  ----- x_train, y_train

void costfunction(mat& gradient1, mat& nn_params, int input_layer_size, int hidden_layer_size, int num_labels, const mat& inputdata, const mat& y, double lambda){
	std::cout << "in costfunction..." << endl;



	mat Theta1 = nn_params.submat(0, 0, 0, hidden_layer_size*(input_layer_size + 1) - 1);
	Theta1.reshape(hidden_layer_size, input_layer_size + 1);
	std::cout << "Theta1 rows: " << Theta1.n_rows << ", cols: " << Theta1.n_cols << endl;

	mat Theta2 = nn_params.submat(0, hidden_layer_size*(input_layer_size + 1) - 1, 0, hidden_layer_size*(input_layer_size + 1) - 1 + num_labels*(hidden_layer_size + 1) - 1);
	Theta2.reshape(num_labels, hidden_layer_size + 1);
	std::cout << "Theta2 rows: " << Theta2.n_rows << ", cols: " << Theta2.n_cols << endl;
	
	//constexpr int training_size = 4000;    //m
	//constexpr int input_layer_size = 784;  //k
	//constexpr int hidden_layer_size = 500;  //n
	// l  y_train size

	int l = y.n_rows;
	int m = inputdata.n_rows;
	int k = input_layer_size;

	mat y_representative(l, num_labels);

	//for (auto i : y_representative){
	//  .row  or .each_col / .each_row ???
	//}



	std::cout << "leaving costfunction..." << endl;
}