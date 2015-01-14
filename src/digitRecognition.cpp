#include <iostream>
#include <armadillo.h>
#include <digitRecognition.h>
//#include <mkl.h>

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

	mat y_representative(l, num_labels,fill::zeros);
	
	//create the  0 1 0 0 0 0 0 0 0 0 style representative matrix
	for (unsigned int i=0;i<=(y.n_rows - 1);i++ ){
		int row_value = y(i,0);
		y_representative(i,row_value) = 1.0;
	}
	
	mat a_1(inputdata.n_rows,inputdata.n_cols + 1,fill::ones);
	a_1.submat(0,1,a_1.n_rows-1,a_1.n_cols-1) = inputdata;
	mat z_2(a_1.n_rows,Theta1.n_rows,fill::zeros);
	
	z_2 = a_1 * Theta1.t();

	//mat a_2(z_2.n_rows,z_2.n_cols+1,fill::ones);
	//sigmoid(z_2,a_2.submat(0,1,a_2.n_rows-1,a_2.n_cols-1));
	
	


	std::cout << "leaving costfunction..." << endl;
}



//         X(:,1:1) = ones
//         X(:,2:) = inputdata        
//         a_1 = X

//         z_2 = 0.0
        
//         call matrix_multiply(a_1,0,Theta1,1,z_2)


//         a_2 = 0.0
//         call sigmoid(z_2,a_2)

        
//         a_2_ones(:,1:1) = ones
//         a_2_ones(:,2:) = a_2
        
//         z_3 = 0.0
//         call matrix_multiply(a_2_ones,0,Theta2,1,z_3)
        
//         call sigmoid(z_3,a_3)

//         summation = 0.0
//         summation = -y_representative * log(a_3) - (1 - y_representative) * log(1 - a_3)
        
//         J = sum(summation)
//         J = J / m





void predict(const mat& Theta1,const mat& Theta2,const mat& inputdata, mat& predictions){
    mat x_holder(inputdata.n_rows,inputdata.n_cols+1,fill::ones);
    mat pre_h1(x_holder.n_rows,Theta1.n_rows);
    mat h1_ones(x_holder.n_rows,Theta1.n_rows+1,fill::ones);
    mat h2(x_holder.n_rows,Theta2.n_rows,fill::zeros);

	std::cout << "inputdata: " << inputdata.n_rows << " " << inputdata.n_cols << endl;
	std::cout << "x_holder: " << x_holder.n_rows << " " << x_holder.n_cols << endl;
    x_holder.submat(0,1,x_holder.n_rows-1,x_holder.n_cols-1) = inputdata;
    
    std::cout << "Theta1: " << Theta1.n_rows << " " << Theta1.n_cols << endl;
    
    pre_h1 = x_holder * Theta1.t();

	std::cout << "pre_h1: " << pre_h1.n_rows << " " << pre_h1.n_cols << endl;

    sigmoid(pre_h1,pre_h1);
    h1_ones.submat(0,1,h1_ones.n_rows-1,h1_ones.n_cols-1) = pre_h1;
    
    std::cout << "h1_ones: " << h1_ones.n_rows << " " << h1_ones.n_cols << endl;
    std::cout << "Theta2: " << Theta2.n_rows << " " << Theta2.n_cols << endl;
    
    h2 = h1_ones * Theta2.t();
    sigmoid(h2,h2);
    
    predictions = arma::max(h2,0);

}

void sigmoid(const mat& input, mat& output){
    output = exp(-input);
    output.transform([](double val){return (1.0/(1.0 + val));});
}

void sigmoidGradient(const mat& input, mat& output){
    output = exp(-input);
    output.transform([](double val){return (1.0/(1.0 + val));});
    output.transform([](double val){return (val * (1.0 - val));});
}
