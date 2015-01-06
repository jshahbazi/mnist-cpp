#include <iostream>
#include <armadillo>

using namespace arma;

constexpr int training_size = 4000;
constexpr int input_layer_size = 784;
constexpr int hidden_layer_size = 25;
constexpr int num_labels = 10;

constexpr double lambda = 0.1;
constexpr int max_iterations = 100;

int main () {
     std::cout <<  "Starting program..." << std::endl;

     mat x_train(training_size,input_layer_size);
     x_train.load("train_x_4000.csv");
     
     mat initial_theta1(input_layer_size,hidden_layer_size,fill::randu);
     mat initial_theta2(hidden_layer_size,num_labels,fill::randu);
     
     initial_theta1.reshape(1,input_layer_size*hidden_layer_size);
     initial_theta2.reshape(1,hidden_layer_size*num_labels);
     
     mat combined_theta = join_rows(initial_theta1,initial_theta2);
     //vec initial_nn_params = vectorise(combined_theta);
     std::cout << "combined_theta rows: " << combined_theta.n_rows << ", cols: " << combined_theta.n_cols << endl;
     
     initial_theta1 = combined_theta.submat(0, 0, 0, initial_theta1.n_cols-1);
     std::cout << "initial_theta1 rows: " << initial_theta1.n_rows << ", cols: " << initial_theta1.n_cols << endl;
     
     initial_theta2 = combined_theta.submat(0, initial_theta1.n_cols-1, 0,initial_theta1.n_cols-1+initial_theta2.n_cols-1);
     std::cout << "initial_theta2rows: " << initial_theta2.n_rows << ", cols: " << initial_theta2.n_cols << endl;
     
     mat m3(training_size,hidden_layer_size,fill::zeros);
     
     mat y_train(1,training_size);
     y_train.load("train_y_4000.csv");
     

     //m3=x_train*initial_theta1;

     //std::cout << m3.row(0) << std::endl;


     return 0;
}
