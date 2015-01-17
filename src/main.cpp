#include <iostream>
#include <armadillo.h>
#include <digitRecognition.h>
//#include <mkl.h>

using namespace arma;

constexpr int training_size = 100;    //m
constexpr int input_layer_size = 784;  //k
constexpr int hidden_layer_size = 500;  //n
constexpr int num_labels = 10;

constexpr double lambda = 0.1;
constexpr int max_iterations = 100;


int main () {
//	std::cout << "Starting program with " << mkl_get_max_threads() << " threads..." << std::endl;

	//mkl_set_num_threads(4);

    mat predictions(training_size,1,fill::zeros);
    
    mat x_train(training_size,input_layer_size,fill::randu);
    x_train.load("train_x_100.csv");
    
    mat y_train(1,training_size,fill::randu);
    y_train.load("train_y_100.csv");
    
    //calculate mean of the training data
    double x_mean = sum(sum(x_train));
    x_mean /= x_train.n_elem;  //33.5026  TODO: check this
    
    //calculate standard devication of the training data
    mat x_train_mean(training_size,input_layer_size);
    x_train_mean = x_train;
    x_train_mean.transform( [x_mean](double val){return (val - x_mean);} );
    double x_std = sum(sum(x_train_mean));
    x_std /= (x_train_mean.n_elem);
    x_std = std::sqrt(x_std);  //7.23942e-06  TODO: check this
    
    
	mat initial_theta1(hidden_layer_size, input_layer_size + 1, fill::ones);  //784 26   n k+1
	initial_theta1.load("theta1.csv");
	mat initial_theta2(num_labels, hidden_layer_size + 1, fill::ones);        //25 11     numlabels n+1
	initial_theta2.load("theta2.csv");

	initial_theta1.reshape(hidden_layer_size*(input_layer_size + 1),1);
	initial_theta2.reshape(num_labels*(hidden_layer_size + 1),1);
    
    mat combined_theta = join_cols(initial_theta1,initial_theta2);
    //std::cout << "combined_theta rows: " << combined_theta.n_rows << ", cols: " << combined_theta.n_cols << endl;
    

	//test section----------------------
	//mat gradient1 = combined_theta;
	//double cost=0.0;
	//costfunction(cost, gradient1, combined_theta, input_layer_size, hidden_layer_size, num_labels, x_train, y_train, lambda);
	fmincg(max_iterations,combined_theta,input_layer_size,hidden_layer_size,num_labels,x_train,y_train,lambda);
    //----------------------------------


    
    initial_theta1 = combined_theta.submat(0, 0, initial_theta1.n_cols-1, 0);
	initial_theta1.reshape(hidden_layer_size, input_layer_size + 1);
    //std::cout << "initial_theta1 rows: " << initial_theta1.n_rows << ", cols: " << initial_theta1.n_cols << endl;
    
    initial_theta2 = combined_theta.submat(initial_theta1.n_cols-1, 0,initial_theta1.n_cols-1+initial_theta2.n_cols-1, 0);
	initial_theta2.reshape(num_labels, hidden_layer_size + 1);
    //std::cout << "initial_theta2 rows: " << initial_theta2.n_rows << ", cols: " << initial_theta2.n_cols << endl;
    
    predict(initial_theta1,initial_theta2,x_train,predictions);
    
    //std::cout << "predictions: " << predictions.row(0) << endl;



	std::cout << std::endl << "Press ENTER to continue...";
	std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    
    return 0;
}




//         real function costandgradient(gradient, nn_params,input_layer_size,hidden_layer_size,num_labels, inputdata, y, lambda)

//         real, allocatable :: inputdata(:,:), X(:,:), y(:,:), y_representative(:,:), gradient(:,:), ones(:,:)
//         real, allocatable :: a_1(:,:),z_2(:,:),a_2(:,:),z_3(:,:),a_3(:,:),hofx(:,:)
//         real, allocatable :: delta_2(:,:), temp_delta_2(:,:), delta_3(:,:), gradient_1(:,:), gradient_2(:,:)
//         real, allocatable :: Theta1_grad(:,:), Theta2_grad(:,:)
//         integer :: input_layer_size,hidden_layer_size,num_labels,m,l,K,i
//         integer :: a1,a2,b1,b2,c1,c2
//         real :: lambda, temp, J
//         real, allocatable :: nn_params(:,:), Theta1(:,:), Theta2(:,:), temptheta1(:,:),temptheta2(:,:)
//         real, allocatable :: a_2_ones(:,:), summation(:,:), summation2(:,:), summation3(:,:),z_2_ones(:,:), sigmoid_z_2(:,:)
    
//         real :: timer1, timer2, holder1, holder2
//         double precision :: average1 = 0.0
//         double precision :: average2 = 0.0
//         double precision :: average3 = 0.0
//         double precision :: counter = 0.0
    
        
        
//         m = size(inputdata,1)
//         l = size(y,1)
//         K = num_labels

//         allocate(Theta1(1:hidden_layer_size,1:(input_layer_size+1)))
//         allocate(Theta2(1:num_labels,1:(hidden_layer_size+1)))
//         allocate(ones(m,1))
//         allocate(X(size(inputdata,1),(size(inputdata,2)+1)))        
//         allocate(y_representative(l,num_labels))        
//         allocate(z_2(m,hidden_layer_size))
//         allocate(a_1(size(X,1),size(X,2)))
//         allocate(a_2(size(z_2,1),size(z_2,2)))              
//         allocate(a_2_ones(size(a_2,1),(size(a_2,2)+1)))      
//         allocate(z_3(size(a_2_ones,1),size(Theta2,1)))  !a_2 * Theta2'      
//         allocate(temptheta2(size(Theta2,2),size(Theta2,1)))        
//         allocate(a_3(size(z_3,1),size(z_3,2)))        
//         allocate(summation(size(z_3,1),size(z_3,2)))
//         allocate(summation2(size(z_3,1),size(z_3,2)))
//         allocate(summation3(size(z_3,1),size(z_3,2)))
//         allocate(delta_3(size(a_3,1),size(a_3,2)))        
//         allocate(z_2_ones(size(z_2,1),(size(z_2,2)+1)))        
//         allocate(sigmoid_z_2(size(z_2_ones,1),size(z_2_ones,2)))        
//         allocate(temp_delta_2(size(delta_3,1),size(Theta2,2)))   
//         allocate(delta_2(size(temp_delta_2,1),size(sigmoid_z_2,2)))
//         allocate(gradient_2(size(delta_3,2),size(a_2_ones,2)))        
//         allocate(gradient_1(size(delta_2,2)-1,size(a_1,2)))
//         allocate(Theta2_grad(1:size(gradient_2),1))
//         allocate(Theta1_grad(1:size(gradient_1),1))        
        
//         allocate(temptheta1(size(Theta1,2),size(Theta1,1)))  
        

//         Theta1 = reshape(nn_params(1:hidden_layer_size*(input_layer_size + 1),1), (/ hidden_layer_size, (input_layer_size + 1) /))
//         Theta2 = reshape(nn_params((1+hidden_layer_size*(input_layer_size + 1)):,1), (/ num_labels, (hidden_layer_size + 1) /))
        
//         !print *,shape(Theta1)   !500 x 785
//         !print *,shape(Theta2)   !10  x 501
        
        
//         ones = 1.0



        
//         !timer1=secnds(0.0)


//         do i=1,m
//             temp = y(i,1)
//             do j=1,K
//                 if(temp == j)then
//                     y_representative(i,j) = 1
//                 else
//                     y_representative(i,j) = 0
//                 end if
//             end do
//         end do

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
        
        

        
//         !do concurrent (i = 1:size(delta_3,1))
//         !    delta_3(i,:) = a_3(i,:) - y(1,:)
//         !end do
        
//         delta_3 = a_3 - y_representative

//         z_2_ones(:,1:1) = ones
//         z_2_ones(:,2:) = z_2   
        

//         sigmoid_z_2 = 0.0
        
     
//         call sigmoidgradient(z_2_ones,sigmoid_z_2)  


//         !temp_delta_2 = matmul(delta_3,Theta2)

// !timer1=secnds(0.0)   
//         !a1 = size(delta_3,1)
//         !a2 = size(delta_3,2)
//         !b1 = size(Theta2,1)
//         !b2 = size(Theta2,2)
//         !call sGEMM('N','N',a1,b2,b1,1.0,delta_3,a1,Theta2,b1,0.0,temp_delta_2,a1)
//         call matrix_multiply(delta_3,0,Theta2,0,temp_delta_2)
// !timer2=secnds(timer1)
// !average1 = average1 + timer2
// !counter = counter + 1
// !print '(a, f8.4)','matrix_multiply timer: ', average1/counter        
        
//         delta_2 =0.0
//         !delta_2 = temp_delta_2 * sigmoid_z_2   !elemental multiplication
//         call vsMul( size(temp_delta_2), temp_delta_2, sigmoid_z_2, delta_2 )   !elemental multiplication    

        
//         gradient_2 = 0.0
//         gradient_1 = 0.0
        
//         call matrix_multiply(delta_3,1,a_2_ones,0,gradient_2)
        
//         call matrix_multiply(delta_2(:,2:),1,a_1,0,gradient_1)        
        
//         gradient_2 = gradient_2 / m
//         gradient_1 = gradient_1 / m
        
//         gradient_2(:,2:) = gradient_2(:,2:) + Theta2(:,2:) * (lambda / m)
//         gradient_1(:,2:) = gradient_1(:,2:) + Theta1(:,2:) * (lambda / m)


//         Theta2_grad = reshape(gradient_2, (/size(gradient_2),1/))
//         Theta1_grad = reshape(gradient_1, (/size(gradient_1),1/))

//         gradient(1:size(Theta1_grad),1:1) = Theta1_grad
//         gradient((size(Theta1_grad)+1):,1:1) = Theta2_grad
        
//         costandgradient = J
        
        
//         !print '(f8.4, a, f8.4)',average1/counter,' ',average2/counter
//         !timer2=secnds(timer1)
//         !average = average*0.9 + timer2*0.1
//         !print '(a, f8.4)','costandgradient timer: ', timer2
//     end function costandgradient