#include <iostream>
#include <armadillo.h>
#include <digitRecognition.h>
//#include <mkl.h>
#include <iomanip>

using namespace arma;


//support functions
//cost = fmincg2(10000, nn_params2, k, n , num_labels, X, y, 0.01)
//real function fmincg2(length, nn_params, input_layer_size, hidden_layer_size, num_labels, inputdata, y, lambda)

void fmincg(double& finalcost, const int length,mat& nn_params,const int input_layer_size,const int hidden_layer_size,const int num_labels,mat& inputdata,mat& y,const double lambda){
// Copyright (C) 2001 and 2002 by Carl Edward Rasmussen. Date 2002-02-13
// (C) Copyright 1999, 2000 & 2001, Carl Edward Rasmussen
// 
// Permission is granted for anyone to copy, use, or modify these
// programs and accompanying documents for purposes of research or
// education, provided this copyright notice is retained, and note is
// made of any changes that have been made.
// 
// These programs and documents are distributed without any warranty,
// express or implied.  As the programs were written for research
// purposes only, they have not been tested to the degree that would be
// advisable in any important application.  All use of these programs is
// entirely at the user's own risk.
//
// [ml-class] Changes Made:
// 1) Function name and argument specifications
// 2) Output display
//
// [John Shahbazian] Changes Made:
// 1) Ported to C++ using the Armadillo (http://arma.sourceforge.net/) library
// 2) Change the cost function call to internal.  Replace the
//    'costfunction' function to whatever you would like.  It returns
//    the cost as the result, and the gradient is returned through the first 
//    argument as an intent(inout) (e.g. 'gradient#').  
// 3) Changed the variable names to be readable.
//    f1 = cost1
//    df1 = gradient1
//    s = search_direction
//    d1 = slope1
//    z1 = point1
//    X0 = backup_params
//    f0 = cost_backup
//    df0 = gradient_backup 

	const double RHO = 0.01;
	const double SIG = 0.5;
	const double INT = 0.1;
	const double EXT = 3.0;
	const int MAXEVALS = 20;
	const double RATIO = 100;
	
	double mintemp, minstuff, M, A, B;
	double fX = 0.0;
	int success;
	int i=0;
	int ls_failed = 0;
	
	mat backup_params = nn_params;
	mat gradient2(nn_params.n_rows,nn_params.n_cols,fill::zeros);
	mat gradient3(nn_params.n_rows,nn_params.n_cols,fill::zeros);
	mat gradient_backup(nn_params.n_rows,nn_params.n_cols,fill::zeros);
	mat tmp(nn_params.n_rows,nn_params.n_cols,fill::zeros);
	double limit;
	double point1, point2, point3;
	double cost1, cost2, cost3, cost_backup;
	double slope1, slope2, slope3;
	mat search_direction(nn_params.n_rows,nn_params.n_cols,fill::zeros);
	mat stemp(nn_params.n_rows,nn_params.n_cols,fill::zeros);
	
	double sqrtnumber;
	//mat sd_calc_1,sd_calc_2,sd_calc_3;
	//double sd_calc_4;
	//mat sd_calc_5(nn_params.n_rows,nn_params.n_cols,fill::zeros);

	
	cost1 = 10000.0;  //lower is better, so init with high
	mat gradient1(nn_params.n_rows,nn_params.n_cols,fill::zeros);
	costfunction(cost1, gradient1, nn_params, input_layer_size, hidden_layer_size, num_labels, inputdata, y, lambda);
	//std::cout << "gradient1: " << gradient1(0,0) << endl;
	//pause();

	//i = i + (length<0);
	search_direction = -gradient1;

	mat slope_vector(1,1);
	slope_vector = -search_direction.t() * search_direction;
	slope1 = slope_vector(0,0);
	point1 = 1.0/(1.0 - slope1);
	//std::cout << "point1: " << point1 << endl;

	while(i < std::abs(length)){
		i = i + 1;
		//std::cout << "loop: " << i << endl;
		backup_params = nn_params;
		cost_backup = cost1;
		gradient_backup = gradient1;
		stemp = point1 * search_direction;
		nn_params = nn_params + stemp;
		//std::cout << "nn_params: " << nn_params.row(0) << endl;

		gradient2 = gradient1;
		
		cost2 = 10000.0;
		costfunction(cost2, gradient2, nn_params, input_layer_size, hidden_layer_size, num_labels, inputdata, y, lambda);
		
		//i = i + (length<0);
		//i++;
		//std::cout << "i: " << i << endl;
		slope_vector = gradient2.t() * search_direction;
		slope2 = slope_vector(0,0);
		//std::cout << "slope2: " << slope2 << endl;


		cost3 = cost1;
		slope3 = slope1;
		point3 = -point1;
		if(length>0)M=MAXEVALS;
		else M = std::min(MAXEVALS,-length-i);
		success=0;
		limit=-1;
		
		while(1){ //3.66252   3.66252  0 -5.40439
			//std::cout << "cost2: " << cost2 << " cost1: " << cost1 << " point1: " << point1 << " slope1: " << slope1 << endl;
			//std::cout << "cost2 vs stuff: " << cost2 << "   " << (cost1 + (point1 * RHO * slope1)) << endl;
			//std::cout << "slope2 vs stuff: " << slope2 << "   " << (-SIG * slope1) << endl;
			//std::cout << "M: " << M << endl;
			//pause();
			while(( (cost2 > (cost1 + (point1 * RHO * slope1))) || (slope2 > (-SIG * slope1)) ) && (M > 0) ){
				//std::cout << "here*******************" << endl;
				limit = point1;
				if(cost2 > cost1)
                    point2 = point3 - (0.5 * slope3 * point3 * point3)/(slope3 * point3 + cost2 - cost3);  //quadratic fit
                else{
                    A = 6*(cost2 - cost3)/point3 + 3*(slope2 + slope3);           //cubic fit
                    B = 3*(cost3 - cost2) - point3 * (slope3 + 2*slope2);
                    point2 = (std::sqrt(B*B - A*slope2*point3*point3) - B)/A;
				}
                if(std::isnan(point2) || (!std::isfinite(point2)))
                    point2 = point3 / 2;                         // if we had a numerical problem then bisect
                point2 = std::max( std::min(point2, (INT * point3)), ((1.0 - INT) * point3));  //don't accept too close to limits
                point1  = point1 + point2;                       // update the step
                stemp = point2 * search_direction;
                nn_params = nn_params + stemp;

                costfunction(cost2, gradient2, nn_params, input_layer_size, hidden_layer_size, num_labels, inputdata, y, lambda);

                M = M - 1.0;
				//std::cout << "i2: " << i << endl;
                //i = i + (length<0);                              // count epochs?!
				//std::cout << "i2: " << i << endl;
                slope_vector = gradient2.t() * search_direction;
                slope2 = slope_vector(0,0);                      //convert to scalar
                point3 = point3 - point2;                        // point3 is now relative to the location of point2                

			}
			
            if ((cost2 > (cost1 + (point1*RHO*slope1)) ) || (slope2 > (-SIG * slope1) )){
				//std::cout << "Break -----------------" << endl;
				break;                                            // this is a failure

            }
            else if (slope2 > (SIG * slope1)){
				//std::cout << "Break -----------------" << endl;
                success = 1;
                break;                                            // success
			}
            else if (M == 0.0){
				//std::cout << "Break -----------------" << endl;
                break;                                            // failure
			}

            A = 6*(cost2 - cost3)/point3 + 3*(slope2 + slope3);  // make cubic extrapolation
            B = 3*(cost3 - cost2) - point3*(slope3 + 2*slope2);
            sqrtnumber = (B*B) - A*slope2*point3*point3;
            
			if((!std::isnormal(sqrtnumber)) || (sqrtnumber < 0.0) ){
				//std::cout << "sqrt ifs" << endl;
                if (limit < -0.5)                          // if we have no upper limit
                    point2 = point1  * (EXT - 1);                // the extrapolate the maximum amount
                else
                    point2 = (limit - point1) / 2;               // otherwise bisect
				//std::cout << "point2 sqrt ifs: " << point2 << endl;
            }
            else{
                point2 = (-slope2 * point3 * point3)/(B + std::sqrt(sqrtnumber));
                if ((limit > -0.5) && ((point2 + point1) > limit))          // extraplation beyond max?
                    point2 = (limit - point1)/2;                 // bisect
                else if ((limit < -0.5) && ((point2 + point1) > (point1 * EXT)))       // extrapolation beyond limit
                    point2 = point1 * (EXT - 1.0);               // set to extrapolation limit
                else if (point2 < (-point3 * INT))
                    point2 = -point3 * INT;
                else if ((limit > -0.5) && (point2 < (limit - point1)*(1.0 - INT)))   // too close to limit?
                    point2 = (limit - point1 ) * (1.0 - INT);
            }			
			
            cost3 = cost2;
            slope3 = slope2;
            point3 = -point2;               
            point1  = point1 + point2;

			//std::cout << "search_direction: " << endl << search_direction.rows(0, 10) << endl;
            stemp = point2 * search_direction;
			//std::cout << "point2: " << point2 << endl;

            nn_params = nn_params + stemp;                       // update current estimates
			//std::cout << "nn_paramsb: " << endl << nn_params.rows(0,10) << endl;

            costfunction(cost2, gradient2, nn_params, input_layer_size, hidden_layer_size, num_labels, inputdata, y, lambda);
			//std::cout << "cost2b: " << cost2 << endl;
            M = M - 1.0;
			//std::cout << "i: " << i << endl;
            //i = i + (length<0);                                  // count epochs?!
			//std::cout << "i: " << i << endl;
			slope_vector = gradient2.t() * search_direction;
            slope2 = slope_vector(0,0);                          //convert to scalar
			//std::cout << "slope2b: " << slope2 << endl;
			//pause();
		}
		
        if (success == 1){                                   // if line search succeeded
            cost1 = cost2;
            fX = cost1;
            std::cout << "Iteration: " << i << " | Cost: " << cost1 << endl;
            mat sd_calc_1 = gradient2.t() * gradient2;
            mat sd_calc_2 = gradient1.t() * gradient2;
            mat sd_calc_3 = gradient1.t() * gradient1;
            double sd_calc_4 = (sd_calc_1(0,0) - sd_calc_2(0,0)) / sd_calc_3(0,0);
            mat sd_calc_5 = sd_calc_4 * search_direction;
            
            search_direction = sd_calc_5 - gradient2;
            tmp = gradient1;
            gradient1 = gradient2;
            gradient2 = tmp;                                     // swap derivatives
			slope_vector = gradient1.t() * search_direction;
            slope2 = slope_vector(0,0);                          //convert to scalar
            if(slope2 > 0.0){                                  // new slope must be negative
                search_direction = -gradient1;                   // otherwise use steepest direction
				slope_vector = -search_direction.t() * search_direction;
                slope2 = slope_vector(0,0);                      //convert to scalar
            }
			mintemp = slope1 / (slope2);// - std::numeric_limits<double>::lowest());  //std::numeric_limits<double>::lowest() is min value double precision float //TODO: figure out why the min number is needed
            minstuff = std::min(RATIO, mintemp);
            point1  = point1 * minstuff;                         // slope ratio but max RATIO
            slope1 = slope2;
            ls_failed = 0;                                        // this line search did not fail
        }
        else{
            nn_params = backup_params;
            cost1 = cost_backup;
            gradient1 = gradient_backup;                         // restore point from before failed line search
            if (ls_failed == 1 || (i > std::abs(length)))       // line search failed twice in a row
				//std::cout << "Break -----------------" << endl;
                break;                                            // or we ran out of time, so we give up
            tmp = gradient1;
            gradient1 = gradient2;
            gradient2 = tmp;                                    // swap derivatives
            search_direction = -gradient1;                      // try steepest
            slope_vector = -search_direction.t() * search_direction;
            slope1 = slope_vector(0,0);                         // convert to scalar
            point1  = 1.0 / (1.0 - slope1);
            ls_failed = 1;                                      // this line search failed
        }		
       
	}
	
	finalcost = fX;		//return finalcost
}


void costfunction(double& cost, mat& gradient, mat& nn_params,const int input_layer_size,const int hidden_layer_size,const int num_labels, const mat& inputdata, const mat& y,const double lambda){
	//std::cout << "in costfunction..." << endl;



	mat Theta1 = nn_params.submat(0, 0, hidden_layer_size*(input_layer_size + 1) - 1, 0);
	Theta1.reshape(hidden_layer_size, input_layer_size + 1);
	//std::cout << "Theta1 rows: " << Theta1.n_rows << ", cols: " << Theta1.n_cols << endl;

	mat Theta2 = nn_params.submat(hidden_layer_size*(input_layer_size + 1), 0, hidden_layer_size*(input_layer_size + 1) - 1 + num_labels*(hidden_layer_size + 1) - 1, 0);
	Theta2.reshape(num_labels, hidden_layer_size + 1);
	//std::cout << "Theta2 rows: " << Theta2.n_rows << ", cols: " << Theta2.n_cols << endl;
	
	//std::cout << "Theta2: " << endl << Theta2.col(0) << endl;
	//pause();

	//constexpr int training_size = 4000;    //m
	//constexpr int input_layer_size = 784;  //k
	//constexpr int hidden_layer_size = 500;  //n
	// l  y_train size

	int l = y.n_rows;
	int m = inputdata.n_rows;
	//int k = input_layer_size;

	//create the  0 1 0 0 0 0 0 0 0 0 style representative matrix
	mat y_representative(l, num_labels,fill::zeros);
	for (unsigned int i=0;i<=(y.n_rows - 1);i++ ){
		int row_value = y(i,0);
		y_representative(i,row_value) = 1.0;
	}
	//std::cout << "y_rep: " << y_representative.row(0) << endl;
	


	//setup a_2
	mat a_1(inputdata.n_rows,inputdata.n_cols + 1,fill::ones);
	a_1.submat(0,1,a_1.n_rows-1,a_1.n_cols-1) = inputdata;
	mat z_2(a_1.n_rows,Theta1.n_rows,fill::zeros);

	//std::cout << "a_1: " << a_1(0, 0) << endl;

	//std::cout << "Theta1: " << Theta1(1, 2) << endl;

	//std::cout << "Theta1.t(): " << theta1t.row(0) << endl;

	z_2 = a_1 * Theta1.t();
	//std::cout << "z_2: " << z_2.row(0) << endl;
	//pause();


	mat a_2=z_2;
	mat a_2_ones(z_2.n_rows,z_2.n_cols+1,fill::ones);
	
	sigmoid(a_2,a_2);
	//std::cout << "a_2: " << a_2.row(0) << endl;
	//pause();



	a_2_ones.submat(0,1,a_2_ones.n_rows-1,a_2_ones.n_cols-1)=a_2;
	//std::cout << "a_2_ones: " << a_2_ones.n_rows << endl;
	//std::cout << "a_2_ones: " << a_2_ones.row(0) << endl;
	//pause();

	mat theta2t = Theta2.t();
	//std::cout << "theta2t: " << theta2t.row(0) << endl;
	//pause();


	//setup a_3
	mat z_3(a_2_ones.n_rows,Theta2.n_rows,fill::zeros);
	z_3 = a_2_ones * Theta2.t();
	//std::cout << "z_3: " << z_3.row(0) << endl;
	//pause();
	
	mat a_3 = z_3;
	//std::cout << "a_3: " << a_3(0, 0) << endl;
	sigmoid(a_3,a_3);



	//mat test1;

	//test1 << 1.0 << 2.0 << 3.0 << endr
	//	  << 4.0 << 5.0 << 6.0 << endr;

	//mat test2 = test1;
	//mat test3 = test1;
	//mat test2l = test2.transform([](double val){ return (log(val)); });

	//test3 = test1 % test2l;
	//std::cout << test3 << endl;
	//pause();

	//test2l = log(test2);
	//test3 = test1 % test2l;
	//std::cout << test3 << endl;
	//pause();




	//calculate cost
	mat summation(z_3.n_rows,z_3.n_cols,fill::zeros);

//	mat a_3l1 = a_3;
//	a_3l1.transform([](double val){ return (log(val)); });

	//std::cout << "log(a_3): " << log(a_3.row(0)) << endl;
	//std::cout << "transform log(a_3): " << a_3l1.row(0) << endl;
	//pause();
	
//	mat a_3l2 = a_3;
//	a_3l2.transform([](double val){ return (log(1 - val)); });

	summation = -y_representative % log(a_3) - ((1 - y_representative) % log(1-a_3));
	//std::cout << "summation: " << summation.row(0) << endl;
	//pause();
	cost = accu(summation);
	//std::cout << "Cost: " << cost << " m: " << m << endl;
	cost /= m;
	//std::cout << "Cost: " << cost << endl;
	//pause();








	//setup delta_3
	mat delta_3 = y_representative;
	delta_3 = a_3 - y_representative;
	//std::cout << "delta_3: " << delta_3.row(0) << endl;
	//pause();	
	
	//setup sigmoid of z_2
	mat z_2_ones(z_2.n_rows, z_2.n_cols + 1, fill::ones);
	z_2_ones.submat(0,1,z_2_ones.n_rows-1,z_2_ones.n_cols-1) = z_2;
	//std::cout << "z_2_ones: " << z_2_ones.row(0) << endl;
	//pause();

	sigmoidGradient(z_2_ones,z_2_ones);

	//std::cout << "sigmoid_z_2: " << z_2_ones.row(0) << endl;
	//pause();
	
	//setup delta_2
	mat delta_2(delta_3.n_rows,Theta2.n_cols);
	delta_2 = (delta_3 * Theta2) % z_2_ones;

	//std::cout << "delta_2: " << delta_2.row(0) << endl;
	//pause();


	//setup and combine gradients
	mat gradient_2(delta_3.n_cols,a_2_ones.n_cols,fill::zeros);
	mat gradient_1(delta_2.n_cols-1,a_1.n_cols,fill::zeros);


	gradient_2 = (delta_3.t() * a_2_ones) / m;
	//std::cout << "gradient_2: " << gradient_2.row(0) << endl;
	//pause();


	gradient_1 = (delta_2.cols(1,delta_2.n_cols-1).t() * a_1) / m;
	//std::cout << "gradient_1: " << gradient_1.row(0) << endl;
	//pause();
	gradient_2.cols(1,gradient_2.n_cols-1) += Theta2.cols(1,Theta2.n_cols-1) * (lambda/m);
	gradient_1.cols(1,gradient_1.n_cols-1) += Theta1.cols(1,Theta1.n_cols-1) * (lambda/m);
	gradient_1.reshape(gradient_1.n_rows*gradient_1.n_cols,1);
	gradient_2.reshape(gradient_2.n_rows*gradient_2.n_cols,1);
	gradient = join_cols(gradient_1,gradient_2);
	//std::cout << "gradient: " << gradient.col(0) << endl;
	//pause();


	
	//std::cout << "leaving costfunction..." << endl;
}



void predict(const mat& Theta1,const mat& Theta2,const mat& inputdata, mat& predictions){
    mat x_holder(inputdata.n_rows,inputdata.n_cols+1,fill::ones);
    mat pre_h1(x_holder.n_rows,Theta1.n_rows);
    mat h1_ones(x_holder.n_rows,Theta1.n_rows+1,fill::ones);
    mat h2(x_holder.n_rows,Theta2.n_rows,fill::zeros);

	//std::cout << "inputdata: " << inputdata.n_rows << " " << inputdata.n_cols << endl;
	// std::cout << "x_holder: " << x_holder.n_rows << " " << x_holder.n_cols << endl;
    x_holder.submat(0,1,x_holder.n_rows-1,x_holder.n_cols-1) = inputdata;
    
    // std::cout << "Theta1: " << Theta1.n_rows << " " << Theta1.n_cols << endl;
    
    pre_h1 = x_holder * Theta1.t();

	// std::cout << "pre_h1: " << pre_h1.n_rows << " " << pre_h1.n_cols << endl;

    sigmoid(pre_h1,pre_h1);
    h1_ones.submat(0,1,h1_ones.n_rows-1,h1_ones.n_cols-1) = pre_h1;
    
    // std::cout << "h1_ones: " << h1_ones.n_rows << " " << h1_ones.n_cols << endl;
    // std::cout << "Theta2: " << Theta2.n_rows << " " << Theta2.n_cols << endl;
    
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

// void pauseJNS(){
// 	std::cout << std::endl << "Press ENTER to continue...";
// 	std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
// }
