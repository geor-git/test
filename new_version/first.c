#include <stdio.h>
#include <stdlib.h>
#include <complex.h>
#include <math.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_odeiv2.h>
#include "first.h"
#include <omp.h>
#include <time.h>
const int threads = 2;               // number of parallel threads 
                
const double g_pps = 1.0;            // pi-pi-sigma coupling constant
const double g_sss = 0.1;            // sigma_sigma_sigma coupling constant

                    
#define N 10                          // number of Gaissian-Lagguare nodes
#define M 31                          // number of grid points
const double L = 12.0;                // range of grid
const double node_0 = 2.3;           // initial point of grid
#define TOTAL (6*(N + M) + 2)      // number of vector components
#define NM (N+M)  
const double pi = 3.14159265358979323846;    // pi number
const double eps = 10E-16;                   // reqularizaion in Im function 
const double Lambda = 1.0;                   // UF cut-off

const double m0_sigma = 0.6 + Lambda * Lambda;                 // bare mass of sigma
const double m0_pi = 0.01 + Lambda * Lambda;                    // bare mass of pion

  

int main (void)
{
  
  FILE* solution;
  solution = fopen("data_cpd.txt", "w");
 // double mu = 1.0;
  
  gsl_odeiv2_system sys = {func, jac, TOTAL};

  gsl_odeiv2_driver * d = gsl_odeiv2_driver_alloc_y_new (&sys, gsl_odeiv2_step_rkf45, 1e-8, 1e-8, 1e-8);
         
  double t = 0.0, t1 = 2.5;
  
  
    double* root;
    root = roots();
    
    
  
  double y[TOTAL] = {0.0};    // INITIAL DATA
  
  for(int i = 0; i < N + M; ++i){
	  	  y[i] = root[i] - m0_sigma;
	  	  y[1 + i + 3*(N + M)] = root[i] - m0_pi;
	  }
	  
  
   y[3*(M+N)] = m0_sigma;
   y[6*(M+N) + 1] = m0_pi;
   
 
      /*
   double* rhs = total_rhs(y, 0.1);
   
   for(int k = 0; k < TOTAL; ++k){
	   printf("rhs: %lf\n", rhs[k]);
	   }
    */
  double start = omp_get_wtime();
  
  
  for(int i = 0; i < 3; i++)
    {
      double ti = i * t1 / 2.0;
      int status = gsl_odeiv2_driver_apply (d, &t, ti, y);

      if (status != GSL_SUCCESS)
        {
          printf ("error, return value=%d\n", status);
          break;
        }
}

    double end = omp_get_wtime();
    printf("Real TIme took: %f seconds to execute OpenMPI \n", end-start);

  for(int i = 0; i < TOTAL; ++i){
	  fprintf(solution, "%.16f\n", y[i]);
	  }
  fclose(solution);
  
   gsl_odeiv2_driver_free (d);
  
   free(root);
  return 0;
}



// *********************** Re part of kernel ***************************
double Re(double p, double l1, double l2, double l3){
	double complex dic_l1, dic_l2, x1_l1, x2_l1, x1_l2, x2_l2, z;
	dic_l1 =  csqrt(  (l3 - l1 + p)*(l3 - l1 + p) - 4 * l3 * p);
	dic_l2 =  csqrt(  (l3 - l2 + p)*(l3 - l2 + p) - 4 * l3 * p);
	x1_l1 = (  (l3 - l1 + p) + dic_l1)/ (2 * p);
	x2_l1 = (  (l3 - l1 + p) - dic_l1)/ (2 * p);
	x1_l2 = (  (l3 - l2 + p) + dic_l2)/ (2 * p);
	x2_l2 = (  (l3 - l2 + p) - dic_l2)/ (2 * p);
	if (abs(l1 - l2) < eps){
		if ( cabs(dic_l1) < eps){
			z =  (clog(1 - 1/x1_l1) - 1.0/(1  - x1_l1)) / (16 * p * pi * pi ) ;
			} else{
			  z  = (x1_l1/dic_l1 * clog(1 - 1.0/x1_l1) - x2_l1/dic_l1 * clog(1 - 1.0/x2_l1) ) / ( p * 16 * pi * pi);
		   }  
		
	} else {
			 z = ((1 - x1_l1)*(clog(1 - x1_l1) - 1) - (0 - x1_l1)*(clog(0 - x1_l1) - 1)  + 
		    (1 - x2_l1)*(clog(1 - x2_l1) - 1) - (0 - x2_l1)*(clog(0 - x2_l1) - 1) - 
		    ((1 - x1_l2)*(clog(1 - x1_l2) - 1) - (0 - x1_l2)*(clog(0 - x1_l2) - 1)  + 
		    (1 - x2_l2)*(clog(1 - x2_l2) - 1) - (0 - x2_l2)*(clog(0 - x2_l2) - 1) ))/(l1 - l2) / (16 * pi * pi) ;
			  }
    return creal(z);
}


// ********** Heaviside - step function *********************
	
double Heaviside(double x){
    if (x > 0.0){
        return 1.0;
    } else{
        return 0.0;
        }	
}



// ********** Im part of kernel ******************************
double Im(double p, double l1, double l2, double l3){
	double complex dic_l1, dic_l2, b_l1, b_l2, z;
	b_l1 = (l3 - l1 + p) / p;
	b_l2 = (l3 - l2 + p) / p;
	dic_l1 = b_l1 * b_l1 - 4 * l3 / p;
	dic_l2 = b_l2 * b_l2 - 4 * l3 / p;
	 
	if (abs(l1 - l2) < eps){
		if (cabs(dic_l1) < eps){
			z = 0;
		} else{
			z  = b_l1 *  Heaviside(dic_l1) * Heaviside(b_l1) * Heaviside(2 - b_l1) / csqrt(dic_l1)  / (16 * pi *p);
			} 
		 } else {
			 z  = - ( csqrt(dic_l1) * Heaviside(dic_l1) * Heaviside(b_l1) * Heaviside(2 - b_l1) -
		         csqrt(dic_l2) * Heaviside(dic_l2) * Heaviside(b_l2) * Heaviside(2 - b_l2)) / ( l1 - l2) / (16 * pi);
			  }
    return creal(z);
}


//////////////////////////////////////////////////////////////////////
// /////////////// Gaussain - Lagguare quadrature///////////////////// 
//////////////////////////////////////////////////////////////////////



// output: array of weights 
double* weight() {
    FILE *fp;
    char* filename = "root_w.txt";
    
    
    double* arr_w = (double*)malloc(N * sizeof(double));
	if (!arr_w){
		printf("No memory. \n");
		exit(1);
	} 
	
 
    fp = fopen(filename, "r");
    if (!fp){
        printf("Could not open file %s",filename);
        exit(EXIT_FAILURE);
       }   
	
    for (int i = 0; i < N; ++i){	    
		if(fscanf(fp, "%lf", &arr_w[i]) == EOF){
		exit(EXIT_FAILURE);
		}	    
	}
 
    
    fclose(fp);
    return arr_w;
}






// output: array of roots

double* roots() {
    FILE *fp;
    char* filename = "root_x.txt";
    
    
    double* arr_r = (double*)malloc((N + M) * sizeof(double));
	if (!arr_r){
		printf("No memory. \n");
		exit(1);
	} 
	
	
	
	    
    fp = fopen(filename, "r");
    if (!fp){
        printf("Could not open file %s",filename);
       }    
		   
	for (int i = 0; i < N; ++i){	    
		if(fscanf(fp, "%lf", &arr_r[i]) == EOF){
			exit(EXIT_FAILURE);
		}	    
	}
	
	for(int i = 0; i < M; ++i){
		arr_r[i + N] =  (double)i * L /(M - 1) + node_0;  // L/2*cos(i*pi/(M-1)) + node0 + L/2;
		}	    
 
    fclose(fp);
    return arr_r; 
}





// **************************** SIGMA RHS **************************

double* sigma_rhs(const double y[], double t, double (*f)(double, double, double, double)){
	int i;
	double* rh_re = (double*)malloc((N + M) * sizeof(double));
	double* root;
	double* w;
    root = roots();
	w = weight();
	
	
	double rho_root[N] = {0};
	for (i = 0; i < N; ++i)
	{rho_root[i] =  y[2*(N + M) + i];}
	
	
	double  rho_pi_root[N] = {0};
	for (i = 0; i < N; ++i)
	{rho_pi_root[i] =  y[5*(N + M) + i];}
	
	double m_sigma = y[3*(N + M)];
	double m_pi    = y[6*(N + M) + 1];
	double m_total = m_sigma + m_pi + 2 * sqrt(m_sigma * m_pi);
	double m = (m_pi < m_sigma) ? m_pi : m_sigma;                   // min value of m_pi and m_sigma
	  
    
    double sum1[NM]  = {0};
    double sum3[NM]   = {0};
    double sum12 [NM] = {0};
    double sum13 [NM] = {0};
    double sum0 [NM]  = {0};
       
	 
    #pragma omp parallel for private ( i ) num_threads(threads)
	for (i = 0; i < N + M; ++i)
	{  
		for(int i1 = 0; i1 < N; ++i1)
		{
			for(int i2 = 0; i2 < N; ++i2 )
			{    double ai1i2 = w[i1] * w[i2] * exp(root[i1]+root[i2]) * rho_root[i1] * rho_root[i2] *  Heaviside(root[i1] - 4*m) * Heaviside(root[i2] - 4*m);
				 double bi1i2 = w[i1] * w[i2] * exp(root[i1]+root[i2]) * rho_pi_root[i1] * rho_pi_root[i2] *  Heaviside(root[i1] - m_total) * Heaviside(root[i2] - m_total);
				 
                 sum3[i] += g_sss * g_sss * (*f)(root[i], root[i1], root[i2], m_sigma) * ai1i2 + 
                            3 * g_pps * g_pps * (*f)(root[i], root[i1], root[i2], m_pi) * bi1i2;
                            
                 sum1[i] += g_sss * g_sss * (*f)(root[i], m_sigma, root[i1], root[i2]) * ai1i2 + 
                            3 * g_pps * g_pps * (*f)(root[i], m_pi, root[i1], root[i2]) * bi1i2;
                   }
         }
      
        for(int i3 = 0; i3 < N; ++i3)
        {   double ai3 = w[i3] * exp(root[i3]) * rho_root[i3] * Heaviside(root[i3] - 4 * m);
            double bi3 = w[i3] * exp(root[i3]) * rho_pi_root[i3] * Heaviside(root[i3] - m_total);
            
            sum12[i] += g_sss * g_sss * (*f)(root[i], m_sigma, m_sigma, root[i3]) * ai3 + 3 * g_pps * g_pps * (*f)(root[i], m_pi, m_pi, root[i3]) * bi3;
            sum13[i] += g_sss * g_sss * (*f)(root[i], m_sigma, root[i3], m_sigma) * ai3 + 3 * g_pps * g_pps * (*f)(root[i], m_pi, root[i3], m_pi) * bi3;
        }
 
       
       for(int i1 = 0; i1 < N; ++i1)
       {
			for(int i2 = 0; i2 < N; ++i2)
			{
				for(int i3 = 0; i3 < N; ++i3)
				{   double ai1i2i3 = (*f)(root[i], root[i1], root[i2], root[i3]) * w[i1] * w[i2] * w[i3] * exp(root[i1]+root[i2]+root[i3]);
					
					sum0[i] += (g_sss * g_sss * rho_root[i1] * rho_root[i2] * rho_root[i3] * Heaviside(root[i1] - 4*m) * Heaviside(root[i2] - 4*m) * Heaviside(root[i3] - 4*m) + 
					           3 * g_pps * g_pps * rho_pi_root[i1] * rho_pi_root[i2] * rho_pi_root[i3] * Heaviside(root[i1] - m_total) * Heaviside(root[i2] - m_total) * Heaviside(root[i3] - m_total)) * ai1i2i3; 
			    }
			}	
       } 
       
     rh_re[i] = Lambda * Lambda * exp(-2 * t) * (g_sss * g_sss * (*f)(root[i], m_sigma, m_sigma, m_sigma) + 3 * g_pps * g_pps * (*f)(root[i], m_pi, m_pi, m_pi) + sum3[i] + 2*sum1[i] + 1*sum12[i]  + 2*sum13[i] + sum0[i]);
         }
    
  //  for(i = 0; i < N + M; ++i)  {
	//	rh_re[i] = Lambda * Lambda * exp(-2 * t) * (g_sss * g_sss * (*f)(root[i], m_sigma, m_sigma, m_sigma) + 3 * g_pps * g_pps * (*f)(root[i], m_pi, m_pi, m_pi) + sum3[i] + 2*sum1[i] + 1*sum12[i]  + 2*sum13[i] + sum0[i]);
     //}    
     
     
     /*   
   free(sum1);
   free(sum3);
   free(sum12);
   free(sum13);
   free(sum0);  
      */
       
   free(root);
   free(w);
 //  free(rho_root);
  // free(rho_pi_root);
   return rh_re;
}


// ************************** PION RHS ********************************

double* pi_rhs(const double y[], double t, double (*f)(double, double, double, double)){
	int i;
	double* rh_re = (double*)malloc((N + M) * sizeof(double));
	double* root;
	double* w;
    root = roots();
	w = weight();
	
	double rho_root[N] ;
	for (i = 0; i < N; ++i)
	{rho_root[i] =  y[2*(N + M) + i];}
	
	double rho_pi_root[N];
	for (i = 0; i < N; ++i)
	{rho_pi_root[i] =  y[5*(N + M) + i];}
	
	double m_sigma = y[3*(N + M)];
	double m_pi    = y[6*(N + M) + 1];
	double m_total = m_sigma + m_pi + 2 * sqrt(m_sigma * m_pi);
	double m = (m_pi < m_sigma) ? m_pi : m_sigma;                   // min value of m_pi and m_sigma
	  
    
    double sum1[NM]  = {0};
    double sum3 [NM] = {0};
    double sum12 [NM] = {0};
    double sum13 [NM] = {0};
    double sum0  [NM] = {0};
       
	
    #pragma omp parallel for private ( i ) num_threads(threads)
	for (i = 0; i < N + M; ++i)
	{    
	 		
		for(int i1 = 0; i1 < N; ++i1)
		{
			for(int i2 = 0; i2 < N; ++i2 )
			{    double ai1i2 = w[i1] * w[i2] * exp(root[i1]+root[i2]) * rho_root[i1] * rho_root[i2] *  Heaviside(root[i1] - 4*m) * Heaviside(root[i2] - 4*m);
				 double bi1i2 = w[i1] * w[i2] * exp(root[i1]+root[i2]) * rho_pi_root[i1] * rho_pi_root[i2] *  Heaviside(root[i1] - m_total) * Heaviside(root[i2] - m_total);
				 
                 sum3[i] += (*f)(root[i], root[i1], root[i2], m_sigma) * bi1i2 + 
                            (*f)(root[i], root[i1], root[i2], m_pi) * ai1i2;
                            
                 double ci1i2 = w[i1] * w[i2] * exp(root[i1]+root[i2]) * rho_root[i1] * rho_pi_root[i2] *  Heaviside(root[i1] - 4*m) * Heaviside(root[i2] - m_total);
			          
                 sum1[i] += ((*f)(root[i], m_sigma, root[i1], root[i2]) + (*f)(root[i], m_pi, root[i2], root[i1]) )* ci1i2; 
                            
                   }
         }
      
        for(int i3 = 0; i3 < N; ++i3)
        {   double ai3 = w[i3] * exp(root[i3]) * rho_root[i3] * Heaviside(root[i3] - 4 * m);
            double bi3 = w[i3] * exp(root[i3]) * rho_pi_root[i3] * Heaviside(root[i3] - m_total);
            
            sum12[i] += (*f)(root[i], m_sigma, m_sigma, root[i3]) * bi3 + (*f)(root[i], m_pi, m_pi, root[i3]) * ai3;
            sum13[i] += (*f)(root[i], m_sigma, root[i3], m_pi) * ai3 + (*f)(root[i], m_pi, root[i3], m_sigma) * bi3;
        }
 
       
       for(int i1 = 0; i1 < N; ++i1)
       {
			for(int i2 = 0; i2 < N; ++i2)
			{
				for(int i3 = 0; i3 < N; ++i3)
				{   double ai1i2i3 = (*f)(root[i], root[i1], root[i2], root[i3]) * w[i1] * w[i2] * w[i3] * exp(root[i1]+root[i2]+root[i3]);
					
					sum0[i] += (rho_root[i1] * rho_root[i2] * rho_pi_root[i3] * Heaviside(root[i1] - 4*m) * Heaviside(root[i2] - 4*m) * Heaviside(root[i3] - m_total) + 
					            rho_pi_root[i1] * rho_pi_root[i2] * rho_root[i3] * Heaviside(root[i1] - m_total) * Heaviside(root[i2] - m_total) * Heaviside(root[i3] - 4*m)) * ai1i2i3; 
			    }
			}	
       }
       
    rh_re[i] = g_pps * g_pps * Lambda * Lambda * exp(-2 * t) * ((*f)(root[i], m_sigma, m_sigma, m_pi) + (*f)(root[i], m_pi, m_pi, m_sigma) + sum3[i] + 2*sum1[i] + 1*sum12[i]  + 2*sum13[i] + sum0[i]);
    
      } 
      
   // for(i = 0; i < N + M; ++i)  {
	//	rh_re[i] = g_pps * g_pps * Lambda * Lambda * exp(-2 * t) * ((*f)(root[i], m_sigma, m_sigma, m_pi) + (*f)(root[i], m_pi, m_pi, m_sigma) + sum3[i] + 2*sum1[i] + 1*sum12[i]  + 2*sum13[i] + sum0[i]);
    // }    
    
     
   free(root);
   free(w);
  return rh_re;
}	



		
// ********************** MASSS RHS *************************************

double sigma_rhs_mass(const double y[], double t){
	int i, i1;
	double rh_mass;
	double* root;
	double* w;
    root = roots();
	w = weight();
	
	double rho_root [N];
	for (i = 0; i < N; ++i)
	{rho_root[i] =  y[2*(N + M) + i];}
	
	double rho_pi_root [N];
	for (i = 0; i < N; ++i)
	{rho_pi_root[i] =  y[5*(N + M) + i];}
	
	double m_sigma = y[3*(N + M)];
	double m_pi    = y[6*(N + M) + 1];
	double m_total = m_sigma + m_pi + 2 * sqrt(m_sigma * m_pi);
	double m = (m_pi < m_sigma) ? m_pi : m_sigma;                   // min value of m_pi and m_sigma
	  
    double sum3 = 0, sum1 = 0, sum12 = 0, sum13 = 0, sum0 = 0;
    
    
    #pragma omp parallel for private ( i1 ) num_threads(threads)
   	for(i1 = 0; i1 < N; ++i1)
		{
	    for(int i2 = 0; i2 < N; ++i2 ){
			double ai1i2 = w[i1] * w[i2] * exp(root[i1]+root[i2]) * rho_root[i1] * rho_root[i2] *  Heaviside(root[i1] - 4*m) * Heaviside(root[i2] - 4*m);
		    double bi1i2 = w[i1] * w[i2] * exp(root[i1]+root[i2]) * rho_pi_root[i1] * rho_pi_root[i2] *  Heaviside(root[i1] - m_total) * Heaviside(root[i2] - m_total);
				 
                 sum3 += g_sss * g_sss * Re(m_sigma, root[i1], root[i2], m_sigma) * ai1i2 + 
                            3 * g_pps * g_pps * Re(m_sigma, root[i1], root[i2], m_pi) * bi1i2;
                            
                 sum1 += g_sss * g_sss * Re(m_sigma, m_sigma, root[i1], root[i2]) * ai1i2 + 
                            3 * g_pps * g_pps * Re(m_sigma, m_pi, root[i1], root[i2]) * bi1i2;
                   }
         double ai1 = w[i1] * exp(root[i1]) * rho_root[i1] * Heaviside(root[i1] - 4 * m);
         double bi1 = w[i1] * exp(root[i1]) * rho_pi_root[i1] * Heaviside(root[i1] - m_total);
             
         sum12 += g_sss * g_sss * Re(m_sigma, m_sigma, m_sigma, root[i1]) * ai1 + 3 * g_pps * g_pps * Re(m_sigma, m_pi, m_pi, root[i1]) * bi1;
         sum13 += g_sss * g_sss * Re(m_sigma, m_sigma, root[i1], m_sigma) * ai1 + 3 * g_pps * g_pps * Re(m_sigma, m_pi, root[i1], m_pi) * bi1;
                  
          
      //   }
      
      //  for(int i3 = 0; i3 < N; ++i3){
	  //	double ai3 = w[i3] * exp(root[i3]) * rho_root[i3] * Heaviside(root[i3] - 4 * m);
      //      double bi3 = w[i3] * exp(root[i3]) * rho_pi_root[i3] * Heaviside(root[i3] - m_total);
             
        //    sum12 += g_sss * g_sss * Re(m_sigma, m_sigma, m_sigma, root[i3]) * ai3 + 3 * g_pps * g_pps * Re(m_sigma, m_pi, m_pi, root[i3]) * bi3;
         //   sum13 += g_sss * g_sss * Re(m_sigma, m_sigma, root[i3], m_sigma) * ai3 + 3 * g_pps * g_pps * Re(m_sigma, m_pi, root[i3], m_pi) * bi3;
       // }
 
             
   	  // for(i1 = 0; i1 < N; ++i1){
			for(int i2 = 0; i2 < N; ++i2){
				for(int i3 = 0; i3 < N; ++i3){
					double ai1i2i3 = Re(m_sigma, root[i1], root[i2], root[i3]) * w[i1] * w[i2] * w[i3] * exp(root[i1]+root[i2]+root[i3]);
					
					sum0 += (g_sss * g_sss * rho_root[i1] * rho_root[i2] * rho_root[i3] * Heaviside(root[i1] - 4*m) * Heaviside(root[i2] - 4*m) * Heaviside(root[i3] - 4*m) + 
					           3 * g_pps * g_pps * rho_pi_root[i1] * rho_pi_root[i2] * rho_pi_root[i3] * Heaviside(root[i1] - m_total) * Heaviside(root[i2] - m_total) * Heaviside(root[i3] - m_total)) * ai1i2i3; 
			    }
			}	
       } 
    
       
   rh_mass = -Lambda * Lambda * exp(-2 * t) * (g_sss * g_sss * Re(m_sigma, m_sigma, m_sigma, m_sigma) + 3 * g_pps * g_pps * Re(m_sigma, m_pi, m_pi, m_pi) + sum3 + 2*sum1 + 1*sum12 + 2*sum13 + sum0);
  
   free(root);
   free(w);
     
  return rh_mass;
}

// ********************** PION RHS MASS ********************************

double pi_rhs_mass(const double y[], double t){
	int i1, i;
	double rh_pi_mass;
	double* root;
	double* w;
    root = roots();
	w = weight();
	
	double rho_root[N] ;
	for (i = 0; i < N; ++i)
	{rho_root[i] =  y[2*(N + M) + i];}
	
	double rho_pi_root [N];
	for (i = 0; i < N; ++i)
	{rho_pi_root[i] =  y[5*(N + M) + i];}
	
	double m_sigma = y[3*(N + M)];
	double m_pi    = y[6*(N + M) + 1];
	double m_total = m_sigma + m_pi + 2 * sqrt(m_sigma * m_pi);
	double m = (m_pi < m_sigma) ? m_pi : m_sigma;                   // min value of m_pi and m_sigma
	 
	double sum3 = 0, sum1 = 0, sum12 = 0, sum13 = 0, sum0 = 0; 
	
	#pragma omp parallel for private ( i1 ) num_threads(threads)
    for(i1 = 0; i1 < N; ++i1){
		for(int i2 = 0; i2 < N; ++i2 ){
			double ai1i2 = w[i1] * w[i2] * exp(root[i1]+root[i2]) * rho_root[i1] * rho_root[i2] *  Heaviside(root[i1] - 4*m) * Heaviside(root[i2] - 4*m);
		    double bi1i2 = w[i1] * w[i2] * exp(root[i1]+root[i2]) * rho_pi_root[i1] * rho_pi_root[i2] *  Heaviside(root[i1] - m_total) * Heaviside(root[i2] - m_total);
				 
            sum3 += Re(m_pi, root[i1], root[i2], m_sigma) * bi1i2 + Re(m_pi, root[i1], root[i2], m_pi) * ai1i2;
                            
            double ci1i2 = w[i1] * w[i2] * exp(root[i1]+root[i2]) * rho_root[i1] * rho_pi_root[i2] *  Heaviside(root[i1] - 4*m) * Heaviside(root[i2] - m_total);
			          
            sum1 += (Re(m_pi, m_sigma, root[i1], root[i2]) + Re(m_pi, m_pi, root[i2], root[i1]) )* ci1i2; 
                   }
         }
      
        for(int i3 = 0; i3 < N; ++i3){
			double ai3 = w[i3] * exp(root[i3]) * rho_root[i3] * Heaviside(root[i3] - 4 * m);
            double bi3 = w[i3] * exp(root[i3]) * rho_pi_root[i3] * Heaviside(root[i3] - m_total);
            
            sum12 += Re(m_pi, m_sigma, m_sigma, root[i3]) * bi3 + Re(m_pi, m_pi, m_pi, root[i3]) * ai3;
            sum13 += Re(m_pi, m_sigma, root[i3], m_pi) * ai3 + Re(m_pi, m_pi, root[i3], m_sigma) * bi3;
        }
 
       #pragma omp parallel for private ( i1 ) num_threads(threads)
       for(i1 = 0; i1 < N; ++i1){
		   for(int i2 = 0; i2 < N; ++i2){
			   for(int i3 = 0; i3 < N; ++i3){
				   double ai1i2i3 = Re(m_pi, root[i1], root[i2], root[i3]) * w[i1] * w[i2] * w[i3] * exp(root[i1]+root[i2]+root[i3]);
					
				   sum0 += (rho_root[i1] * rho_root[i2] * rho_pi_root[i3] * Heaviside(root[i1] - 4*m) * Heaviside(root[i2] - 4*m) * Heaviside(root[i3] - m_total) + 
					            rho_pi_root[i1] * rho_pi_root[i2] * rho_root[i3] * Heaviside(root[i1] - m_total) * Heaviside(root[i2] - m_total) * Heaviside(root[i3] - 4*m)) * ai1i2i3; 
			                    }
			         }	
            }
      
   rh_pi_mass = -g_pps * g_pps * Lambda * Lambda * exp(-2 * t) * (Re(m_pi, m_sigma, m_sigma, m_pi) + Re(m_pi, m_pi, m_pi, m_sigma) + sum3 + 2*sum1 + 1*sum12 + 2*sum13 + sum0);
        
  
   free(root);
   free(w);
     
   return rh_pi_mass;
}	

		
		
// ************************** RHO RHS ********************************
// ************************** SIGMA RHS RHO **************************

double* sigma_rhs_rho(const double y[], double t){
	int i;
	
    double *rh_rho  = (double *)malloc((N+M) * sizeof(double));
          
    double re_Gamma [NM];
    for(i = 0; i < M + N; ++i){
        re_Gamma[i] = y[i];
    }  
        
    double im_Gamma[NM] ;
    for(i = 0; i < M + N; ++i){
        im_Gamma[i] = y[N+M+i];
     }
     
    double* rh_im;
    rh_im =  sigma_rhs(y, t, Im);
    
    
    double* rh_re;
    rh_re =  sigma_rhs(y, t, Re);
    
    #pragma omp parallel for private ( i ) num_threads(threads) 
    for(i = 0; i < N + M; ++i){
        rh_rho[i] = (1.0/pi * (-im_Gamma[i]*im_Gamma[i] + (re_Gamma[i]-exp(-2*t))*(re_Gamma[i]-exp(-2*t)))/ pow(im_Gamma[i]*im_Gamma[i] + (re_Gamma[i]- exp(-2*t))*(re_Gamma[i]- exp(-2*t)),2) * rh_im[i] - 2.0/pi * (im_Gamma[i]*(re_Gamma[i]-exp(-2*t)) ) / pow(im_Gamma[i]*im_Gamma[i] + (re_Gamma[i]- exp(-2*t))*(re_Gamma[i]- exp(-2*t)) ,2) * (rh_re[i]+2*exp(-2*t)));
    }
    
  
    free(rh_im);
    free(rh_re);
   
    return rh_rho;

}

// ************************** PI RHS RHO **************************
double* pi_rhs_rho(const double y[], double t){
	int i;
	
    double* rh_pi_rho  = (double *)malloc((N+M) * sizeof(double));
          
    double re_Gamma[NM];
    for(i = 0; i < M + N; ++i){
        re_Gamma[i] = y[i + 3 * (M + N) + 1];
    }  
        
    double im_Gamma[NM];
    for(i = 0; i < M + N; ++i){
        im_Gamma[i] = y[4 * (N + M) + 1 + i];
     }
     
    double* rh_im;
    rh_im =  pi_rhs(y, t, Im);
    
    
    double* rh_re;
    rh_re =  pi_rhs(y, t, Re);
    
    #pragma omp parallel for private ( i ) num_threads(threads) 
    for(i = 0; i < N + M; ++i){
        rh_pi_rho[i] = (1.0/pi * (-im_Gamma[i]*im_Gamma[i] + (re_Gamma[i]-exp(-2*t))*(re_Gamma[i]-exp(-2*t)))/ pow(im_Gamma[i]*im_Gamma[i] + (re_Gamma[i]- exp(-2*t))*(re_Gamma[i]- exp(-2*t)),2) * rh_im[i] - 2.0/pi * (im_Gamma[i]*(re_Gamma[i]-exp(-2*t)) ) / pow(im_Gamma[i]*im_Gamma[i] + (re_Gamma[i]- exp(-2*t))*(re_Gamma[i]- exp(-2*t)) ,2) * (rh_re[i]+2*exp(-2*t)));
    }
    
    
    free(rh_im);
    free(rh_re);
   
    return rh_pi_rho;
}		


		
// ************************ TOTAL RHS **********************************


double* total_rhs(const double y[], double t){
	double* rh_total  = (double *)malloc((6*(N + M) + 2) * sizeof(double));
	int i;
	
	double* re_sigma;
	re_sigma = sigma_rhs(y, t, Re);
	
	double* im_sigma;
	im_sigma = sigma_rhs(y, t, Im);
	
	double* rho_sigma;
	rho_sigma =  sigma_rhs_rho(y, t);
	
	double m_sigma = sigma_rhs_mass(y, t);
	
	double* re_pi;
	re_pi = pi_rhs(y, t, Re);
	
	double* im_pi;
	im_pi = pi_rhs(y, t, Im);
	
	double* rho_pi;
	rho_pi =  pi_rhs_rho(y, t);
	
	double m_pi = pi_rhs_mass(y, t);
	
	#pragma omp parallel for private ( i ) num_threads(threads) 
	for(i = 0; i < N + M; ++i){
		rh_total[i] = -re_sigma[i];
		rh_total[i + N + M] = -im_sigma[i];
		rh_total[i + 2 * (M + N)] = rho_sigma[i];
		rh_total[i + 3 * (N + M) + 1] = -re_pi[i];
		rh_total[i + 4 * (N + M) + 1] = -im_sigma[i];
		rh_total[i + 5 * (N + M) + 1] = rho_pi[i];
		}
		
	rh_total[3*(M + N)] = -m_sigma;
	rh_total[6*(M + N) + 1] = -m_pi;
		
	free(re_sigma);
	free(im_sigma);
	free(rho_sigma);
	
	free(re_pi);
	free(im_pi);
	free(rho_pi);
		
    return rh_total;
		
}	
	


int jac (double t, const double y[], double *dfdy, double dfdt[], void *params)
{
  (void)(t); /* avoid unused parameter warning */
  double mu = *(double *)params;
  gsl_matrix_view dfdy_mat
    = gsl_matrix_view_array (dfdy, TOTAL, TOTAL);
  gsl_matrix * m = &dfdy_mat.matrix;
  for(int i = 0; i < TOTAL; ++i){
	  dfdt[i] = 0.0;
	  for(int j = 0; j < TOTAL; ++j){
		  gsl_matrix_set (m, i, j, 0.0*mu);
			  }
		  }	  
  return GSL_SUCCESS;
}


   
int func (double t, const double y[], double f[], void *params)
{
  //(void)(t); /* avoid unused parameter warning */
  (void)(params);
  //double mu = *(double *)params;
  double* rhs_prt;
  
  rhs_prt = total_rhs(y, t);
     
  for(int i = 0; i < TOTAL; ++i)
  {
	  f[i] = rhs_prt[i];
	  }
  //printf("Value: %f\n ", f[2]);
  
  free(rhs_prt);
 
  return GSL_SUCCESS;
}
	
