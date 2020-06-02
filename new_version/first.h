double Re(double p, double l1, double l2, double l3);
double Heaviside(double x);
double Im(double p, double l1, double l2, double l3);
double* weight();
double* roots();
double* sigma_rhs_re(const double y[], double t);
double* sigma_rhs_im(const double y[], double t);
double  sigma_rhs_mass(const double y[], double t);
double* pi_rhs_re(const double y[], double t);
double* pi_rhs_im(const double y[], double t);
double  pi_rhs_mass(const double y[], double t);
double* sigma_rhs_rho(const double y[], double t);
double* pi_rhs_rho(const double y[], double t);
double* total_rhs(const double y[], double t);
int jac (double t, const double y[], double *dfdy,  double dfdt[], void *params);

int func (double t, const double y[], double f[], void *params);

