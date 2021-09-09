#include<iostream>
#include<cmath>
#include<iomanip>
#include<tuple>

#define deg2rad M_PI/180.
#define rad2deg 180./M_PI
#define kpc2cm 3.08567758e+21

extern "C" {
    

//////////////////////////////////////////////////////////////////////
// MODELS FOR GALACTIC ROTATION
//////////////////////////////////////////////////////////////////////
double rotationModel(double R, double z, double *pars) {
    // Simple flat rotation model with a vertical las
    // (R,z) in kpc, vflat in km/s, lag in km/s/kpc
    double vflat = pars[0], lag = pars[1];
    return vflat - lag*fabs(z);
}
 

//////////////////////////////////////////////////////////////////////
// GAS DENSITY DISTRIBUTIONS
// In all function below, lenghts are in kpc, densities in cm^-3
//////////////////////////////////////////////////////////////////////

double Constant_Density(double R, double z, double *pars) {
    // Density field is the same everywhere: dens(R,z) = const
    return pars[0];
}

double VerticalExponential_Density(double R, double z, double *pars) {
    // Density is constant in R, but drops exponentially in z: 
    //    dens(R,z) = rho0*exp(-|z|/z0)
    // where rho0=pars[0], z0=pars[1]
    return pars[0]*exp(-fabs(z)/pars[1]);
}

double RadialVerticalExponential_Density(double R, double z, double *pars) {
    // Density drops exponentially in both z and R: 
    //    dens(R,z) = rho0*exp(-R/R0)*exp(-|z|/z0)
    // where rho0=pars[0], R0=pars[1] and z0=pars[2]
    return pars[0]*exp(-R/pars[1])*exp(-fabs(z)/pars[2]);
}

double ThickSandwich_Density(double R, double z, double *pars) {
    // Density is constant in a thick layer at some height:
    //    dens(R,z) = rho0 if hmin<z<hmax else 0
    // where rho0=pars[0], hmin=pars[1], hmax=pars[2]
    if (fabs(z)>pars[1] && fabs(z)<pars[2]) return pars[0];
    else return 0;
}

double GaussianSandwich_Density(double R, double z, double *pars) {
    // Density is Gaussian in a layer at some height:
    //    dens(R,z) = rho0*exp(-(z-|z0|)**2/(2*sigma**2))
    // where rho0=pars[0], z0=pars[1], sigma=pars[2]
    return pars[0]*(exp(-(z-pars[1])*(z-pars[1])/(2*pars[2]*pars[2]))+
                    exp(-(z+pars[1])*(z+pars[1])/(2*pars[2]*pars[2])));
}



//////////////////////////////////////////////////////////////////////
// DEFINE A MODEL GIVEN DENSITY AND VELOCITY FIELD
//////////////////////////////////////////////////////////////////////

std::tuple<double, double> kinematic_model_single(double lon, double lat, double *velopars, 
                                                  const char *densmodel, double *denspars, 
                                                  double RSun=8.2, double VSun=240.) {
    
    // This function calculate the predicted column-density weighted VLSR
    // along a sightline (l,b) for a given density and velocity model.
    
    double D_min = 0.0001, D_max = 50;      // kpc
    double deltaD = 0.01;                   // kpc
    int nsteps = (D_max-D_min)/deltaD+1;
        
    // Now we point to the correct density function based on densmodel
    double (*densPtr) (double, double, double *);
    std::string dm = std::string(densmodel);
    if (dm=="Constant_Density") densPtr = &Constant_Density;
    else if (dm=="VerticalExponential_Density") densPtr = &VerticalExponential_Density;
    else if (dm=="RadialVerticalExponential_Density") densPtr = &RadialVerticalExponential_Density;
    else if (dm=="ThickSandwich_Density") densPtr =  &ThickSandwich_Density;
    else if (dm=="GaussianSandwich_Density") densPtr =  &GaussianSandwich_Density;
    else {
        std::cerr << "Unknown density function!!" << std::endl;
        std::terminate();
    }
    
    double num = 0; // Numerator of weighted average
    double den = 0; // Denominator of weighted average
    
    // Velocity field parameters
    double rotmodpars[2] = {velopars[0],velopars[1]};
    double vR = velopars[2];
    double vz = lat>0 ? velopars[3] : -velopars[3] ;
    
    // Loop over distances for a given sightline
    for (int i=0; i<nsteps; i++) {
        // Calculate position of a particle at (l,b,D)
        double d  = D_min+i*deltaD;
        double z  = d*sin(lat*deg2rad);
        double dp = d*cos(lat*deg2rad);
        double x  = RSun - dp*cos(lon*deg2rad);
        double y  = dp*sin(lon*deg2rad);
        double R  = sqrt(x*x+y*y);
        double th = atan2(y,x); 
        
        // Getting density and column density
        double rho = densPtr(R,z,denspars);
        double N = rho*deltaD*kpc2cm;
        
        // Getting velocity field
        double vtheta = rotationModel(R,z,rotmodpars);
        
        // Calculating predicted VLSR
        double vlsr = (RSun*sin(lon*deg2rad))*(-VSun/RSun+vtheta/R)*cos(lat*deg2rad) - 
                       vR*cos(lon*deg2rad+th)*cos(lat*deg2rad) + vz*sin(lat*deg2rad);
        
        num += (vlsr*N);
        den += N;
    }
    
    // Return the weighted average vlsr and the total logN
    return std::make_tuple(num/den, log10(den));
    
}


double* kinematic_model(double *lon, double *lat, int ns, double *velopars,\
                        const char *densmodel, double *denspars, 
                        double RSun=8.2, double VSun=240.,int nthreads=8) {

    double *retvalue = new double[2*ns];

#pragma omp parallel for num_threads(nthreads) 
    for (int s=0; s<ns; s++) {
        double vlsr_aver, logN;
        std::tie(vlsr_aver, logN) = kinematic_model_single(lon[s],lat[s],velopars,
                                       densmodel,denspars,RSun,VSun);
        retvalue[s] = vlsr_aver;
        retvalue[s+ns] = logN;
    }
    return retvalue;
}


}