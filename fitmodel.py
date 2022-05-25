import numpy as np
import pandas as pd
from astropy.io import fits
import matplotlib as mpl
import matplotlib.pyplot as plt
import emcee, corner,time
from multiprocessing import Pool
from scipy.optimize import minimize
from scipy.stats import norm
from DHmodels import *
from sklearn.metrics import r2_score, mean_squared_error

# Gaussian prior values for all parameters (defining mean and stddev)
pr = dict(vflat = [240., 20.],
          lag   = [15.,  10.],
          vR    = [0.,   10.],
          vz    = [0.,   10.],
          h0    = [2.,   1.],
          hmin  = [3.,   1.],
          hmax  = [5.,   1.],
          sigma = [1.,   1.])


def lnprior(pars,allpars,freepar,densmod):

    params = np.array(allpars)
    params[freepar] = pars
    
    ##### FlatSandwich priors ######
    if densmod == FlatSandwich_Density:
        vflat, lag, vR, vz, h0 = params
        if vflat<0 or h0<0: return -np.inf
        # Gaussian priors (pdfs)
        prior_mean = [pr['vflat'][0], pr['lag'][0],pr['vR'][0],pr['vz'][0],pr['h0'][0]]
        prior_std  = [pr['vflat'][1], pr['lag'][1],pr['vR'][1],pr['vz'][1],pr['h0'][1]]
        # Flat priors
        #pr = (0.001<h0<10) and (0<lag<200) and (200<vflat<280) and (-20<vz<20)
        #prior = 0 if pr else -np.inf
        # return prior
        #"""
    
    ##### GaussianSandwich priors ######
    elif densmod == GaussianSandwich_Density:
        vflat, lag, vR, vz, h0, sigma = params
        if vflat<0 or h0<0 or sigma<0: return -np.inf
        prior_mean = [pr['vflat'][0], pr['lag'][0],pr['vR'][0],pr['vz'][0],pr['h0'][0],pr['sigma'][0]]
        prior_std  = [pr['vflat'][1], pr['lag'][1],pr['vR'][1],pr['vz'][1],pr['h0'][1],pr['sigma'][1]]
        
    ##### ThickSandwich priors ######
    elif densmod == ThickSandwich_Density:
        vflat, lag, vR, vz, hmin, hmax = params
        if vflat<0 or hmin<0 or hmax<0 or hmin>hmax: return -np.inf
        prior_mean = [pr['vflat'][0], pr['lag'][0],pr['vR'][0],pr['vz'][0],pr['hmin'][0],pr['hmax'][0]]
        prior_std  = [pr['vflat'][1], pr['lag'][1],pr['vR'][1],pr['vz'][1],pr['hmin'][1],pr['hmax'][1]]
    
    ##### ConstantDensity priors ######
    elif densmod == Constant_Density:
        vflat, lag, vR, vz = params
        if vflat<0 or lag<0: return -np.inf
        prior_mean = [pr['vflat'][0], pr['lag'][0],pr['vR'][0],pr['vz'][0]]
        prior_std  = [pr['vflat'][1], pr['lag'][1],pr['vR'][1],pr['vz'][1]]
        
    else:
        raise NameError('WARNING: invalid density model in lnprior!')
    
    
    # Calculating Gaussian priors
    p_mean = np.array(prior_mean)[freepar]
    p_std  = np.array(prior_std)[freepar]
    prior  = 1
    for i in range(len(p_mean)):
        prior *= norm.pdf(pars[i],loc=p_mean[i],scale=p_std[i])
    
    return np.log(prior)


def lnlike(pars,data,allpars,freepar,densmod):

    params = np.array(allpars)
    params[freepar] = pars
    rho0 = 1E-05
    
    ##### FlatSandwich parameters ######
    if densmod == FlatSandwich_Density:
        vflat, lag, vR, vz, h0 = params
        denspars = (rho0,h0)
    
    ##### GaussianSandwich parameters ######
    elif densmod == GaussianSandwich_Density:
        vflat, lag, vR, vz, h0, sigma = params
        denspars = (rho0,h0,sigma)

    ##### ThickSandwich parameters ######
    elif densmod == ThickSandwich_Density:
        vflat, lag, vR, vz, hmin, hmax = params
        denspars = (rho0,hmin,hmax)

    ##### ConstantDensity parameters ######
    elif densmod == Constant_Density:
        vflat, lag, vR, vz = params
        denspars = (rho0,)
    
    else:
        raise NameError('WARNING: invalid density model in lnlike!')
    
    velopars = (vflat,lag,vR,vz)
    
    # Getting kinematical model
    mod = kinematic_model(data.lon,data.lat,velopars=velopars,densmodel=densmod,\
                          denspars=denspars,useC=True,nthreads=4)
    # Calculating residuals
    diff = np.nansum((mod.vlsr-data.vlsr)**2)
    return -diff


def lnprob(pars,data,allpars,freepar,densmod):
    lp = lnprior(pars,allpars,freepar,densmod)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(pars,data,allpars,freepar,densmod)



if __name__ == '__main__':

    
    ###########################################################################
    # User input section
    ###########################################################################
    doMCMC  = False
    dospace = False
    
    densmod = ThickSandwich_Density
    
    # FlatSandwich
    if densmod == FlatSandwich_Density:
        params = [230., 15., 0., -5., 1.]               # Initial guesses for all parameters
        labels = ["vflat","lag","vR","vz","h0"]         # Names of all parameters 
        free   = [True,True,False,True,True]            # Parameters to fit

    # GaussianSandwich 
    elif densmod == GaussianSandwich_Density:
        params = [230, 15, 0., -5., 5., 0.5]
        labels = ["vflat","lag","vR","vz","h0","sigma"]
        free   = [True,True,False,True,True,True]
    
    # ThickSandwich 
    elif densmod == ThickSandwich_Density:
        params = [230, 15, 0., -5., 1., 2.]
        labels = ["vflat","lag","vR","vz","hmin","hmax"]
        free   = [True,True,False,True,True,True]
    
    # ConstantDensity
    elif densmod == Constant_Density:
        params = [230., 15., 0., -5.]
        labels = ["vflat","lag","vR","vz"]
        free   = [True,True,False,True]
    ###########################################################################

    #Reading in sightlines
    ds = pd.read_table("data/sightlines_flag_4.txt", sep=' ', skipinitialspace=True)

    # Here we chose which ion we want to fit
    ion = 'CIV'
    di = ds[ds['ion']==ion]

    outdir = f'results/{ion}'
    if not os.path.exists(outdir): os.makedirs(outdir)
    denstype = densmod.__name__
    
    # We select only latitudes below 60 deg for the fit
    glon, glat, vl = di['Glon'].values, di['Glat'].values, di['weighted_v_LSR'].values
    glon[glon>180] -= 360
    m = (np.abs(glat)<60) #& (glon<0)
    # Just storing the data in a Sightline object for convenience
    data = Sightlines()
    data.add_sightlines(glon[m],glat[m],vl[m],None,None)
    print (f"Sightlines to fit: {len(data.lon)}")

    # Arrays with only parameters to fit
    p0     = np.array(params)[free]
    labs   = np.array(labels)[free]

    nll = lambda *args: -lnprob(*args)
    
    # This is just to minimize the likelihood in a classical way
    soln = minimize(nll, p0, args=(data,params,free,densmod),method='Nelder-Mead')
    pp = soln.x
    print ("Best-fit parameters: ", ["%.5f" % i for i in soln.x])
    
    
    if doMCMC:
        # Initializing chains and walkers
        ndim, nwalkers, nsteps = len(p0), 500, 500
        pos = p0 + 1e-1*np.random.randn(nwalkers,ndim)

        print ("\n Running MCMC...")
        with Pool() as pool:
            sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(data,params,free,densmod), pool=pool)
            start = time.time()
            res = sampler.run_mcmc(pos, nsteps, progress=True)
            multi_time = time.time()-start
            print("Computational time {0:.1f} minutes".format(multi_time/60.))

        # Saving samples
        fits.writeto(f"{outdir}/{ion}_{denstype}_samples.fits",np.float32(sampler.get_chain()),overwrite=True)

        # Plotting chains 
        fig, axes = plt.subplots(ndim, figsize=(10, 7), sharex=True)
        samples = sampler.get_chain()
        for i in range(ndim):
            ax = axes[i]
            ax.plot(samples[:, :, i], "k", alpha=0.3)
            ax.set_xlim(0, len(samples))
            ax.set_ylabel(labs[i])
            ax.yaxis.set_label_coords(-0.1, 0.5)

        axes[-1].set_xlabel("step number")
        fig.savefig(f"{outdir}/{ion}_{denstype}_chains.png")

        # Burn-in
        burnin = 200
        thin = 1
        samples = sampler.get_chain(discard=burnin, thin=thin, flat=True)

        print ("\n MCMC parameters:")
        pp = []
        for i in range(ndim):
            mcmc = np.percentile(samples[:, i], [15.865, 50, 84.135])
            q = np.diff(mcmc)
            txt = "%10s = %10.3f %+10.3f %+10.3f"%(labs[i],mcmc[1], -q[0], q[1])
            print (txt)
            pp.append(mcmc[1])
    
        # Autocorrelation function
        #tau = sampler.get_autocorr_time(quiet=True)
        #burnin = int(2*np.nanmax(tau))
        #thin = int(0.5*np.nanmin(tau))
        #print(tau,burnin,thin)

        # Save corner plot
        # Levels 
        levels = 1.0-np.exp(-0.5*np.arange(0.5, 2.1, 3.5) ** 2)
        #levels = 1.0-np.exp(-0.5*np.array([1., 2., 3.]) ** 2)

        fig = corner.corner(samples, truths=pp, labels=labs, show_titles=True, title_kwargs={"fontsize": lsize},\
                            truth_color='firebrick') #,fill_contours=True,levels=levels)
        fig.savefig(f"{outdir}/{ion}_{denstype}_corner.pdf",bbox_inches='tight')

    # Plot model vs data
    modpars = np.copy(params)
    modpars[free] = pp
    velopars = modpars[:4]
    denspars = modpars[4:]
    denspars = np.insert(denspars,0,1E-5)

    model = kinematic_model(data.lon,data.lat,velopars=velopars,densmodel=densmod,\
                            denspars=denspars,useC=True,nthreads=8,getSpectra=False)
    
    lab = ''
    for i,l in enumerate(labels): lab += '%s = %-10.2f'%(l,modpars[i])
    fig, ax = plot_datavsmodel(data,model,vrange=[-80,80],label=lab)
    fig.savefig(f"{outdir}/{ion}_{denstype}_comp.pdf",bbox_inches='tight')
    #fig, ax = plot_residuals(data,model)
    #fig.savefig(f"{outdir}/{ion}_residuals.pdf",bbox_inches='tight')



    # Calculate goodness of fit
    try:
        r_squared = r2_score(data.vlsr, model.vlsr, sample_weight=None, multioutput='uniform_average')
    except:
        r_squared = -999
    try:
        # Mean squared error
        MSE = mean_squared_error(data.vlsr, model.vlsr)# Mean squared error
        RMS = np.sign(MSE)*np.sqrt(abs(MSE))
    except:
        RMS = 999
    print ("")
    print (f" R-squared = {r_squared:12.4f}")
    print (f" MSE error = {MSE:12.4f}")
    print (f" RMS error = {RMS:12.4f}")
    
    
    
    # Parameter space for pairs of parameters
    if dospace:
        #p1min,p1max,p1lab = 0,5,"h0"
        p1min,p1max,p1lab = 180,280,"vflat"
        p2min,p2max,p2lab = 0,40,"lag"
        #p2min,p2max,p2lab = -20,20,"vz"
        #p2min,p2max,p2lab = 0,5,"h0"
        
        print ("\n Filling parameter space %s-%s"%(p1lab,p2lab))
        
        p1idx, p2idx = labels.index(p1lab),labels.index(p2lab)
        p1s, p2s = np.meshgrid(np.linspace(p1min,p1max,50),np.linspace(p2min,p2max,50))
        res = np.zeros_like(p1s)
    
        p = np.copy(params)
        f = np.ones_like(p).astype(bool)
        for i in range(p1s.shape[0]):
            for j in range(p1s.shape[1]):
                p[p1idx],p[p2idx] = p1s[i,j],p2s[i,j]
                res[i,j] = -lnlike(p,data,p,f,densmod)
    
        ext = [p1min,p1max,p2min,p2max]
        fig, ax = plt.subplots(figsize=(10,10),nrows=1, ncols=1)
        im = ax.imshow(res/1000,origin='lower',extent=ext,aspect='auto')
        ax.contour(res/1000,origin='lower',extent=ext,colors='k',levels=20)
        ax.set_xlabel(p1lab)
        ax.set_ylabel(p2lab)
        cb = plt.colorbar(im)
        fig.savefig(f"{outdir}/{ion}_spacepar_{denstype}_{p1lab}-{p2lab}.pdf",bbox_inches='tight')
