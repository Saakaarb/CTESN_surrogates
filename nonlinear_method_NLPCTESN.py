import numpy as np
import math
import scipy
from setup_robertsons_ode import robertsons_ode, generate_data
from matplotlib import pyplot as plt
import random
import pickle
import time
import timeit


class CTESN(object):

    def __init__(self,Nx,density,spectral_radius,Nu,alpha):

        self.Nx=Nx
        self.Nu=Nu
        self.density=density
        self.spectral_radius=spectral_radius
        self.alpha=alpha
        
    # initialize W_in, W
    def init_matrices(self):

        # initialize matrices
        self.W_in=np.random.random((self.Nx,self.Nu))*2-1.0
        
        self.W=scipy.sparse.random(self.Nx,self.Nx,density=self.density)
        
        self.W=self.W.toarray()*2-1.0

        # set spectral radius of W

        eig_W=np.linalg.eig(self.W)[0]
        
        max_eig=0
        
        for i in range(len(eig_W)):
        
            
            if np.abs(eig_W[i]) > max_eig:
            
                max_eig=np.abs(eig_W[i])
        
        # scaling spectral radius of matrix
        self.W=self.W*(self.spectral_radius/max_eig)


    def setup_reservoir_ODE(self,soln_t,soln_y):
        
        #setup ODE run
        
        self.fit_solution_term(soln_t,soln_y)
        
    def reservoir_deriv(self,t,x,solution_fit,t_scale):
    
        dx_dt=np.tanh(self.W@x+self.W_in@solution_fit(t/t_scale))

        return dx_dt    

    # solution term requires a fit to be used with ODE integrator
    def fit_solution_term(self,soln_t,soln_y):
    
        solution_fit=scipy.interpolate.CubicSpline(soln_t,soln_y.T)
        
        return solution_fit
    
    def initialize_reservoir(self):

        self.r0=np.random.uniform(size=self.Nx)

    def solve_reservoir_ode(self,soln_t,solution_fit,t_scale):
    
        r_t=scipy.integrate.odeint(self.reservoir_deriv,self.r0,soln_t*t_scale,tfirst=True,args=(solution_fit,t_scale))

        nts=r_t.shape[0]

        return r_t
  
    def get_RBF_weights(self,r_t,y_t):
        
        curr_RBF=scipy.interpolate.RBFInterpolator(r_t,y_t.T)
  
        weights=curr_RBF._coeffs
  
        return weights,curr_RBF
  
    def fit_RBF(self,x,y):
    
        return scipy.interpolate.RBFInterpolator(x,y,neighbors=4)

    def query_RBF_w_weights(self,RBF,weights,r_t):
    
        RBF._coeffs=weights
        return RBF(r_t)
    
    def query_new_param(self,query_params,param_RBF,weight_RBF,r_t):
    
        weights=param_RBF(np.expand_dims(np.array(query_params),axis=0))
        
        x_pred=self.query_RBF_w_weights(weight_RBF,weights,r_t)
    
        return x_pred
  

    def plot_r_fit(self,query_times,r_soln):
    
        query_r=self.query_r_interpolant(query_times)
        

        plt.xscale('log')
        plt.plot ( query_times, query_r[:,21], linewidth = 6 )
        plt.plot ( query_times, r_soln[:,21], linewidth = 3 )
        plt.grid ( True )
        
        plt.show()
        
    def plot_solve(self,t,y,y_orig=None,ylim=None,title=None,filename='plot.png'):


          plt.xscale('log')
          plt.rcParams["font.weight"] = "bold"
          plt.rcParams["axes.labelweight"] = "bold"
          plt.plot ( t, y, linewidth = 6,label='Surrogate Prediction' )
          
          if y_orig is not None:
          
            plt.plot ( t, y_orig, linewidth = 3,label='True Solution' )
          if ylim is not None:
            plt.ylim(ylim)
          if title is not None:
          
            plt.title(title)
          plt.grid ( True )
          plt.xlim([10**(-4),10**(4)])
          plt.xlabel ( '<---  t  --->' )
          plt.ylabel ( '<---  y  --->' )
          plt.legend()
          plt.savefig(filename)
          plt.close()
          
          

if __name__=="__main__":

    # mode: "train" or "predict"
    mode="train"

    data_generated=False
    np.random.seed(0)
    # Hyper parameters of problem
    Nx=50

    #control sparsity of matrix
    density=0.01
    # should be <1 or close to 1
    spectral_radius=0.1
    alpha=0.5
        
    t0=0
    y0=[1.0,0,0.0]
    tstop=10000
    
    t_scale=1
     
    r1=0.04
    r3=10000
    r2=3*10**(7)
    
    #scaling factors
    scale_r1=r1
    scale_r2=r2
    scale_r3=r3
    
    #generate parameter space
    num_samples=50
       
    all_vals,all_solns,scaling_values =generate_data(t0,y0,tstop,r1,r2,r3,scale_r1,scale_r2,scale_r3,num_samples,data_generated)
    if mode=="train":    
        _,soln_y=all_solns[0]

        # input dimension
        Nu=soln_y.shape[0]
        
        # create object
        ctesn=CTESN(Nx,density,spectral_radius,Nu,alpha) 
        #initialize matrices of reservoir
        ctesn.init_matrices()
        # initial condition of reservoir ODE
        ctesn.initialize_reservoir()
        
        all_params=[]
        all_RBF_weights=[]
        for i,soln in enumerate(all_solns):
        
            soln_t,soln_y=all_solns[i]
        
            # take the first solution and solve the reservoir using the CTESN method
            # reservoir ODE is solved only once
            if i==0:
        
                soln_t_main=soln_t
                
                solution_fit=ctesn.fit_solution_term(soln_t,soln_y)
        
                # solve reservoir ODE
                soln_r_t=ctesn.solve_reservoir_ode(soln_t,solution_fit,t_scale)
                curr_RBF_weights,curr_RBF=ctesn.get_RBF_weights(soln_r_t,soln_y)
            # fit the solution term to time queryable object
            
            else:
            
                solution_fit=ctesn.fit_solution_term(soln_t,soln_y)
                soln_y_interp=solution_fit(soln_t_main).T
                curr_RBF_weights,curr_RBF=ctesn.get_RBF_weights(soln_r_t,soln_y_interp)
            all_params.append([all_vals[0][i],all_vals[1][i],all_vals[2][i]])

            
            all_RBF_weights.append(curr_RBF_weights)
        
        
        # assemble RBF weights and params to matrices, and fit another RBF
        
        for i in range(len(all_RBF_weights)):
        
            if i==0:
            
                all_params_mat=np.expand_dims(np.array(all_params[i]),axis=0)
                all_weights_mat=np.expand_dims(all_RBF_weights[i],axis=0)
        
            else:
                all_params_mat=np.concatenate([all_params_mat,np.expand_dims(np.array(all_params[i]),axis=0)],axis=0)
                all_weights_mat=np.concatenate([all_weights_mat,np.expand_dims(all_RBF_weights[i],axis=0)],axis=0)
        
        
        
        param_to_weight_RBF=ctesn.fit_RBF(all_params_mat,all_weights_mat)
        weight_RBF=curr_RBF
        
        
        with open('trained_CTESN.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
                pickle.dump(ctesn, f)
      
    else:
        with open('trained_CTESN.pkl', 'rb') as f:  # Python 3: open(..., 'wb')
            ctesn=pickle.load(f)   

  
    # get new parameters to test:
 
    test_params=[[0.95,1.05,0.95],[0.9,1.1,0.9],[1.1,0.9,1.1],[1.03,0.99,1.04],[0.85,1.12,1.12]]
    
    errs_accum_y2=[]
    
    for i,test_param in enumerate(test_params):
        print("Predicting case ",str(i+1))
        #query_params=np.expand_dims(np.array([test_param[0],test_param[1],test_param[2]]),axis=0)
        query_param=test_param
        # solve ODE at this parameter
        #------------
        
        
        ode_inst=robertsons_ode(t0,y0,tstop,test_param[0]*scale_r1,test_param[1]*scale_r2,test_param[2]*scale_r3)

        ode_inst.solve_robertson_ivp()
            
        soln_t_query,soln_y_query=ode_inst.export_solution()
        #------------
        t1=timeit.default_timer()
        y_query=ctesn.query_new_param(query_param,param_to_weight_RBF,weight_RBF,soln_r_t).T
        
        
        solution_fit=ctesn.fit_solution_term(soln_t_main,y_query)
        y_query=solution_fit(soln_t_query).T
        
        y_query=np.squeeze(y_query*scaling_values[:,None])
        t2=timeit.default_timer()
        print("Time taken for pred:",t2-t1)
 
        ctesn.plot_solve(soln_t_query,y_query[0,:],soln_y_query[0,:],ylim=[0,1.1],title="Parameter set "+str(i+1),filename="y1_"+str(i)+".png")
        ctesn.plot_solve(soln_t_query,y_query[1,:],soln_y_query[1,:],title="Parameter set "+str(i+1),filename="y2_"+str(i)+".png")
        ctesn.plot_solve(soln_t_query,y_query[2,:],soln_y_query[2,:],ylim=[0,1.1],title="Parameter set "+str(i+1),filename="y3_"+str(i)+".png")
        
        
        perc_err_y1=np.mean(np.abs(np.divide(y_query[0,:]-soln_y_query[0,:],1)))
        perc_err_y2=np.mean(np.abs(np.divide(y_query[1,:]-soln_y_query[1,:],1)))
        perc_err_y3=np.mean(np.abs(np.divide(y_query[2,:]-soln_y_query[2,:],1)))
    
        print("Abs error in y1:",perc_err_y1)
        print("Abs error in y2:",perc_err_y2)
        print("Abs error in y3:",perc_err_y3)
        errs_accum_y2.append(perc_err_y2)
    
    print("Done")
    
