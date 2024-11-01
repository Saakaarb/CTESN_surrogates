import numpy as np
import math
import scipy
from setup_robertsons_ode import robertsons_ode,generate_data
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
        print("max eig:",max_eig)
        # scaling spectral radius of matrix
        self.W=self.W*(self.spectral_radius/max_eig)


    def setup_reservoir_ODE(self,soln_t,soln_y):
    
        # setup initial condition
        
        #setup ODE run
        
        self.fit_solution_term(soln_t,soln_y)
        
    def reservoir_deriv(self,t,x,solution_fit,t_scale):
    
        dx_dt=np.tanh(self.W@x+self.W_in@solution_fit(t/t_scale))

        return dx_dt
    

    # solution term requires a fit to be used with ODE integrator
    def fit_solution_term(self,soln_t,soln_y):
    
        solution_fit=scipy.interpolate.CubicSpline(soln_t,soln_y.T)
        
        return solution_fit
    

        #self.plot_solve(soln_t,self.fit_objects[0](soln_t),soln_y[0,:],"spline_fit_1.png")    
        #self.plot_solve(soln_t,self.fit_objects[1](soln_t),soln_y[1,:],"spline_fit_2.png")
        #self.plot_solve(soln_t,self.fit_objects[2](soln_t),soln_y[2,:],"spline_fit_3.png")

    def initialize_reservoir(self):

        #self.r0=np.zeros(self.Nx)
        self.r0=np.random.uniform(size=self.Nx)

    def solve_reservoir_ode(self,soln_t,solution_fit,t_scale):
    
        r_t=scipy.integrate.odeint(self.reservoir_deriv,self.r0,soln_t*t_scale,tfirst=True,args=(solution_fit,t_scale))
        # add bias term to r_t
        nts=r_t.shape[0]
        

        return r_t
    
    def fit_r_interpolant(self,soln_t,r_t):
    
        self.r_interp_object=scipy.interpolate.CubicSpline(soln_t,r_t,axis=0)

    def query_r_interpolant(self,soln_t_i):

        r_t_i=self.r_interp_object(soln_t_i)
        return r_t_i

    def fit_W_out(self,r_t,soln_y):

        W_out_T=np.linalg.lstsq(r_t,soln_y.T)[0]
        
        W_out=W_out_T.T
      
        
      
        return W_out
  
    
    def fit_W_out_interpolant(self,all_W_outs,all_params):
    
        collapsed_params=np.concatenate([np.expand_dims(all_params[0],axis=1),np.expand_dims(all_params[1],axis=1),np.expand_dims(all_params[2],axis=1)],axis=1)
    
        W_outs_matrix=np.expand_dims(all_W_outs[0],axis=0)
        #print(W_outs_matrix)

        for i_W in range(1,len(all_W_outs)):
        
            W_outs_matrix=np.concatenate([W_outs_matrix,np.expand_dims(all_W_outs[i_W],axis=0)],axis=0)
    
        #print(collapsed_params.shape,W_outs_matrix.shape)
        #input("check")
        self.W_out_interp=scipy.interpolate.RBFInterpolator(collapsed_params,W_outs_matrix,neighbors=4)
    
        
        

    def query_W_out_interpolant(self,query_params):
    
        W_out=self.W_out_interp(query_params)
    
        return W_out
    
    def query_new_param(self,query_params,query_times):
    
        t1=time.time()
        r_t_query=self.query_r_interpolant(query_times)
        t2=time.time()
        W_out_query=self.query_W_out_interpolant(query_params)
        t3=time.time()
    
        return W_out_query@r_t_query.T
  
    def test_W_out(self,soln_t,W_out,r_t):

        x_all=W_out@r_t.T
        
        self.plot_solve(soln_t,x_all[0,:],filename="CTESN_1.png")
        self.plot_solve(soln_t,x_all[1,:],filename="CTESN_2.png")
        self.plot_solve(soln_t,x_all[2,:],filename="CTESN_3.png")

    def plot_r_fit(self,query_times,r_soln):
    
        query_r=self.query_r_interpolant(query_times)
        

        plt.xscale('log')
        plt.plot ( query_times, query_r[:,21], linewidth = 6 )
        plt.plot ( query_times, r_soln[:,21], linewidth = 3 )
        plt.grid ( True )
        
        plt.show()
        
    def plot_solve(self,t,y,y_orig=None,ylim=None,title=None,ylabel='<---  y  --->',filename='plot.png'):


          plt.xscale('log')
          plt.rcParams["font.weight"] = "bold"
          plt.rcParams["axes.labelweight"] = "bold"
          
          plt.plot ( t, y, linewidth = 6,label='Surrogate Prediction' )
          
          if y_orig is not None:
          
            plt.plot ( t, y_orig, linewidth = 3 ,label='True Solution')
          if ylim is not None:
            plt.ylim(ylim)
          if title is not None:
          
            plt.title(title)
          plt.grid ( True )
          plt.xlim([10**(-4),10**(4)])
          plt.xlabel ( '<---  t  --->' )
          plt.ylabel ( ylabel )
          plt.legend()
          plt.savefig(filename)
          plt.close()
          
          

if __name__=="__main__":

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
         
    r1=0.04
    r3=10000
    r2=3*10**(7)
    
    #scaling factors
    scale_r1=r1
    scale_r2=r2
    scale_r3=r3
    t_scale=1
    
    #generate parameter space
    num_samples=50
        
    all_vals,all_solns,scaling_values =generate_data(t0,y0,tstop,r1,r2,r3,scale_r1,scale_r2,scale_r3,num_samples,data_generated)
    if mode=="train":    

        soln_t,soln_y=all_solns[0]

        # input dimension
        Nu=soln_y.shape[0]
        
        # take the first solution and solve the reservoir using the CTESN method
        
        # create object
        ctesn=CTESN(Nx,density,spectral_radius,Nu,alpha) 
        #initialize matrices of reservoir
        ctesn.init_matrices()
        # fit the solution term to time queryable object
        
        solution_fit=ctesn.fit_solution_term(soln_t,soln_y)
        # initial condition of reservoir ODE
        ctesn.initialize_reservoir()
        # solve reservoir ODE
        
        soln_r_t=ctesn.solve_reservoir_ode(soln_t,solution_fit,t_scale)
        
        W_out=ctesn.fit_W_out(soln_r_t,soln_y)
        
        ctesn.test_W_out(soln_t,W_out,soln_r_t)
        
        ctesn.fit_r_interpolant(soln_t,soln_r_t)
        
        # for all the rest stiff ODE solves,
        #1. interpolate the r to the timesteps for those solutions
        #2. then fit the W_out for them
        
        all_W_outs=[W_out]
        
        for i in range(1,len(all_solns)):
        
        
            curr_t,curr_y=all_solns[i]
            r_t_i=ctesn.query_r_interpolant(curr_t)    
            all_W_outs.append(ctesn.fit_W_out(r_t_i,curr_y))
        
        # get RBF interpolants for W_outs
        ctesn.fit_W_out_interpolant(all_W_outs,all_vals)
 
        
        with open('trained_CTESN.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
                pickle.dump(ctesn, f)
      
    else:
        with open('trained_CTESN.pkl', 'rb') as f:  # Python 3: open(..., 'wb')
            ctesn=pickle.load(f)   

  
    # get new parameters to test:
    
    
    test_params=[[0.95,1.05,0.95],[0.9,1.1,0.9],[1.1,0.9,1.1],[1.03,0.99,1.04],[0.85,1.12,1.12]]
    #test_params=[[r1/scale_r1,r2/scale_r2,r3/scale_r3]]
    
    errs_accum_y2=[]
    
    for i,test_param in enumerate(test_params):
    
        query_params=np.expand_dims(np.array([test_param[0],test_param[1],test_param[2]]),axis=0)
        # solve ODE at this parameter
        #------------
        
        ode_inst=robertsons_ode(t0,y0,tstop,test_param[0]*scale_r1,test_param[1]*scale_r2,test_param[2]*scale_r3)

        ode_inst.solve_robertson_ivp()
            
        soln_t_query,soln_y_query=ode_inst.export_solution()
        #------------
        
        y_query=ctesn.query_new_param(query_params,soln_t_query/t_scale)
        
        
        y_query=np.squeeze((y_query)*scaling_values[:,None])
        
        
        ctesn.plot_solve(soln_t_query,y_query[0,:],soln_y_query[0,:],ylim=[0,1.1],ylabel='<---  y1  --->',title="Parameter set "+str(i+1),filename="y1_"+str(i)+".png")
        ctesn.plot_solve(soln_t_query,y_query[1,:],soln_y_query[1,:],ylabel='<---  y2  --->',title="Parameter set "+str(i+1),filename="y2_"+str(i)+".png")
        ctesn.plot_solve(soln_t_query,y_query[2,:],soln_y_query[2,:],ylabel='<---  y3  --->',ylim=[0,1.1],title="Parameter set "+str(i+1),filename="y3_"+str(i)+".png")
        
        
        perc_err_y1=np.mean(np.abs(np.divide(y_query[0,:]-soln_y_query[0,:],1)))
        perc_err_y2=np.mean(np.abs(np.divide(y_query[1,:]-soln_y_query[1,:],1)))
        perc_err_y3=np.mean(np.abs(np.divide(y_query[2,:]-soln_y_query[2,:],1)))
    
        print("Abs error in y1:",perc_err_y1)
        print("Abs error in y2:",perc_err_y2)
        print("Abs error in y3:",perc_err_y3)
        errs_accum_y2.append(perc_err_y2)
    
    print("Done")
    
