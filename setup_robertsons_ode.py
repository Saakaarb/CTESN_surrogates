import numpy as np
import scipy
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import pickle

class robertsons_ode(object):

    def __init__(self,t0,y0,tstop,r1,r2,r3):

        self.t0=t0
        self.y0=y0

        self.tstop=tstop
        self.p1=r1
        self.p2=r2
        self.p3=r3


    def robertson_deriv (self,t, y ):


        y1 = y[0]
        y2 = y[1]
        y3 = y[2]

        dydt = np.zeros(3)

        dydt[0] = - self.p1 * y1 + self.p3 * y2 * y3
        dydt[1] =   self.p1 * y1 - self.p3 * y2 * y3 - self.p2 * y2 * y2
        dydt[2] =                                    + self.p2 * y2 * y2  

        return dydt


    def solve_robertson_ivp (self):



          tspan = np.array ( [ self.t0, self.tstop ] )
          
          self.sol = solve_ivp ( self.robertson_deriv, tspan, self.y0, method = 'LSODA' )

          
    def robertson_conserved(self,t,y):
    

        h = np.sum ( y, axis = 0 )

        return h

    def export_solution(self):
    
    
        return self.sol.t,self.sol.y

    def plot_solve(self):

          plt.xscale('log')
          plt.plot ( self.sol.t, self.sol.y[0,:], linewidth = 3 )
          filename="species_1.png"
          plt.grid ( True )
          plt.xlabel ( '<---  t  --->' )
          plt.ylabel ( '<---  y1  --->' )
          plt.savefig(filename)
          plt.close()
          
          plt.xscale('log')
          plt.plot ( self.sol.t, self.sol.y[1,:], linewidth = 3 )
          filename="species_2.png"
          plt.grid ( True )
          plt.xlabel ( '<---  t  --->' )
          plt.ylabel ( '<---  y2  --->' )
          plt.savefig(filename)
          plt.close()
          
          plt.xscale('log')
          plt.plot ( self.sol.t, self.sol.y[2,:], linewidth = 3 )
          filename="species_3.png"
          plt.grid ( True )
          plt.xlabel ( '<---  t  --->' )
          plt.ylabel ( '<---  y3  --->' )
          plt.savefig(filename)
          plt.close()
          
          
          #h = robertson_conserved ( sol.t, sol.y )

def generate_data(t0,y0,tstop,r1,r2,r3,scale_r1,scale_r2,scale_r3,num_samples,data_generated):

    if not data_generated:
    
        min_factor=0.8
        max_factor=1.2
        
        axis_r1=np.random.uniform(size=num_samples)
        axis_r2=np.random.uniform(size=num_samples)
        axis_r3=np.random.uniform(size=num_samples)
    
        
        vals_r1=(min_factor*r1+(max_factor-min_factor)*axis_r1*r1)/scale_r1
        vals_r2=(min_factor*r2+(max_factor-min_factor)*axis_r2*r2)/scale_r2
        vals_r3=(min_factor*r3+(max_factor-min_factor)*axis_r3*r3)/scale_r3

        all_vals=[vals_r1,vals_r2,vals_r3]

        all_solns=[]

        for i_sample in range(num_samples):
        
        
            print("Generating data sample:",i_sample+1)
            ode_inst=robertsons_ode(t0,y0,tstop,vals_r1[i_sample]*scale_r1,vals_r2[i_sample]*scale_r2,vals_r3[i_sample]*scale_r3)

            ode_inst.solve_robertson_ivp()
            
            soln_t,soln_y=ode_inst.export_solution()

            # scaling of y
            if i_sample==0:
                scaling_values=np.max(soln_y,axis=1)

            soln_y=soln_y/scaling_values[:,None]
            
            #scaling of t
            
            soln_t=soln_t
            
            all_solns.append([soln_t,soln_y])

        with open('generated_data.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
            pickle.dump([all_vals,all_solns,scaling_values], f)

    # use previously generated data
    else:
    
        with open('generated_data.pkl','rb') as f:  # Python 3: open(..., 'rb')
            all_vals,all_solns,scaling_values= pickle.load(f)    

    return  all_vals,all_solns,scaling_values
