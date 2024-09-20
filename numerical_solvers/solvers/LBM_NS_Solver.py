import taichi as ti
import taichi.math as tm
import numpy as np

from numerical_solvers.solvers.LBM_SolverBase import LBM_SolverBase
from numerical_solvers.solvers.SpectralTurbulenceGenerator import SpectralTurbulenceGenerator
from numerical_solvers.solvers.GaussianTurbulenceGenerator import get_gaussian_noise

# Fluid solver based on lattice boltzmann method using taichi language
# Inspired by: https://github.com/hietwll/LBM_Taichi


@ti.data_oriented
class LBM_NS_Solver(LBM_SolverBase):
    def __init__(self, name, domain_size, kin_visc, bulk_visc, turbulenceGenerator: SpectralTurbulenceGenerator):
        self.omega_bulk = 1.0 / (3.0 * bulk_visc + 0.5) 
        super().__init__(name, domain_size, kin_visc, turbulenceGenerator)
            
            
    def init(self, np_gray_image): 
        self.rho.from_numpy(np_gray_image)
        self.vel.fill(0)
        # u_spec, v_spec = self.spectralTurbulenceGenerator.generate_turbulence(0)     
        # force_numpy = np.stack((u_spec, v_spec), axis=-1)  # Shape becomes (128, 128, 2)
        # self.Force.from_numpy(force_numpy)
        
        self.init_gaussian_force_field(0*1E-3, 0, 1)
        self.init_fields()
                   
    def solve(self, iterations):
        for iteration in range(iterations):                
            self.stream()
            self.update_macro_var()
            # self.collide_srt()
            self.collide_cm()
             
            # u_turb, v_turb = self.turbulenceGenerator.generate_turbulence(self.iterations_counter)     
            # turb_numpy = np.stack((u_turb, v_turb), axis=-1)  # Shape becomes (128, 128, 2)
            # self.Force.from_numpy(turb_numpy)
            
            # self.init_gaussian_force_field(1E-2, 0, 1)
            self.apply_bb()
            # self.apply_nee_bc()
            self.iterations_counter = self.iterations_counter +1
            
        if self.iterations_counter % 10 == 0:
            print(f"iterations: {self.iterations_counter}")
                            

               
        # periodic wip
        # for j in range(0, self.ny):
        # # right bc to left bc
        #     self.f_new[0, j][1] = self.f_new[self.nx-1, j][1] # east
        #     self.f_new[0, j][5] = self.f_new[self.nx-1, j][5] # north-east
        #     self.f_new[0, j][8] = self.f_new[self.nx-1, j][8] # south-east
        # # left bc to right bc
        #     self.f_new[self.nx-1, j][3] = self.f_new[0, j][3] # west
        #     self.f_new[self.nx-1, j][6] = self.f_new[0, j][6] # north-west
        #     self.f_new[self.nx-1, j][7] = self.f_new[0, j][7] # south-west

    @ti.kernel
    def collide_srt(self):
        for i, j in ti.ndrange((1, self.nx - 2), (1, self.ny - 2)):
            for k in ti.static(range(9)):
                feq = self.f_eq(i, j)
                self.f_new[i, j][k] = (1 - self.omega_kin) * self.f[i, j][k] + feq[k] * self.omega_kin
    

        
    @ti.kernel
    def collide_cm(self):
        for i, j in ti.ndrange((1, self.nx - 2), (1, self.ny - 2)):
            # magnitude = 1E-12
            # noise =magnitude*get_gaussian_noise(0,1)
            
            # noise = 0.1*get_gaussian_noise(0,1)
            # self.vel[i, j] = ti.Vector([noise[0], noise[1]]) # run as ade
            
            
            #=== THIS IS AUTOMATICALLY GENERATED CODE ===
            ux = self.vel[i, j][0]
            uy = self.vel[i, j][1]
            
            uxuy = self.vel[i, j][0]*self.vel[i, j][1]
            ux2 = self.vel[i, j][0]*self.vel[i, j][0]
            uy2 = self.vel[i, j][1]*self.vel[i, j][1]
            
            
            #raw moments from density-probability functions
            self.f_new[i,j][0] = self.f[i,j][0] + self.f[i,j][1] + self.f[i,j][2] + self.f[i,j][3] + self.f[i,j][4] + self.f[i,j][5] + self.f[i,j][6] + self.f[i,j][7] + self.f[i,j][8]
            self.f_new[i,j][1] = self.f[i,j][1] - self.f[i,j][3] + self.f[i,j][5] - self.f[i,j][6] - self.f[i,j][7] + self.f[i,j][8]
            self.f_new[i,j][2] = self.f[i,j][2] - self.f[i,j][4] + self.f[i,j][5] + self.f[i,j][6] - self.f[i,j][7] - self.f[i,j][8]
            self.f_new[i,j][3] = self.f[i,j][1] + self.f[i,j][3] + self.f[i,j][5] + self.f[i,j][6] + self.f[i,j][7] + self.f[i,j][8]
            self.f_new[i,j][4] = self.f[i,j][2] + self.f[i,j][4] + self.f[i,j][5] + self.f[i,j][6] + self.f[i,j][7] + self.f[i,j][8]
            self.f_new[i,j][5] = self.f[i,j][5] - self.f[i,j][6] + self.f[i,j][7] - self.f[i,j][8]
            self.f_new[i,j][6] = self.f[i,j][5] + self.f[i,j][6] - self.f[i,j][7] - self.f[i,j][8]
            self.f_new[i,j][7] = self.f[i,j][5] - self.f[i,j][6] - self.f[i,j][7] + self.f[i,j][8]
            self.f_new[i,j][8] = self.f[i,j][5] + self.f[i,j][6] + self.f[i,j][7] + self.f[i,j][8]

            # central moments from raw moments
            self.f[i,j][0] = self.f_new[i,j][0]
            self.f[i,j][1] = -self.f_new[i,j][0]*ux + self.f_new[i,j][1]
            self.f[i,j][2] = -self.f_new[i,j][0]*uy + self.f_new[i,j][2]
            self.f[i,j][3] = self.f_new[i,j][0]*ux2 - 2.*self.f_new[i,j][1]*ux + self.f_new[i,j][3]
            self.f[i,j][4] = self.f_new[i,j][0]*uy2 - 2.*self.f_new[i,j][2]*uy + self.f_new[i,j][4]
            self.f[i,j][5] = self.f_new[i,j][0]*uxuy - self.f_new[i,j][1]*uy - self.f_new[i,j][2]*ux + self.f_new[i,j][5]
            self.f[i,j][6] = -self.f_new[i,j][0]*ux2*uy + 2.*self.f_new[i,j][1]*uxuy + self.f_new[i,j][2]*ux2 - self.f_new[i,j][3]*uy - 2.*self.f_new[i,j][5]*ux + self.f_new[i,j][6]
            self.f[i,j][7] = -self.f_new[i,j][0]*ux*uy2 + self.f_new[i,j][1]*uy2 + 2.*self.f_new[i,j][2]*uxuy - self.f_new[i,j][4]*ux - 2.*self.f_new[i,j][5]*uy + self.f_new[i,j][7]
            self.f[i,j][8] = self.f_new[i,j][0]*ux2*uy2 - 2.*self.f_new[i,j][1]*ux*uy2 - 2.*self.f_new[i,j][2]*ux2*uy + self.f_new[i,j][3]*uy2 + self.f_new[i,j][4]*ux2 + 4.*self.f_new[i,j][5]*uxuy - 2.*self.f_new[i,j][6]*uy - 2.*self.f_new[i,j][7]*ux + self.f_new[i,j][8]

            m000 = self.f[i,j][0]
            
            #collide with bulk relaxation in trt flavour
            self.f_new[i,j][0] = m000
            self.f_new[i,j][1] = 1/2.*self.Force[i,j][0] # self.Force[None][0]
            self.f_new[i,j][2] = 1/2.*self.Force[i,j][1]
            self.f_new[i,j][3] = 1/6.*m000*(self.omega_bulk - self.omega_kin) + 1/6.*m000*(self.omega_bulk + self.omega_kin) - 1/2.*self.f[i,j][3]*(self.omega_bulk + self.omega_kin - 2.) - 1/2.*self.f[i,j][4]*(self.omega_bulk - self.omega_kin)
            self.f_new[i,j][4] = 1/6.*m000*(self.omega_bulk - self.omega_kin) + 1/6.*m000*(self.omega_bulk + self.omega_kin) - 1/2.*self.f[i,j][3]*(self.omega_bulk - self.omega_kin) - 1/2.*self.f[i,j][4]*(self.omega_bulk + self.omega_kin - 2.)
            self.f_new[i,j][5] = -self.f[i,j][5]*(self.omega_kin - 1.)
            self.f_new[i,j][6] = 1/6.*self.Force[i,j][1]
            self.f_new[i,j][7] = 1/6.*self.Force[i,j][0]
            # self.f_new[i,j][8] = 1/9.*m000 # this is not nice
            self.f_new[i,j][8] = self.f[i,j][8]*(1.- self.omega_kin) +  self.omega_kin*m000/9.
            
            #SRT
            # self.f_new[i,j][0] = m000
            # self.f_new[i,j][1] = self.f[i,j][1]*(1.- self.omega_kin) + (1 - self.omega_kin/2.)*self.Force[i,j][0]
            # self.f_new[i,j][2] = self.f[i,j][2]*(1.- self.omega_kin) + (1 - self.omega_kin/2.)*self.Force[i,j][1]
            # self.f_new[i,j][3] = self.f[i,j][3]*(1.- self.omega_kin) + self.omega_kin*m000/3.
            # self.f_new[i,j][4] = self.f[i,j][4]*(1.- self.omega_kin) + self.omega_kin*m000/3.
            # self.f_new[i,j][5] = self.f[i,j][5]*(1.- self.omega_kin)
            # self.f_new[i,j][6] = self.f[i,j][6]*(1.- self.omega_kin) + (1 - self.omega_kin/2.)*self.Force[i,j][1]/3.
            # self.f_new[i,j][7] = self.f[i,j][7]*(1.- self.omega_kin) + (1 - self.omega_kin/2.)*self.Force[i,j][0]/3.
            # self.f_new[i,j][8] = self.f[i,j][8]*(1.- self.omega_kin) +  self.omega_kin*m000/9.
            
            
            #SRT without force
            # self.f_new[i,j][0] = m000
            # self.f_new[i,j][1] = self.f[i,j][1]*(1.- self.omega_kin) 
            # self.f_new[i,j][2] = self.f[i,j][2]*(1.- self.omega_kin)
            # self.f_new[i,j][3] = self.f[i,j][3]*(1.- self.omega_kin) + self.omega_kin*m000/3.
            # self.f_new[i,j][4] = self.f[i,j][4]*(1.- self.omega_kin) + self.omega_kin*m000/3.
            # self.f_new[i,j][5] = self.f[i,j][5]*(1.- self.omega_kin)
            # self.f_new[i,j][6] = self.f[i,j][6]*(1.- self.omega_kin) 
            # self.f_new[i,j][7] = self.f[i,j][7]*(1.- self.omega_kin) 
            # self.f_new[i,j][8] = self.f[i,j][8]*(1.- self.omega_kin) +  self.omega_kin*m000/9.
            
            #back to raw moments
            self.f[i,j][0] = self.f_new[i,j][0]
            self.f[i,j][1] = self.f_new[i,j][0]*ux + self.f_new[i,j][1]
            self.f[i,j][2] = self.f_new[i,j][0]*uy + self.f_new[i,j][2]
            self.f[i,j][3] = self.f_new[i,j][0]*ux2 + 2.*self.f_new[i,j][1]*ux + self.f_new[i,j][3]
            self.f[i,j][4] = self.f_new[i,j][0]*uy2 + 2.*self.f_new[i,j][2]*uy + self.f_new[i,j][4]
            self.f[i,j][5] = self.f_new[i,j][0]*uxuy + self.f_new[i,j][1]*uy + self.f_new[i,j][2]*ux + self.f_new[i,j][5]
            self.f[i,j][6] = self.f_new[i,j][0]*ux2*uy + 2.*self.f_new[i,j][1]*uxuy + self.f_new[i,j][2]*ux2 + self.f_new[i,j][3]*uy + 2.*self.f_new[i,j][5]*ux + self.f_new[i,j][6]
            self.f[i,j][7] = self.f_new[i,j][0]*ux*uy2 + self.f_new[i,j][1]*uy2 + 2.*self.f_new[i,j][2]*uxuy + self.f_new[i,j][4]*ux + 2.*self.f_new[i,j][5]*uy + self.f_new[i,j][7]
            self.f[i,j][8] = self.f_new[i,j][0]*ux2*uy2 + 2.*self.f_new[i,j][1]*ux*uy2 + 2.*self.f_new[i,j][2]*ux2*uy + self.f_new[i,j][3]*uy2 + self.f_new[i,j][4]*ux2 + 4.*self.f_new[i,j][5]*uxuy + 2.*self.f_new[i,j][6]*uy + 2.*self.f_new[i,j][7]*ux + self.f_new[i,j][8]

            #back to density-probability functions
            self.f_new[i,j][0] = self.f[i,j][0] - self.f[i,j][3] - self.f[i,j][4] + self.f[i,j][8]
            self.f_new[i,j][1] = 1/2.*self.f[i,j][1] + 1/2.*self.f[i,j][3] - 1/2.*self.f[i,j][7] - 1/2.*self.f[i,j][8]
            self.f_new[i,j][2] = 1/2.*self.f[i,j][2] + 1/2.*self.f[i,j][4] - 1/2.*self.f[i,j][6] - 1/2.*self.f[i,j][8]
            self.f_new[i,j][3] = -1/2.*self.f[i,j][1] + 1/2.*self.f[i,j][3] + 1/2.*self.f[i,j][7] - 1/2.*self.f[i,j][8]
            self.f_new[i,j][4] = -1/2.*self.f[i,j][2] + 1/2.*self.f[i,j][4] + 1/2.*self.f[i,j][6] - 1/2.*self.f[i,j][8]
            self.f_new[i,j][5] = 1/4.*self.f[i,j][5] + 1/4.*self.f[i,j][6] + 1/4.*self.f[i,j][7] + 1/4.*self.f[i,j][8]
            self.f_new[i,j][6] = -1/4.*self.f[i,j][5] + 1/4.*self.f[i,j][6] - 1/4.*self.f[i,j][7] + 1/4.*self.f[i,j][8]
            self.f_new[i,j][7] = 1/4.*self.f[i,j][5] - 1/4.*self.f[i,j][6] - 1/4.*self.f[i,j][7] + 1/4.*self.f[i,j][8]
            self.f_new[i,j][8] = -1/4.*self.f[i,j][5] - 1/4.*self.f[i,j][6] + 1/4.*self.f[i,j][7] + 1/4.*self.f[i,j][8]
                                    
    
    @ti.kernel
    def update_macro_var(self): 
        for i, j in ti.ndrange(self.nx, self.ny):
            self.rho[i, j] = 0
            self.vel[i, j] = 0, 0
            for k in ti.static(range(9)):
                self.rho[i, j] += self.f[i, j][k]
                self.vel[i, j] += tm.vec2(self.e[k, 0], self.e[k, 1]) * self.f[i, j][k] 

            self.vel[i, j] += 0.5*self.Force[i,j] # self.vel[i, j] += 0.5*self.Force[None]
            self.vel[i, j] /= self.rho[i, j]
            
    @ti.kernel
    def apply_nee_bc(self):  # impose boundary conditions
        for j in range(1, self.ny - 1):
            self.apply_bc_core(0, j, 1, j) # left: ibc = 0; jbc = j; inb = 1; jnb = j
            self.apply_bc_core(self.nx - 1, j, self.nx - 2, j) # right: ibc = nx-1; jbc = j; inb = nx-2; jnb = j

        for i in range(self.nx):
            self.apply_bc_core(i, self.ny - 1, i, self.ny - 2) # top: ibc = i; jbc = ny-1; inb = i; jnb = ny-2
            self.apply_bc_core(i, 0, i, 1) # bottom: ibc = i; jbc = 0; inb = i; jnb = 1
    
    @ti.func
    def apply_nee_bc_core(self, ibc, jbc, inb, jnb):
        #Non-Equilibrium Extrapolation method, see 5.3.4.3 p194, LBM: Principles and Practise, T. Kruger
        self.rho[ibc, jbc] = self.rho[inb, jnb]
        self.f[ibc, jbc] = self.f_eq(ibc, jbc) - self.f_eq(inb, jnb) + self.f[inb, jnb]
        