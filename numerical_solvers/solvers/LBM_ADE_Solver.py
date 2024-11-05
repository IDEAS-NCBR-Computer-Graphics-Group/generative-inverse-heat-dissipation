import taichi as ti
import taichi.math as tm
import torch

from numerical_solvers.solvers.SpectralTurbulenceGenerator import SpectralTurbulenceGenerator
from numerical_solvers.solvers.GaussianTurbulenceGenerator import get_gaussian_noise1d, get_gaussian_noise2d
from numerical_solvers.solvers.LBM_SolverBase import LBM_SolverBase

# Fluid solver based on lattice boltzmann method using taichi language
# Inspired by: https://github.com/hietwll/LBM_Taichi

@ti.data_oriented
class LBM_ADE_Solver(LBM_SolverBase):
    def __init__(self, domain_size, kin_visc, bulk_visc, cs2,
                turbulenceGenerator: SpectralTurbulenceGenerator):
        print(f"LBM_ADE_Solver constructor called.")
        super().__init__(domain_size, kin_visc, cs2, turbulenceGenerator)
        
        
    def init(self, np_gray_image): 
        self.rho.from_numpy(np_gray_image)
        self.vel.fill(0)
        self.init_fields()
                   
    def solve(self, iterations):
        if self.iterations_counter[None] + iterations > self.max_iter[None]:
            iterations = self.max_iter[None] - self.iterations_counter[None]
            
        for iteration in range(iterations):                
            self.stream()
            self.update_macro_var()

            # self.collide_srt()
            self.collide_cm()
             
            u_turb, v_turb = self.turbulenceGenerator.generate_turbulence(self.iterations_counter[None])     
            turb_numpy = torch.stack((u_turb, v_turb), axis=-1)  # Shape becomes (128, 128, 2)
            self.vel.from_torch(turb_numpy)
            self.Force.from_torch(turb_numpy)
            
            # self.init_gaussian_force_field(1E-2, 0, 1)
            # self.init_gaussian_velocity_field(1E-2, 0, 1)
            
            # self.apply_bb()
            self.apply_nee_bc()
            self.update_iteration_counter()
        
            # if self.iterations_counter % 10 == 0:
            #     print(f"iterations: {self.iterations_counter}")
            
        # print(f"Solver run for iterations: {self.iterations_counter[None]}")
                            
        if self.iterations_counter[None] == self.max_iter[None]:
             print(f"Solver run for max iterations {self.max_iter}.")
                    

    @ti.kernel
    def collide_srt(self):
        omega_kin = self.omega_kin[self.iterations_counter[None]]
        
        for i, j in ti.ndrange((1, self.nx - 1), (1, self.ny - 1)):
            for k in ti.static(range(9)):
                feq = self.f_eq(i, j)
                self.f_new[i, j][k] = (1. - omega_kin) * self.f[i, j][k] + feq[k] * omega_kin
        
    @ti.kernel
    def collide_cm(self):
        
        omega_kin = self.omega_kin[self.iterations_counter[None]]
        cs2 = self.cs2[self.iterations_counter[None]]
        
        for i, j in ti.ndrange((1, self.nx - 1), (1, self.ny - 1)): 
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

            # m000 = self.f[i,j][0]
            m000 = self.rho[i,j]          
            
            #SRT
            # self.f_new[i,j][0] = m000
            # self.f_new[i,j][1] = self.f[i,j][1]*(1.- omega_kin) 
            # self.f_new[i,j][2] = self.f[i,j][2]*(1.- omega_kin) 
            # self.f_new[i,j][3] = self.f[i,j][3]*(1.- omega_kin) + omega_kin*m000/3.
            # self.f_new[i,j][4] = self.f[i,j][4]*(1.- omega_kin) + omega_kin*m000/3.
            # self.f_new[i,j][5] = self.f[i,j][5]*(1.- omega_kin)
            # self.f_new[i,j][6] = self.f[i,j][6]*(1.- omega_kin) 
            # self.f_new[i,j][7] = self.f[i,j][7]*(1.- omega_kin) 
            # self.f_new[i,j][8] = self.f[i,j][8]*(1.- omega_kin) +  omega_kin*m000/9.
            
            #TRT
            self.f_new[i,j][0] = m000
            self.f_new[i,j][1] = self.f[i,j][1]*(1.- omega_kin) 
            self.f_new[i,j][2] = self.f[i,j][2]*(1.- omega_kin) 
            self.f_new[i,j][3] = cs2*m000
            self.f_new[i,j][4] = cs2*m000
            self.f_new[i,j][5] = 0
            self.f_new[i,j][6] = self.f[i,j][6]*(1.- omega_kin) 
            self.f_new[i,j][7] = self.f[i,j][7]*(1.- omega_kin) 
            self.f_new[i,j][8] = omega_kin*m000*cs2*cs2
            
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
        for i, j in ti.ndrange((1, self.nx-1), (1,self.ny-1)):
            self.rho[i, j] = 0
            
            for k in ti.static(range(9)):
                self.rho[i, j] += self.f[i, j][k] #+ 0.01*get_gaussian_noise1d(0,1)

            # self.rho[i, j] += 0.01*get_gaussian_noise1d(0,1)
         
            # self.vel[i, j] = self.Force[i, j]
            # self.rho[i, j] += 1E1*self.Force[i, j][0]
            
            # alfa = 0.999
            # mean = 1.
            # self.rho[i, j] = alfa*self.rho[i, j] + (1.-alfa)* (mean + 10*self.Force[i, j][0])
