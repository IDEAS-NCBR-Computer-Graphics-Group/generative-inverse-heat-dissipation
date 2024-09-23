
import taichi as ti
import taichi.math as tm

import numpy as np

# Fluid solver based on lattice boltzmann method using taichi language
# Inspired by: https://github.com/hietwll/LBM_Taichi

@ti.data_oriented
class LBM_NS_Solver_OLD:
    def __init__(self, name, domain_size, kin_visc, bulk_visc=None, spectralTurbulenceGenerator=None):
        self.name = name # name of the flow case
        self.nx, self.ny = domain_size    #domain size, by convention, dx = dy = dt = 1.0 (lattice units)

        self.omega = 1.0 / (3.0 * kin_visc + 0.5) 
        self.rho = ti.field(float, shape=(self.nx, self.ny))
        self.vel = ti.Vector.field(2, float, shape=(self.nx, self.ny))

        self.f = ti.Vector.field(9, float, shape=(self.nx, self.ny))
        self.f_new = ti.Vector.field(9, float, shape=(self.nx, self.ny))
        self.w = ti.types.vector(9, float)(4, 1, 1, 1, 1, 1 / 4, 1 / 4, 1 / 4, 1 / 4) / 9.0
        self.e = ti.types.matrix(9, 2, int)([0, 0], [1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [-1, 1], [-1, -1], [1, -1])

        self.Force = ti.Vector.field(2, float, shape=(self.nx, self.ny))
        
    def init(self, np_gray_image):   
        self.rho.from_numpy(np_gray_image)
        self.vel.fill(0)    
        self.init_gaussian_force_field(0*1E-3, 0, 1)
        self.init_fields()
               
    @ti.kernel
    def init_fields(self):
        for i, j in ti.ndrange(self.nx, self.ny):
            self.f[i, j] = self.f_new[i, j] = self.f_eq(i, j)
    
    @ti.kernel
    def init_gaussian_force_field(self, magnitude: float,  mu: float, variance: float):
        # noise = magnitude*get_gaussian_noise(mu,variance)
        # self.Force[None] = ti.Vector([noise[0], noise[1]])
        
        for i, j in ti.ndrange((1, self.nx - 2), (1, self.ny - 2)):
            noise = magnitude*get_gaussian_noise(mu,variance)
            self.Force[i,j] = ti.Vector([noise[0], noise[1]])
            
            
    @ti.kernel
    def create_ic_hill(self, amplitude: float, size: float, x: float, y: float):
        for i, j in ti.ndrange(self.nx, self.ny):
            r2 = (i - x) ** 2 + (j - y) ** 2
            self.rho[i,j] = self.rho[i,j] + amplitude * ti.exp(-size * r2)
            self.f[i, j] = self.f_new[i, j] = self.f_eq(i, j)
                               
    @ti.func 
    def f_eq(self, i, j):
        eu = self.e @ self.vel[i, j]
        uv = tm.dot(self.vel[i, j], self.vel[i, j])
        return self.w * self.rho[i, j] * (1 + 3 * eu + 4.5 * eu * eu - 1.5 * uv)


    @ti.kernel
    def collide_and_stream(self):  # lbm core equation
        for i, j in ti.ndrange((1, self.nx - 1), (1, self.ny - 1)):
            for k in ti.static(range(9)):
                ip = i - self.e[k, 0]
                jp = j - self.e[k, 1]
                feq = self.f_eq(ip, jp)
                self.f_new[i, j][k] = (1 - self.omega) * self.f[ip, jp][k] + feq[k] * self.omega
    

    @ti.kernel
    def swap_and_update_macro_var(self):  # compute rho u v
        for i, j in ti.ndrange((1, self.nx - 1), (1, self.ny - 1)):
            self.rho[i, j] = 0
            self.vel[i, j] = 0, 0
            for k in ti.static(range(9)):
                self.f[i, j][k] = self.f_new[i, j][k] 
                self.rho[i, j] += self.f_new[i, j][k]
                self.vel[i, j] += tm.vec2(self.e[k, 0], self.e[k, 1]) * self.f_new[i, j][k] 

            self.vel[i, j] /= self.rho[i, j] 
            
    @ti.kernel
    def apply_bc(self):  # impose boundary conditions
        for j in range(1, self.ny - 1):
            self.apply_bc_core(0, j, 1, j) # left: ibc = 0; jbc = j; inb = 1; jnb = j
            self.apply_bc_core(self.nx - 1, j, self.nx - 2, j) # right: ibc = nx-1; jbc = j; inb = nx-2; jnb = j

        for i in range(self.nx):
            self.apply_bc_core(i, self.ny - 1, i, self.ny - 2) # top: ibc = i; jbc = ny-1; inb = i; jnb = ny-2
            self.apply_bc_core(i, 0, i, 1) # bottom: ibc = i; jbc = 0; inb = i; jnb = 1
    
    @ti.func
    def apply_bc_core(self, ibc, jbc, inb, jnb):
        #Non-Equilibrium Extrapolation method, see 5.3.4.3 p194, LBM: Principles and Practise, T. Kruger
        self.rho[ibc, jbc] = self.rho[inb, jnb]
        self.f[ibc, jbc] = self.f_eq(ibc, jbc) - self.f_eq(inb, jnb) + self.f[inb, jnb]
    
    
    # @ti.kernel
    # def apply_bb(self):
    #     for i in range(0, self.nx):
    #         self.apply_bb_core(i, 0) # gui bottom  # left
    #         self.apply_bb_core(i, self.ny-2) # gui top # right
                 
    #     for j in range(1, self.ny-1):
    #         self.apply_bb_core(0, j) # gui left # top
    #         self.apply_bb_core(self.nx-2, j) # gui right # bottom
            
    # @ti.func
    # def apply_bb_core(self, i: int, j: int):
    #     tmp = ti.f32(0.0)          
    #     for k in ti.static([1,2,5,6]):
    #         tmp = self.f[i, j][k]
    #         self.f[i, j][k] = self.f[i, j][k+2]
    #         self.f[i, j][k+2] = tmp    

    #     for k in ti.static(range(9)):
    #          self.f_new[i, j][k] =  self.f[i, j][k]
                 
    def solve(self, iterations):
        for _ in range(iterations):    
            self.collide_and_stream()
            self.swap_and_update_macro_var()
            self.apply_bc()
            # self.apply_bb()



@ti.func
def get_gaussian_noise(mean: float, variance: float) -> ti.types.vector(2, float):
    noise = ti.Vector([0.0, 0.0])  # We need two values, as the Box-Muller gives two values
        
    # build in ti.randn() 
    # Generate a random float sampled from univariate standard normal (Gaussian) distribution of mean 0 and variance 1, using the Box-Muller transformation.
    noise[0] = ti.randn()
    noise[1] = ti.randn()
    
    # Apply limiter

    max_noise = 1E-1
    min_noise = -1E-1
    
    noise[0] = ti.min(ti.max(noise[0], min_noise), max_noise)
    noise[1] = ti.min(ti.max(noise[1], min_noise), max_noise)
    
    return noise