import taichi as ti
import taichi.math as tm
import numpy as np
from math import pi
from solvers.SpectralTurbulenceGenerator import SpectralTurbulenceGenerator
from solvers.GaussianTurbulenceGenerator import get_gaussian_noise

@ti.data_oriented
class LBM_SolverBase:
    def __init__(self, name, domain_size, kin_visc, turbulenceGenerator: SpectralTurbulenceGenerator):
        self.name = name # name of the flow case
        self.nx, self.ny = domain_size  
        # nx, ny as np and ti have different convention
        # moreover gui is also different ;p
        # domain size, by convention, dx = dy = dt = 1.0 (lattice units)

        self.omega_kin = 1.0 / (3.0* kin_visc + 0.5) 
        
        self.rho = ti.field(float, shape=(self.nx, self.ny))
        self.vel = ti.Vector.field(2, float, shape=(self.nx, self.ny))

        self.f = ti.Vector.field(9, float, shape=(self.nx, self.ny))
        self.f_new = ti.Vector.field(9, float, shape=(self.nx, self.ny))
        
        
        # self.Force = ti.Vector.field(2, float, shape=()) # single vector
        self.Force = ti.Vector.field(2, float, shape=(self.nx, self.ny))
        self.w = ti.types.vector(9, float)(4, 1, 1, 1, 1, 1 / 4, 1 / 4, 1 / 4, 1 / 4) / 9.0
        self.e = ti.types.matrix(9, 2, int)([0, 0], 
                                            [1, 0], 
                                            [0, 1], 
                                            [-1, 0], 
                                            [0, -1], 
                                            [1, 1], 
                                            [-1, 1], 
                                            [-1, -1], 
                                            [1, -1])


        self.turbulenceGenerator = turbulenceGenerator
        self.iterations_counter = 0

    @ti.kernel
    def init_gaussian_force_field(self, magnitude: float,  mu: float, variance: float):
        # noise = magnitude*get_gaussian_noise(mu,variance)
        # self.Force[None] = ti.Vector([noise[0], noise[1]])
        
        for i, j in ti.ndrange((1, self.nx - 2), (1, self.ny - 2)):
            noise = magnitude*get_gaussian_noise(mu,variance)
            self.Force[i,j] = ti.Vector([noise[0], noise[1]])
            
    @ti.kernel
    def init_gaussian_velocity_field(self, magnitude: float,  mu: float, variance: float):
        # noise = magnitude*get_gaussian_noise(mu,variance)
        # self.Force[None] = ti.Vector([noise[0], noise[1]])
        
        for i, j in ti.ndrange((1, self.nx - 2), (1, self.ny - 2)):
            noise = magnitude*get_gaussian_noise(mu,variance)
            self.vel[i,j] = ti.Vector([noise[0], noise[1]])
                   
    @ti.kernel
    def create_ic_hill(self, amplitude: float, size: float, x: float, y: float):
        for i, j in ti.ndrange(self.nx, self.ny):
            r2 = (i - x) ** 2 + (j - y) ** 2
            self.rho[i,j] = self.rho[i,j] + amplitude * ti.exp(-size * r2)
            self.f[i, j] = self.f_new[i, j] = self.f_eq(i, j)
               
    @ti.kernel
    def init_fields(self):
        for i, j in ti.ndrange(self.nx, self.ny):
            self.f[i, j] = self.f_new[i, j] = self.f_eq(i, j)
            

    @ti.kernel
    def apply_bb(self):
        for i in range(0, self.nx):
            self.apply_bb_core(i, 0) # gui bottom  # left
            self.apply_bb_core(i, self.ny-2) # gui top # right
                 
        for j in range(1, self.ny-1):
            self.apply_bb_core(0, j) # gui left # top
            self.apply_bb_core(self.nx-2, j) # gui right # bottom
            
    @ti.func
    def apply_bb_core(self, i: int, j: int):
        tmp = ti.f32(0.0)          
        for k in ti.static([1,2,5,6]):
            tmp = self.f[i, j][k]
            self.f[i, j][k] = self.f[i, j][k+2]
            self.f[i, j][k+2] = tmp    

        for k in ti.static(range(9)):
             self.f_new[i, j][k] =  self.f[i, j][k]

    @ti.func 
    def f_eq(self, i, j):
        eu = self.e @ self.vel[i, j]
        uv = tm.dot(self.vel[i, j], self.vel[i, j])
        return self.w * self.rho[i, j] * (1 + 3 * eu + 4.5 * eu * eu - 1.5 * uv)
        
    @ti.kernel
    def stream(self):
        for i, j in ti.ndrange(self.nx, self.ny):
            for k in ti.static(range(9)):
                ip = i - self.e[k, 0]
                jp = j - self.e[k, 1]
                self.f[i, j][k] = self.f_new[ip, jp][k] 