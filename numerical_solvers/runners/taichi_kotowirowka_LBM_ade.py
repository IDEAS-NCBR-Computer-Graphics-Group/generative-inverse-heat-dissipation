# Fluid solver based on lattice boltzmann method using taichi language
# Author : Wang (hietwll@gmail.com)
# https://github.com/hietwll/LBM_Taichi
# another version: https://github.com/yjhp1016/taichi_LBM3D


import sys
import matplotlib

import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt

import taichi as ti
import taichi.math as tm
import itertools

import cv2 # conda install conda-forge::opencv || pip install opencv-python

# %% read IC
# https://github.com/taichi-dev/image-processing-with-taichi/blob/main/image_transpose.py
image = cv2.imread('cat_768x768.jpg') 
# image = cv2.imread('japan_1024x640.png') 

gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # convert to grayscale
np_gray_image = np.array(gray_image)

def normalize_grayscale_image_range(image, min_val, max_val):
    """
    Normalize the pixel values of a grayscale image to have a specified range [min_val, max_val].

    Parameters:
    image (np.ndarray): Grayscale image array.
    min_val (float): The minimum value of the desired range.
    max_val (float): The maximum value of the desired range.

    Returns:
    np.ndarray: Normalized image array with pixel values in the range [min_val, max_val].
    """
    # Convert image to float32 to avoid issues with integer division
    image = image.astype(np.float32)
    
    # Compute the mean and standard deviation of the original image
    original_mean = np.mean(image)
    original_std = np.std(image)
    
    # Normalize the image to have mean 0 and std 1
    standardized_image = (image - original_mean) / original_std
    
    # Scale standardized image to fit in range [0, 1]
    min_std = np.min(standardized_image)
    max_std = np.max(standardized_image)
    scaled_image = (standardized_image - min_std) / (max_std - min_std)
    
    # Scale to the desired range [min_val, max_val]
    normalized_image = scaled_image * (max_val - min_val) + min_val
    
    return normalized_image

np_gray_image = normalize_grayscale_image_range(np_gray_image, 0.9, 1.1)

shape = gray_image.shape # 768, 768

print(np_gray_image.shape)
plt.imshow(np_gray_image, cmap='gist_gray')
plt.colorbar()
plt.title(f'image')
# plt.show()

# %% Create an external velocity field

maxu = 0.1 # velocity field max magnitude
L = 1
x = np.linspace(0, L, shape[0], endpoint=True)
y = np.linspace(0, L, shape[1], endpoint=True)
xx, yy = np.meshgrid(x, y)
R = L/5.
xx0 = xx - L/2.
yy0 = yy - L/2.
r = np.sqrt(xx0**2 + yy0**2)
w = np.exp(-r**2/(2*R**2)) / (np.exp(-1/2)*R) * maxu
np_ux =  yy0 * w
np_uy = -xx0 * w 


fig, (ax1, ax2) = plt.subplots(1, 2,  figsize=(16, 4))
im1 = ax1.imshow(np.sqrt(np_ux**2 + np_uy**2), cmap = 'coolwarm', extent=(0, L, 0, L))
ax1.set_title(r'$u_{magnitude}$')
ax1.grid()

spacer = 150
ax2.quiver(xx[::spacer],yy[::spacer],np_ux[::spacer],np_uy[::spacer], units="xy", scale=0.05) # linewidth=None
# plt.show()


# %% run sovler

ti.init(arch=ti.gpu)
ti_float_precision = float # ti.f64

ux = ti.field(dtype=ti_float_precision, shape=shape)
uy = ti.field(dtype=ti_float_precision, shape=shape)
ux.from_numpy(np_ux.T.astype('float32'))  # use transpose as tachi uses SoA layout, yyyyxxxx
uy.from_numpy(np_uy.T.astype('float32'))


@ti.data_oriented
class lbm_solver:
    def __init__(
        self,
        name, # name of the flow case
        nx,  # domain size
        ny,
        niu,  # viscosity of fluid
        bc_type,  # [left,top,right,bottom] boundary conditions: 0 -> Dirichlet ; 1 -> Neumann
        bc_value,  # if bc_type = 0, we need to specify the velocity in bc_value
        cy=0,  # whether to place a cylindrical obstacle
        cy_para=[0.0, 0.0, 0.0],  # location and radius of the cylinder
        ):
        self.name = name
        self.nx = nx  # by convention, dx = dy = dt = 1.0 (lattice units)
        self.ny = ny
        
        self.niu = niu
        self.tau = 3.0 * niu + 0.5
        self.inv_tau = 1.0 / self.tau
        
        self.omega_field = ti.field(float, shape=(nx, ny))
        self.omega_field.fill(self.inv_tau)
        
        self.force_field = ti.Vector.field(2, float, shape=(nx, ny))
        self.force_field.fill(0)
        
        self.rho = ti.field(float, shape=(nx, ny))
        self.rho.from_numpy(np_gray_image)
        self.vel = ti.Vector.field(2, float, shape=(nx, ny))
        self.mask = ti.field(float, shape=(nx, ny))
        self.f_old = ti.Vector.field(9, float, shape=(nx, ny))
        self.f_new = ti.Vector.field(9, float, shape=(nx, ny))
        self.w = ti.types.vector(9, float)(4, 1, 1, 1, 1, 1 / 4, 1 / 4, 1 / 4, 1 / 4) / 9.0
        self.e = ti.types.matrix(9, 2, int)([0, 0], [1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [-1, 1], [-1, -1], [1, -1])
        self.bc_type = ti.field(int, 4)
        self.bc_type.from_numpy(np.array(bc_type, dtype=np.int32))
        self.bc_value = ti.Vector.field(2, float, shape=4)
        self.bc_value.from_numpy(np.array(bc_value, dtype=np.float32))
        self.cy = cy
        self.cy_para = tm.vec3(cy_para)

        self.combined_np_u= np.stack((np_ux.T.astype('float32'), np_uy.T.astype('float32')), axis=-1)
        self.vel.from_numpy(self.combined_np_u)

            
    @ti.func  # compute equilibrium distribution function
    def f_eq(self, i, j):
        eu = self.e @ self.vel[i, j]
        uv = tm.dot(self.vel[i, j], self.vel[i, j])
        return self.w * self.rho[i, j] * (1 + 3 * eu + 4.5 * eu * eu - 1.5 * uv)

    @ti.kernel
    def init(self):
        # self.vel.fill(0)
        # self.rho.fill(1)
        
        self.mask.fill(0)
        for i, j in self.rho:
            self.f_old[i, j] = self.f_new[i, j] = self.f_eq(i, j)
            if self.cy == 1:
                if (i - self.cy_para[0]) ** 2 + (j - self.cy_para[1]) ** 2 <= self.cy_para[2] ** 2:
                    self.mask[i, j] = 1.0

    @ti.kernel
    def change_domain_properties(self, amplitude: ti.f32, x: int, y: int):
        niu = (1./self.omega_field[x,y] -0.5)/3.
        new_omega = 1.0 / 3.0 * (niu + amplitude) + 0.5 

        omega_min = self.omega_field[x,y]
        omega_max = 0.
        # make a thick brush
        brush_radius = 16
        for i, j in ti.ndrange((-brush_radius, brush_radius), (-brush_radius, brush_radius)):   
            self.omega_field[x + i, y + j] = new_omega
            self.force_field[x + i, y + j][0] = amplitude*0.005

            
            if self.omega_field[x + i, y + j] > omega_max:
                omega_max = self.omega_field[x + i, y + j]
            
        print(f"changed omega from {omega_min} to {omega_max}.")
        
        
    
    @ti.kernel
    def collide_and_stream(self):  # lbm core equation
        for i, j in ti.ndrange((1, self.nx - 1), (1, self.ny - 1)):
            for k in ti.static(range(9)):
                ip = i - self.e[k, 0]
                jp = j - self.e[k, 1]
                feq = self.f_eq(ip, jp)
                # self.f_new[i, j][k] = (1 - self.inv_tau) * self.f_old[ip, jp][k] + feq[k] * self.inv_tau
                self.f_new[i, j][k] = (1 - self.omega_field[i,j]) * self.f_old[ip, jp][k] + feq[k] * self.omega_field[i,j]

    @ti.kernel
    def update_macro_var(self):  # compute rho u v
        for i, j in ti.ndrange((1, self.nx - 1), (1, self.ny - 1)):
            self.rho[i, j] = 0
            
            # self.vel[i, j] = 0, 0
            for k in ti.static(range(9)):
                self.f_old[i, j][k] = self.f_new[i, j][k]
                self.rho[i, j] += self.f_new[i, j][k]
                # self.vel[i, j] += tm.vec2(self.e[k, 0], self.e[k, 1]) * self.f_new[i, j][k] 

            # self.vel[i, j] /= self.rho[i, j] 
            # self.vel[i, j] += self.force_field[i,j]/self.omega_field[i,j]
            
    @ti.kernel
    def apply_bc(self):  # impose boundary conditions
        # left and right
        for j in range(1, self.ny - 1):
            # left: dr = 0; ibc = 0; jbc = j; inb = 1; jnb = j
            self.apply_bc_core(1, 0, 0, j, 1, j)

            # right: dr = 2; ibc = nx-1; jbc = j; inb = nx-2; jnb = j
            self.apply_bc_core(1, 2, self.nx - 1, j, self.nx - 2, j)

        # top and bottom
        for i in range(self.nx):
            # top: dr = 1; ibc = i; jbc = ny-1; inb = i; jnb = ny-2
            self.apply_bc_core(1, 1, i, self.ny - 1, i, self.ny - 2)

            # bottom: dr = 3; ibc = i; jbc = 0; inb = i; jnb = 1
            self.apply_bc_core(1, 3, i, 0, i, 1)

        # cylindrical obstacle
        # Note: for cuda backend, putting 'if statement' inside loops can be much faster!
        for i, j in ti.ndrange(self.nx, self.ny):
            if self.cy == 1 and self.mask[i, j] == 1:
                self.vel[i, j] = 0, 0  # velocity is zero at solid boundary
                inb = 0
                jnb = 0
                if i >= self.cy_para[0]:
                    inb = i + 1
                else:
                    inb = i - 1
                if j >= self.cy_para[1]:
                    jnb = j + 1
                else:
                    jnb = j - 1
                self.apply_bc_core(0, 0, i, j, inb, jnb)

    @ti.func
    def apply_bc_core(self, outer, dr, ibc, jbc, inb, jnb):
        if outer == 1:  # handle outer boundary
            if self.bc_type[dr] == 0:
                self.vel[ibc, jbc] = self.bc_value[dr]

            elif self.bc_type[dr] == 1:
                self.vel[ibc, jbc] = self.vel[inb, jnb]

        self.rho[ibc, jbc] = self.rho[inb, jnb]
        self.f_old[ibc, jbc] = self.f_eq(ibc, jbc) - self.f_eq(inb, jnb) + self.f_old[inb, jnb]

    def solve(self):
        gui = ti.GUI(self.name, (self.nx, 2 * self.ny))
        # gui = ti.GUI(self.name, (self.nx, self.ny))  
        self.init()
        while gui.running:
            for e in gui.get_events(ti.GUI.PRESS):
                if e.key in [ti.GUI.ESCAPE, ti.GUI.EXIT]:
                    gui.running = False
                # elif e.key == "r":
                #     reset()
                elif e.key == ti.GUI.LMB:
                    x, y = e.pos
                    node_x, node_y = int(x*self.nx), int(y*self.ny)
                    print(f"changing viscosity at {node_x,node_y}")
                    self.change_domain_properties(0.1,node_x, node_y)
                    
            for _ in range(10):    
                self.collide_and_stream()
                self.update_macro_var()
                self.apply_bc()

            ##  code fragment displaying vorticity is contributed by woclass
            vel = self.vel.to_numpy()
    
            
            vel_mag = (vel[:, :, 0] ** 2.0 + vel[:, :, 1] ** 2.0) ** 0.5
            ## color map
            colors = [
                (1, 1, 0),
                (0.953, 0.490, 0.016),
                (0, 0, 0),
                (0.176, 0.976, 0.529),
                (0, 1, 1),
            ]
            my_cmap = matplotlib.colors.LinearSegmentedColormap.from_list("my_cmap", colors)
            # vel_img = cm.ScalarMappable(norm=matplotlib.colors.Normalize(vmin=0.0, vmax=0.02), cmap=my_cmap).to_rgba(vel_mag)
            vel_img = cm.plasma(vel_mag / 0.1)

            
            # niu_cpu = self.omega_field.to_numpy()
            # niu_img = cm.ScalarMappable(norm=matplotlib.colors.Normalize(vmin=niu_cpu.min(), vmax=niu_cpu.max()), cmap="bwr").to_rgba(niu_cpu)
            
            
            rho_cpu = self.rho.to_numpy()
            # rho_img = cm.plasma(rho_cpu / 0.15)
            # rho_img = cm.ScalarMappable(norm=matplotlib.colors.Normalize(vmin=rho_cpu.min(), vmax=rho_cpu.max()), cmap="bwr").to_rgba(rho_cpu)
            rho_img = cm.ScalarMappable(norm=matplotlib.colors.Normalize(vmin=rho_cpu.min(), vmax=rho_cpu.max()), cmap="viridis").to_rgba(rho_cpu)
            
            # ugrad = np.gradient(vel[:, :, 0])
            # vgrad = np.gradient(vel[:, :, 1])
            # vor = ugrad[1] - vgrad[0]
            # vor_img = cm.ScalarMappable(norm=matplotlib.colors.Normalize(vmin=-0.02, vmax=0.02), cmap=my_cmap).to_rgba(vor)
            # img = np.concatenate((vor_img, vel_img), axis=1)
            
            img = np.concatenate((rho_img, vel_img), axis=1)
            
            # img = np.concatenate((niu_img, vel_img), axis=1)
            # gui.set_image(img)
            gui.set_image(img)
         
            gui.show()
            
            

if __name__ == '__main__':
    # von Karman vortex street: Re = U*D/niu = 200
    # nx = 1024
    # ny = 192
    
    nx = 768
    ny = 768
    is_cylinder_present = 0

    
    lbm = lbm_solver(
        "Box",
        nx,
        ny,
        0.001,
        [0, 0, 0, 0],
        [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
    lbm.solve()
  

    # lid-driven cavity flow: Re = U*L/niu = 1000
        # lbm = lbm_solver(
        #     "Lid-driven Cavity Flow",
        #     256,
        #     256,
        #     0.0255,
        #     [0, 0, 0, 0],
        #     [[0.0, 0.0], [0.1, 0.0], [0.0, 0.0], [0.0, 0.0]])
        # lbm.solve()
