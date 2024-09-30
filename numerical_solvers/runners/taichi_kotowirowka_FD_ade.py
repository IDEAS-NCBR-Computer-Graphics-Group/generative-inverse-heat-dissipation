
# import libraries
import matplotlib.pyplot as plt
import taichi as ti
import cv2 # conda install conda-forge::opencv
import numpy as np
import os

ti.init(arch=ti.gpu)

ti_float_precision = ti.f32
# %% read IC
# https://github.com/taichi-dev/image-processing-with-taichi/blob/main/image_transpose.py
# image = cv2.imread('cat_768x768.jpg') 
image = cv2.imread('cat_256x256.jpg') 

gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # convert to grayscale
np_gray_image = np.array(gray_image)
shape = gray_image.shape # 768, 768

print(gray_image.shape)
plt.imshow(gray_image, cmap='gist_gray')
plt.colorbar()
plt.title(f'image')
plt.show()



# %%


dx = 1
dt = 1

# diffusivity
diff_coeff = 0.01  # f1 = ti.field(int, shape=()) - a scalar field in taichi

# create velocity field
maxu = 0.01 # velocity field max magnitude
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

spacer = 25 # 150
ax2.quiver(xx[::spacer],yy[::spacer],np_ux[::spacer],np_uy[::spacer], units="xy", scale=0.05) # linewidth=None
plt.show()


ux = ti.field(dtype=ti_float_precision, shape=shape)
uy = ti.field(dtype=ti_float_precision, shape=shape)
ux.from_numpy(np_ux.T.astype('float32'))  # use transpose as tachi uses SoA layout, yyyyxxxx
uy.from_numpy(np_uy.T.astype('float32'))



# %%
# Taichi kernels

@ti.kernel
def create_ic_hill(pf: ti.template(),
                   amplitude: ti_float_precision, size: ti_float_precision, 
                   x: ti_float_precision, y: ti_float_precision):
    
    for i, j in ti.ndrange((1, shape[0] - 1), (1, shape[1] - 1)):
        r2 = (i - x) ** 2 + (j - y) ** 2
        pf[i, j] = pf[i, j] + amplitude * ti.exp(-size * r2)


@ti.func
def laplacian(psi: ti.template(), i: ti.int32, j: ti.int32):
    return (-4 * psi[i,j] + psi[i,j-1] + psi[i,j+1] + psi[i+1,j] + psi[i-1,j]) / (dx**2)


@ti.func
def gradients(psi: ti.template(), i: ti.int32, j: ti.int32):
    grad_hx =  (psi[i+1,j]-psi[i-1,j])/(2*dx)
    grad_hy =  (psi[i,j+1]-psi[i,j-1])/(2*dx)
    return (grad_hx, grad_hy)

    
@ti.kernel
def update(pf: ti.template(), new_pf: ti.template()):
    for i, j in ti.ndrange((1, shape[0] - 1), (1, shape[1] - 1)):

        pf_lap = laplacian(pf, i, j)
        tmp_coeff = diff_coeff*diff_coeff * pf_lap    
        
        pf_grad = gradients(pf,i,j)
        tmp_coeff += - ux[i,j]*pf_grad[0] - uy[i,j]*pf_grad[1]
        
        new_pf[i, j] = pf[i, j] + tmp_coeff*dt
        
        
# %%
# 

class SwapPairWrapper:
    def __init__(self, cur, nxt):
        self.cur = cur
        self.nxt = nxt

    def swap(self):
        self.cur, self.nxt = self.nxt, self.cur

phi_in = ti.field(dtype=ti_float_precision, shape=shape) 
phi_out = ti.field(dtype=ti_float_precision, shape=shape)

# %%
# iterate


phi_in.from_numpy(np_gray_image)
phi_out.fill(0)
phi_pair = SwapPairWrapper(phi_in, phi_out) 
def main():
    print("ADE")

    # create_ic_hill(phi_pair.cur, 100, 1E-3, int(0.25*shape[0]), int(0.125*shape[0]) )
    # create_ic_hill(phi_pair.cur, 150, 1E-4, int(0.25*shape[0]),  int(0.8*shape[0])  )

    np_phi = phi_pair.cur.to_numpy()
    phi_min = np.min(np_phi)
    phi_max = np.max(np_phi)
    
    os.makedirs("output", exist_ok=True)
    
    for i in range(10001):
        update(phi_pair.cur, phi_pair.nxt)
        phi_pair.swap()

        if i % 2000 == 0:
            np_phi = phi_pair.cur.to_numpy()
            plt.imshow(np_phi, cmap='jet', vmin = phi_min, vmax = phi_max) # hot seismic
            # plt.imshow(phi_pair.cur.to_numpy(), cmap='coolwarm', vmin = -10, vmax = 260)
            plt.colorbar()
            plt.title(f'GPU Solution: {i} iterations'
                      f'\n phi range: {np.min(np_phi):.2f} - {np.max(np_phi):.2f}')
            plt.show()
            
            
            # cv2.imwrite(f'output/rotated_cat_768x768_at_{i}.jpg', np_phi)
            cv2.imwrite(f'output/rotated_cat_256x256_at_{i}.jpg', np_phi)
            


if __name__ == "__main__":
    main()


# %%