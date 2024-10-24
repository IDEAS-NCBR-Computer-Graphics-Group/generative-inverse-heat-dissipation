import taichi as ti
import matplotlib
import matplotlib.cm as cm
import numpy as np
import matplotlib


# Insipiration
# https://github.com/taichi-dev/taichi/blob/master/python/taichi/examples/simulation/waterwave.py
# https://docs.tclb.io/tutorials/model-development/1.-finite-difference-wave-equation/
#https://github.com/ggruszczynski/gpu_colab/blob/main/90_taichi_wave.ipynb


ti.init(arch=ti.gpu)

c2 = 1.0        # wave propagation speed
damping = 0.01  # larger damping makes wave vanishes faster when propagating
dx = 0.02
dt = 0.01
shape = 512, 512

h = ti.field(dtype=float, shape=shape) # like height
v = ti.field(dtype=float, shape=shape)
h.fill(0)
v.fill(0)

@ti.kernel
def create_wave(amplitude: ti.f32, x: ti.f32, y: ti.f32):
    for i, j in ti.ndrange((1, shape[0] - 1), (1, shape[1] - 1)):
        r2 = (i - x) ** 2 + (j - y) ** 2
        h[i, j] = h[i, j] + amplitude * ti.exp(-0.02 * r2)


@ti.func
def laplacian_h(i: ti.int32, j: ti.int32):
    return (-4 * h[i, j] + h[i, j - 1] + h[i, j + 1] + h[i + 1, j] + h[i - 1, j]) / (dx**2)


@ti.kernel
def update():
    for i, j in ti.ndrange((1, shape[0] - 1), (1, shape[1] - 1)):
        v[i, j] = v[i, j] + (c2 * laplacian_h(i, j) - damping * v[i, j]) * dt

    for i, j in ti.ndrange((1, shape[0] - 1), (1, shape[1] - 1)):
        h[i, j] = h[i, j] + v[i, j] * dt


# def render_frame():
#
#     gui_res = (1 * solver.nx, 3 * solver.ny)
#     window = ti.ui.Window('CG - Renderer', res=gui_res)
#     gui = window.get_gui()
#     canvas = window.get_canvas()

    # while window.running:
    #     rho_cpu = solver.rho.to_numpy()
    #     rho_img = cm.ScalarMappable(
    #         norm=matplotlib.colors.Normalize(vmin=0.9 * np_init_gray_image.min(),
    #                                          vmax=1.1 * np_init_gray_image.max()),
    #         cmap="gist_gray").to_rgba(rho_cpu)
    #
    #     vel = solver.vel.to_numpy()
    #     vel_mag = np.sqrt((vel[:, :, 0] ** 2 + vel[:, :, 1] ** 2))
    #     vel_img = cm.ScalarMappable(norm=matplotlib.colors.Normalize(vmin=0.0, vmax=0.05),
    #                                 cmap="coolwarm").to_rgba(vel_mag)
    #
    #     force = solver.Force.to_numpy()
    #     force_mag = np.sqrt((force[:, :, 0] ** 2 + force[:, :, 1] ** 2))
    #     force_mag = cm.ScalarMappable(cmap="inferno").to_rgba(force_mag)
    #
    #     img = np.concatenate((rho_img, vel_img, force_mag), axis=1)
    #     canvas.set_image(img.astype(np.float32))
    #
    #     solver.solve(iter_per_frame)
    #     window.show()


def main():
    print("waves")

    create_wave( 10, int(0.25*shape[0]), int(0.125*shape[0]) )
    create_wave(-10, int(0.5*shape[0]),  int(0.75*shape[0])  )

    gui_res_x, gui_res_y = h.shape
    window = ti.ui.Window('CG - Renderer', res=(gui_res_x, 2*gui_res_y))
    gui = window.get_gui()
    canvas = window.get_canvas()

    for i in range(5000):
      update()

      h_cpu = h.to_numpy()
      h_img = cm.ScalarMappable(norm=matplotlib.colors.Normalize(vmin=-10,vmax=10), cmap="gist_gray").to_rgba(h_cpu)

      v_cpu = v.to_numpy()
      v_img = cm.ScalarMappable(norm=matplotlib.colors.Normalize(vmin=-10, vmax=10), cmap="coolwarm").to_rgba(v_cpu)

      img = np.concatenate((h_img, v_img), axis=1)
      canvas.set_image(img.astype(np.float32))
      window.show()

      # if i % 100 == 0:
      #   plt.imshow(h.to_numpy(), vmin = -10, vmax =10, cmap='viridis')
      #   plt.colorbar()
      #   plt.title(f'GPU Solution \n {i} iterations')
      #   plt.show()

if __name__ == "__main__":
    main()