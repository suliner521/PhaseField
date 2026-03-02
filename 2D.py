import numpy as np
import pyvista as pv
import os
import time

# ============================================================
# Parameters
# ============================================================
EVOLUTION = "AC"   # "AC" or "CH"

data = {
    'ndim': 2,
    'model': {
        'dN': [320, 451],
        'dLen': [50, 50]
    },
    'iter':{
        'dtime': 0.0005,
        'nstep': 3000,
        'tstart': 0,
        'theta': 0.5
    },
    'out':{
        'ifSavePNG': True,
        'Mp4path': './1.Mp4',
        'Energy_File': './1.txt',
        'PNGpath': './he/final/2D/'
    }
}

# ============================================================
# mathTools (2D operators)
# ============================================================
class mathTools:
    def getGrad0(self, f, dLen):
        fx = (np.roll(f, -1, 0) - np.roll(f, 1, 0)) / (2 * dLen[0])
        fy = (np.roll(f, -1, 1) - np.roll(f, 1, 1)) / (2 * dLen[1])
        return np.array([fx, fy])
    
    def getGrad1(self, f, dLen):
        dx, dy = dLen

        fx = (np.take(f, np.arange(f.shape[0]) + 1, axis=0, mode='wrap')
            - np.take(f, np.arange(f.shape[0]) - 1, axis=0, mode='wrap')) * (0.5 / dx)

        fy = (np.take(f, np.arange(f.shape[1]) + 1, axis=1, mode='wrap')
            - np.take(f, np.arange(f.shape[1]) - 1, axis=1, mode='wrap')) * (0.5 / dy)

        return np.stack((fx, fy), axis=0)

    def getDiv0(self, F, dLen):
        return (
            (np.roll(F[0], -1, 0) - np.roll(F[0], 1, 0)) / (2 * dLen[0]) +
            (np.roll(F[1], -1, 1) - np.roll(F[1], 1, 1)) / (2 * dLen[1])
        )
    def getDiv1(self, F, dLen):
        dx, dy = dLen
        Fx, Fy = F
        nx, ny = Fx.shape

        div = (
            (np.take(Fx, np.arange(nx)+1, axis=0, mode='wrap')
        - np.take(Fx, np.arange(nx)-1, axis=0, mode='wrap')) * (0.5/dx)

        + (np.take(Fy, np.arange(ny)+1, axis=1, mode='wrap')
        - np.take(Fy, np.arange(ny)-1, axis=1, mode='wrap')) * (0.5/dy)
        )

        return div


mt = mathTools()

# ============================================================
# Mesh & Initialization (2D → 3D extrude)
# ============================================================
class mesh:
    def run(self):
        Nx, Ny = data['model']['dN']
        dx = data['model']['dLen'][0]

        con2d = np.zeros((Nx, Ny))
        for i in range(Nx):
            x = i * dx / 100
            for j in range(Ny):
                z = j * dx / 100
                if z <= 30:
                    con2d[i, j] = 1
                elif z <= 120:
                    if 25 <= x <= 30 and (z - 120) <= 18 * (x - 30):
                        con2d[i, j] = 1
                    elif 30 < x < 50:
                        con2d[i, j] = 1
                    elif 50 <= x <= 55 and (z - 30) <= -18 * (x - 55):
                        con2d[i, j] = 1
                    elif 105 <= x <= 110 and (z - 120) <= 18 * (x - 110):
                        con2d[i, j] = 1
                    elif 110 < x < 130:
                        con2d[i, j] = 1
                    elif 130 <= x <= 135 and (z - 120) <= -18 * (x - 130):
                        con2d[i, j] = 1

        con = con2d

        mobi = np.ones_like(con)
        mobi[con > 0.9] = 0
        mobi[:, -10:] = 0

        return con, {
            'Free': mobi * 10,
            'Gradient': mobi * 16 * 100 ** 2 * 16 ** 2,
            'Bulk': mobi * 1 * 16
        }

class mesh:
    def run(self):
        Nx, Ny = 320, 451
        dx = 50

        x = np.arange(Nx)*dx
        y = np.arange(Ny)*dx

        X, Y = np.meshgrid(x, y, indexing='ij')

        # ---------- 中心 ----------
        cx = (Nx-1)*dx/2
        cy = (Ny-1)*dx/2

        # ---------- 半径 ----------
        R0 = 12.0 * 50   # 真实晶核半径（建议 10~15 个网格）

        # ---------- 球形距离 ----------
        # r2 = (X-cx)**2 + (Y-cy)**2 + (np.minimum(Z-cz, Nz - Z + cz)**2)
        r2 = (X-cx)**2 + (Y-cy)**2
        # r2 = (np.minimum(X-cx, Nx - X + cx)**2) + (Y-cy)**2 + (Z-cz)**2

        # ---------- 相场 ----------
        phi = np.exp(-r2 / R0**2)

        # phi = np.maximum(phi, phi_i)

        con = np.clip(phi,0,1)

        # con = np.repeat(con2d[:, :, None], Nz, axis=2)

        mobi = np.ones_like(con)
        # mobi[con > 0.9] = 0
        # mobi[:, -10:, :] = 0

        return con, {
            'Free': mobi * 10,
            'Gradient': mobi * 16 * 100 ** 2 * 16 ** 2,
            'Bulk': mobi * 1 * 16
        }

# ============================================================
# Free energy
# ============================================================
class calEnergyFree:
    def update(self, con, mobi):
        self.con = con
        self.mobi = mobi

    def iter(self):
        A = 0.534 
        # A = 5
        phi = self.con
        dF = 2 * A * phi * (1 - phi) * (1 - 2 * phi)
        self.dcon = -data['iter']['dtime'] * self.mobi * dF

# ============================================================
# Anisotropic Gradient Energy (YOUR VERSION, 2.5D)
# ============================================================
class calEnergyGradient:
    def __init__(self):
        self.dLen = data['model']['dLen']
        self.theta = data['iter']['theta']
        self.old_dF = None
        self.energy = 0
        self.dcon = None
        self.epsilon = 1e-12
        self.grad_thresh = 1e-12
        self.epsilon_field = None
        self.dcon_grad = None

    def update(self, con, mobility):
        self.con = con
        self.mobility = mobility

    def iter(self):
        dF = np.zeros_like(self.con)
        dE = np.zeros_like(self.con)

        Dphi = mt.getGrad1(self.con, self.dLen)

        gx, gy = Dphi
        grad_mag = np.sqrt(gx**2 + gy**2)
        nx = np.zeros_like(grad_mag)
        ny = np.zeros_like(grad_mag)
        DphiDxAbsInv = np.zeros_like(grad_mag)

        mask = grad_mag > self.grad_thresh
        nx[mask] = gx[mask] / grad_mag[mask]
        ny[mask] = gy[mask] / grad_mag[mask]
        DphiDxAbsInv[mask] = 1 / grad_mag[mask]

        DnDphi = self._calDnDphi(Dphi)

        Ep, DEpDphi = self._cal_DEpDDdphi(DphiDxAbsInv, nx, ny, DnDphi)
        dE = 0.5 * Ep**2 * grad_mag ** 2
        dF = -mt.getDiv1(Ep**2 * Dphi + DEpDphi, self.dLen)

        self.epsilon_field = Ep * 16
        self.dcon_grad = dF

        if self.old_dF is not None:
            dF = self.theta * dF + (1 - self.theta) * self.old_dF

        if EVOLUTION == "AC":
            self.dcon = -data['iter']['dtime'] * self.mobility * dF
        else:  # CH
            self.dcon = data['iter']['dtime'] * mt.getDiv1(
                self.mobility * mt.getGrad1(dF, self.dLen),
                self.dLen
            )

        self.energy = np.sum(dE) * np.prod(self.dLen)
        self.old_dF = dF.copy()
        self.dcon_grad = self.dcon.copy()

    def _cal_DEpDDdphi(self, DphiDxAbsInv, nx, ny, DnDphi):
        EParam = [0.12483, -6.48935000e-01, 2.76550000e-02, -9.04989914e+00, 6.65145020e+00, 
                  3.84564882e+01, -1.16793771e+01, -2.94860465e+01, 5.04480942e+00]
        # EParam = [0.12483, 0, 0, 0, 0, 0, 0, 0, 0]

        E0,a12,a6,c1,c2,c3,c4,c5,c6 = EParam
        
        nx2, ny2 = nx ** 2, ny ** 2
        P6 = (nx2 ** 3 - 15 * nx2 ** 2 * ny2 + 15 * nx2 * ny2 ** 2 - ny2 ** 3)
        P12 = 2 * P6 ** 2 - 1
        Q6 = 2 * nx * ny * (3 * nx2 - ny2) * (nx2 - 3 * ny2)

        M0 = a6*P6 + a12*P12

        Ep = E0*(1 + M0)

        K1 = -6*(a6 + 4*a12*P6)*Q6

        DEDn = E0 * np.stack([-ny * K1, nx * K1], axis=0)
        
        # a12, a6 = 0, 0
        # c1, c2, c3, c4, c5, c6 = 0, 0, 0, 0, 0, 0

        # M1 = c1*ny + c2*ny**2 + c3*ny**3 + c4*ny**4 + c5*ny**5 + c6*ny**6
        # M2 = c1 + 2*c2*ny + 3*c3*ny**2 + 4*c4*ny**3 + 5*c5*ny**4 + 6*c6*ny**5

        # Ep = E0*(1 + (a12 + a6) * nx + M1)
        # K1 = -(a12 + a6) * ny + M2 * nx

        # Inv_nx = np.zeros_like(nx)
        # mask = np.abs(nx) > self.grad_thresh
        # Inv_nx[mask] = 1 / nx[mask]

        # DEDn = E0 * np.stack([-ny * 0, Inv_nx * K1], axis=0)

        DEpDphi = DphiDxAbsInv * Ep * np.einsum("i...,ij...->j...", DEDn, DnDphi)
        return Ep, DEpDphi
    
    def _calDnDphi(self, DphiDx):
        p = DphiDx                  # (2, Nx, Ny)
        p2 = np.sum(p**2, axis=0)    # |∇φ|^2
        pp = p[:, None, ...] * p[None, :, ...]
        I = np.eye(2)[:, :, None, None]
        delta_p2 = I * p2
        DnDphi = delta_p2 - pp
        return DnDphi

# ============================================================
# Bulk energy
# ============================================================
class calEnergyBulk:
    def update(self, con, mobi):
        self.con = con
        self.mobi = mobi

    def iter(self):
        mu = -0.3444
        # mu = -5
        phi = self.con
        dF = mu * 5 * (1 - np.tanh(10 * phi - 5)**2)
        self.dcon = -data['iter']['dtime'] * self.mobi * dF


class draw:
    def __init__(self):
        import matplotlib.pyplot as plt
        import os

        self.plt = plt
        self.ifSavePNG = data['out']['ifSavePNG']
        self.png_dir = data['out']['PNGpath']   # 例如 "out_png/"

        if self.ifSavePNG:
            os.makedirs(self.png_dir, exist_ok=True)

        self.origin_range = None

        # ===== 屏幕显示（三列）=====
        self.fig, (self.ax1, self.ax2, self.ax3) = plt.subplots(1, 3, figsize=(18, 5))
        self.fig.suptitle("Phase Field Simulation", fontsize=14)

        # ===== PNG保存用（只画 con）=====
        # if self.ifSavePNG:
        #     self.fig_png, self.ax_png = plt.subplots(1, 1, figsize=(6, 6))

        self.contour_con = None


    def draw_init(self, con, ep, dcon):
        self.origin_range = con.copy()
        ny, nx = con.shape
        self.extent = [0, 0.5*ny, 0, 0.5*nx]

        # ===== 屏幕 =====
        self.ax1.contourf(con.T, levels=np.linspace(0,1,11), cmap='rainbow')
        self.ax1.contour(self.origin_range.T, levels=[0.9], colors="k", linewidths=3)
        self.ax1.set_title('Concentration')
        self.ax1.set_aspect('equal')

        self.ax2.contourf(ep.T, cmap='rainbow', levels=np.linspace(0, 4, 11))
        self.ax2.set_title('Ep')
        self.ax2.set_aspect('equal')

        self.ax3.contourf(dcon.T, cmap='rainbow', levels=np.linspace(-2e-4, 2e-4, 11))
        self.ax3.set_title('dcon')
        self.ax3.set_aspect('equal')

        self.plt.ion()
        self.plt.show(block=False)

        # ===== PNG（第0帧）=====
        if self.ifSavePNG:
            self._save_png(con, 0)

    def draw_update(self, t, con, ep, dcon):

        # ===== 屏幕刷新 =====
        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()

        self.ax1.contourf(con.T, levels=np.linspace(0,1,11), cmap='rainbow')
        self.ax1.contour(self.origin_range.T, levels=[0.9], colors="k", linewidths=3)
        self.ax1.set_title(f'Concentration  t={t:.3f}')
        self.ax1.set_aspect('equal')

        self.ax2.contourf(ep.T, cmap='rainbow', levels=np.linspace(0, 4, 11))
        self.ax2.set_title('Ep')
        self.ax2.set_aspect('equal')

        self.ax3.contourf(dcon.T, cmap='rainbow', levels=np.linspace(-2e-4, 2e-4, 11))
        self.ax3.set_title('dcon')
        self.ax3.set_aspect('equal')

        self.fig.canvas.draw_idle()
        self.plt.pause(0.01)

        # ===== PNG 只存 con =====
        if self.ifSavePNG:
            self._save_png(con, t)
    def _save_png(self, con, t):
        # self.ax_png.clear()

        # self.ax_png.contourf(con.T, levels=np.linspace(0,1,11), cmap='rainbow', extent=self.extent)
        # self.ax_png.contour(self.origin_range.T, levels=[0.99], colors="k", linewidths=3, extent=self.extent)
        # self.ax_png.set_aspect('equal')
        # # self.ax_png.set_title(f"Concentration  t={t:.3f}")

        # fname = f"{self.png_dir}/con_{t:08.3f}.png"
        # self.fig_png.savefig(fname, dpi=600, bbox_inches='tight')
        pass

    def draw_finish(self):
        self.plt.ioff()
        self.plt.close(self.fig)
        # if self.ifSavePNG:
        #     self.plt.close(self.fig_png)

class write:
    def write_init(self, fname):
        if os.path.exists(fname): os.remove(fname)
        open(fname, 'a').close()
    def write_update(self, fname, line):
        with open(fname, 'a') as f: f.write(line + '\n')

class run:
    def __init__(self):
        self.init_all()
        self.loop()
        # np.save('con_0107.npy', self.con)

    def init_all(self):
        self.con, self.mobilitys = mesh().run()
        self.free = calEnergyFree()
        self.grad = calEnergyGradient()
        self.bulk = calEnergyBulk()
        
        self.grad.update(self.con, self.mobilitys['Gradient'])
        self.grad.iter()
        
        self.isShow = True
        self.dw = draw() if self.isShow else None
        if self.isShow: 
            self.dw.draw_init(self.con, self.grad.epsilon_field, self.grad.dcon)

        self.Energy_File = data['out']['Energy_File']
    def loop(self):
        nstep = data['iter']['nstep']
        dtime = data['iter']['dtime']
        for istep in range(nstep):
            t = istep * dtime
            self.step()
            
            if self.isShow and (istep % 50 == 0):
                self.dw.draw_update(t, self.con, self.grad.epsilon_field, self.grad.dcon)
        
        if self.isShow: self.dw.draw_finish()

    def step(self):
        self.free.update(self.con, self.mobilitys['Free'])
        self.free.iter()
        self.grad.update(self.con, self.mobilitys['Gradient'])
        self.grad.iter()
        self.bulk.update(self.con, self.mobilitys['Bulk'])
        self.bulk.iter()
        self.con += self.free.dcon + self.grad.dcon + self.bulk.dcon
        self.con = np.clip(self.con, 1e-7, 1 - 1e-7)

if __name__ == "__main__":
    run()