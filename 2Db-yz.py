import numpy as np
import matplotlib.pyplot as plt
import os

# ============================================================
# User control
# ============================================================

PLANE = "yz"          # "xy", "xz", "yz"
EVOLUTION = "AC"      # "AC" or "CH"
OUTDIR = f"./he/final/2D/png_yz"
os.makedirs(OUTDIR, exist_ok=True)

data = {
    'model': {
        'dN': [500, 500],
        'dLen': [1.0, 1.0]
    },
    'iter': {
        'dtime': 5e-4,
        'nstep': 400,
        'theta': 0.5
    }
}

# ============================================================
# 2D math tools
# ============================================================

class mathTools:
    def getGrad(self, f, dLen):
        dx, dy = dLen
        fx = (np.roll(f, -1, 0) - np.roll(f, 1, 0)) / (2*dx)
        fy = (np.roll(f, -1, 1) - np.roll(f, 1, 1)) / (2*dy)
        return np.stack((fx, fy), axis=0)

    def getDiv(self, F, dLen):
        dx, dy = dLen
        Fx, Fy = F
        return (
            (np.roll(Fx, -1, 0) - np.roll(Fx, 1, 0)) / (2*dx) +
            (np.roll(Fy, -1, 1) - np.roll(Fy, 1, 1)) / (2*dy)
        )

mt = mathTools()

# ============================================================
# Mesh & Initialization (TRUE 2D)
# ============================================================

class mesh:
    def run(self):
        Nx, Ny = data['model']['dN']
        dx, dy = data['model']['dLen']

        x = np.arange(Nx)*dx
        y = np.arange(Ny)*dy
        X, Y = np.meshgrid(x, y, indexing='ij')

        cx = (Nx-1)*dx/2
        cy = (Ny-1)*dy/2

        R0 = 50.0

        r2 = (X-cx)**2 + (Y-cy)**2
        phi = np.exp(-r2/R0**2)

        con = np.clip(phi, 0, 1)

        mobi = np.ones_like(con)

        return con, {
            'Free': mobi,
            'Gradient': mobi * 16,
            'Bulk': mobi * 4
        }

# ============================================================
# Free Energy
# ============================================================

class calEnergyFree:
    def update(self, con, mobi):
        self.con = con
        self.mobi = mobi

    def iter(self):
        A = 5
        phi = self.con
        dF = 2*A*phi*(1-phi)*(1-2*phi)
        self.dcon = -data['iter']['dtime'] * self.mobi * dF

# ============================================================
# 2D Anisotropic Gradient Energy
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

        Dphi = mt.getGrad(self.con, self.dLen)

        gy, gz = Dphi
        grad_mag = np.sqrt(gy**2 + gz**2)
        ny = np.zeros_like(grad_mag)
        nz = np.zeros_like(grad_mag)
        DphiDxAbsInv = np.zeros_like(grad_mag)

        mask = grad_mag > self.grad_thresh
        ny[mask] = Dphi[0][mask] / grad_mag[mask]
        nz[mask] = Dphi[1][mask] / grad_mag[mask]
        DphiDxAbsInv[mask] = 1 / grad_mag[mask]

        DnDphi = self._calDnDphi(Dphi)

        Ep, DEpDphi = self._cal_DEpDDdphi(DphiDxAbsInv, ny, nz, DnDphi)
        dE = 0.5 * Ep**2 * grad_mag ** 2
        dF = -mt.getDiv(Ep**2 * Dphi + DEpDphi, self.dLen)

        if self.old_dF is not None:
            dF = self.theta * dF + (1 - self.theta) * self.old_dF

        if EVOLUTION == "AC":
            self.dcon = -data['iter']['dtime'] * self.mobility * dF
        else:  # CH
            self.dcon = data['iter']['dtime'] * mt.getDiv(
                self.mobility * mt.getGrad(dF, self.dLen),
                self.dLen
            )

        self.energy = np.sum(dE) * np.prod(self.dLen)
        self.old_dF = dF.copy()
        self.dcon_grad = self.dcon.copy()

    def _cal_DEpDDdphi(self, DphiDxAbsInv, ny, nz, DnDphi):
        EParam = [2.0, -0.648935, 0.027655]  # 十二边形 
        EParam = [2.0, -6.48935000e-01, 2.76550000e-02, -9.04989914e+00, 6.65145020e+00, 
                  3.84564882e+01, -1.16793771e+01, -2.94860465e+01, 5.04480942e+00]

        E0,a12,a6,c1,c2,c3,c4,c5,c6 = EParam

        pz2 = np.maximum(1 - nz ** 2, self.grad_thresh)
        pz = np.sqrt(pz2)
        Inv_pz = np.zeros_like(pz)
        mask = pz > self.grad_thresh
        Inv_pz[mask] = 1 / pz[mask]

        nx = np.zeros_like(ny)
        
        nx2, ny2 = nx ** 2, ny ** 2
        Inv_pz6 = Inv_pz ** 6
        P6 = (nx2 ** 3 - 15 * nx2 ** 2 * ny2 + 15 * nx2 * ny2 ** 2 - ny2 ** 3) * Inv_pz6
        P12 = 2 * P6 ** 2 - 1
        Q6 = 2 * nx * ny * (3 * nx2 - ny2) * (nx2 - 3 * ny2) * Inv_pz6

        M0 = a6*P6 + a12*P12
        M1 = c1*nz + c2*nz**2 + c3*nz**3 + c4*nz**4 + c5*nz**5 + c6*nz**6
        M2 = c1 + 2*c2*nz + 3*c3*nz**2 + 4*c4*nz**3 + 5*c5*nz**4 + 6*c6*nz**5

        Ep = E0*(1 + M0*pz + M1)

        K1 = -6*(a6 + 4*a12*P6)*Q6
        K2 = -M0*nz + pz*M2

        DEDn = E0 * np.stack([nx * K1, K2], axis=0)

        DEpDphi = DphiDxAbsInv * Ep * np.einsum("i...,ij...->j...", DEDn, DnDphi) * Inv_pz

        return Ep, DEpDphi
    
    def _calDnDphi(self, DphiDx):
        p = DphiDx                        # (2, Nx, Ny)
        p2 = np.sum(p**2, axis=0)         # |∇φ|^2 -> (Nx, Ny)
        pp = p[:, None, ...] * p[None, :, ...]   # (2,2,Nx,Ny)
        I = np.eye(2)[:, :, None, None]          # (2,2,1,1)
        delta_p2 = I * p2                        # (2,2,Nx,Ny)
        DnDphi = delta_p2 - pp                   # 分子部分
        return DnDphi


# ============================================================
# Bulk energy
# ============================================================

class calEnergyBulk:
    def update(self, con, mobi):
        self.con = con
        self.mobi = mobi

    def iter(self):
        mu = -5
        phi = self.con
        dF = mu * 5 * (1 - np.tanh(10*phi - 5)**2)
        self.dcon = -data['iter']['dtime'] * self.mobi * dF

# ============================================================
# Run
# ============================================================

class run:
    def __init__(self):
        self.con, self.mobilitys = mesh().run()
        self.free = calEnergyFree()
        self.grad = calEnergyGradient()
        self.bulk = calEnergyBulk()
        self.loop()

    def loop(self):
        for istep in range(data['iter']['nstep']):
            self.step()

            if istep % 10 == 0:
                plt.figure(figsize=(5,5))
                plt.imshow(self.con.T, origin='lower',
                           cmap='jet', vmin=0, vmax=1)
                plt.colorbar(label="φ")
                plt.title(f"2D Phase Field ({PLANE})  step={istep}")
                plt.axis("equal")
                plt.axis("off")

                fname = f"{OUTDIR}/{PLANE}_{istep:04d}.png"
                plt.savefig(fname, dpi=200)
                plt.close()
                print("saved", fname)

    def step(self):

        self.free.update(self.con, self.mobilitys['Free'])
        self.free.iter()

        self.grad.update(self.con, self.mobilitys['Gradient'])
        self.grad.iter()

        self.bulk.update(self.con, self.mobilitys['Bulk'])
        self.bulk.iter()

        self.con += (
            self.free.dcon +
            self.grad.dcon +
            self.bulk.dcon
        )

        self.con = np.clip(self.con, 1e-7, 1-1e-7)

# ============================================================

if __name__ == "__main__":
    run()
