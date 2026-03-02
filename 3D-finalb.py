import numpy as np

# ============================================================
# Parameters
# ============================================================
EVOLUTION = "AC"   # "AC" or "CH"

data = {
    'ndim': 3,
    'model': {
        'dN': [320, 3, 451],
        'dLen': [50, 50, 50]
    },
    'iter': {
        'dtime': 0.0005,
        'nstep': 500,
        'theta': 0.5
    },
    'out': {
        'vtk_file': './he/final/3D/con_field_3D/con_field_3D_final2_*.vtk'
    }
}

# ============================================================
# mathTools (2D operators)
# ============================================================
class mathTools:
    def getGrad0(self, f, dLen):
        fx = (np.roll(f, -1, 0) - np.roll(f, 1, 0)) / (2 * dLen[0])
        fy = (np.roll(f, -1, 1) - np.roll(f, 1, 1)) / (2 * dLen[1])
        fz = (np.roll(f, -1, 2) - np.roll(f, 1, 2)) / (2 * dLen[2])
        return np.array([fx, fy, fz])
    
    def getGrad1(self, f, dLen):
        dx, dy, dz = dLen

        fx = (np.take(f, np.arange(f.shape[0]) + 1, axis=0, mode='wrap')
            - np.take(f, np.arange(f.shape[0]) - 1, axis=0, mode='wrap')) * (0.5 / dx)

        fy = (np.take(f, np.arange(f.shape[1]) + 1, axis=1, mode='wrap')
            - np.take(f, np.arange(f.shape[1]) - 1, axis=1, mode='wrap')) * (0.5 / dy)

        fz = (np.take(f, np.arange(f.shape[2]) + 1, axis=2, mode='wrap')
            - np.take(f, np.arange(f.shape[2]) - 1, axis=2, mode='wrap')) * (0.5 / dz)

        return np.stack((fx, fy, fz), axis=0)

    def getDiv0(self, F, dLen):
        return (
            (np.roll(F[0], -1, 0) - np.roll(F[0], 1, 0)) / (2 * dLen[0]) +
            (np.roll(F[1], -1, 1) - np.roll(F[1], 1, 1)) / (2 * dLen[1]) +
            (np.roll(F[2], -1, 2) - np.roll(F[2], 1, 2)) / (2 * dLen[2])
        )
    def getDiv1(self, F, dLen):
        dx, dy, dz = dLen
        Fx, Fy, Fz = F

        nx, ny, nz = Fx.shape

        div = (
            (np.take(Fx, np.arange(nx)+1, axis=0, mode='wrap')
        - np.take(Fx, np.arange(nx)-1, axis=0, mode='wrap')) * (0.5/dx)

        + (np.take(Fy, np.arange(ny)+1, axis=1, mode='wrap')
        - np.take(Fy, np.arange(ny)-1, axis=1, mode='wrap')) * (0.5/dy)

        + (np.take(Fz, np.arange(nz)+1, axis=2, mode='wrap')
        - np.take(Fz, np.arange(nz)-1, axis=2, mode='wrap')) * (0.5/dz)
        )

        return div


mt = mathTools()

# ============================================================
# Mesh & Initialization (2D → 3D extrude)
# ============================================================
# class mesh:
#     def run(self):
#         Nx, Ny, Nz = data['model']['dN']
#         dx = data['model']['dLen'][0]

#         NNN = 50

#         con2d = np.zeros((Nx, Nz))
#         for i in range(Nx):
#             x = i * dx / 100
#             for j in range(Nz):
#                 z = j * dx / 100
#                 if z <= 30:
#                     con2d[i, j] = 1
#                 elif z <= 120:
#                     if 25 <= x <= 30 and (z - 120) <= 18 * (x - 30):
#                         con2d[i, j] = 1
#                     elif 30 < x < 50:
#                         con2d[i, j] = 1
#                     elif 50 <= x <= 55 and (z - 30) <= -18 * (x - 55):
#                         con2d[i, j] = 1
#                     elif (55 + NNN) <= x <= (60 + NNN) and (z - 120) <= 18 * (x - 60 - NNN):
#                         con2d[i, j] = 1
#                     elif (60 + NNN) < x < (80 + NNN):
#                         con2d[i, j] = 1
#                     elif (80 + NNN) <= x <= (85 + NNN) and (z - 120) <= -18 * (x - 80 - NNN):
#                         con2d[i, j] = 1
#                 # if 30 <= z <= 70 and 30 <= x <= 70:
#                 #     con2d[i, j] = 1

#         con = np.repeat(con2d[:, None, :], Ny, axis=1)

#         mobi = np.ones_like(con) * 16
#         mobi[con > 0.9] = 0
#         mobi[:, :, -10:] = 0

#         return con, {
#             'Free': mobi,
#             'Gradient': mobi * 100 ** 2 * 14 ** 2,
#             'Bulk': mobi
#         }
class mesh:
    def run(self):
        Nx, Ny, Nz = 200, 3, 200
        dx = 1.0

        x = np.arange(Nx)*dx
        y = np.arange(Ny)*dx
        z = np.arange(Nz)*dx

        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

        # ---------- 中心 ----------
        cx = (Nx-1)*dx/2
        cy = (Ny-1)*dx/2
        cz = (Nz-1)*dx/2
        # cy = 0

        # ---------- 半径 ----------
        R0 = 20.0   # 真实晶核半径（建议 10~15 个网格）

        # ---------- 球形距离 ----------
        # r2 = (X-cx)**2 + (Y-cy)**2 + (np.minimum(Z-cz, Nz - Z + cz)**2)
        r2 = (X-cx)**2 + (Z-cz)**2
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
            'Free': mobi,
            'Gradient': mobi * 16,
            'Bulk': mobi * 2
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

        gx, gy, gz = Dphi
        grad_mag = np.sqrt(gx**2 + gy**2 + gz**2)
        nx = np.zeros_like(grad_mag)
        ny = np.zeros_like(grad_mag)
        nz = np.zeros_like(grad_mag)
        DphiDxAbsInv = np.zeros_like(grad_mag)

        mask = grad_mag > self.grad_thresh
        nx[mask] = Dphi[0][mask] / grad_mag[mask]
        ny[mask] = Dphi[1][mask] / grad_mag[mask]
        nz[mask] = Dphi[2][mask] / grad_mag[mask]
        DphiDxAbsInv[mask] = 1 / grad_mag[mask]

        pz2 = np.maximum(1 - nz ** 2, self.grad_thresh)
        pz = np.sqrt(pz2)

        DnDphi = self._calDnDphi(Dphi)

        Ep, DEpDphi = self._cal_DEpDDdphi(DphiDxAbsInv, nx, ny, nz, pz, DnDphi)
        dE = 0.5 * Ep**2 * grad_mag ** 2
        dF = -mt.getDiv1(Ep**2 * Dphi + DEpDphi, self.dLen)

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

    def _cal_DEpDDdphi(self, DphiDxAbsInv, nx, ny, nz, pz, DnDphi):
        EParam = [2.0, -0.648935, 0.027655]  # 十二边形 
        EParam = [0.12483, -6.48935000e-01, 2.76550000e-02, -9.04989914e+00, 6.65145020e+00, 
                  3.84564882e+01, -1.16793771e+01, -2.94860465e+01, 5.04480942e+00]
        # EParam = [2.0, -6.48935000e-01, 2.76550000e-02, 0, 0, 
        #           0, 0, 0, 0]
        # EParam = [2.0, 0.0, 0.75, 0, 0, 
        #           0, 0, 0, 0]
        # EParam = [2.0, 0.0, 0, 0, 0, 
        #           0, 0, 0, 0]

        E0,a12,a6,c1,c2,c3,c4,c5,c6 = EParam

        Inv_pz = np.zeros_like(pz)
        mask = np.abs(pz) > self.grad_thresh
        Inv_pz[mask] = 1 / pz[mask]
        
        nx2, ny2 = nx ** 2, ny ** 2
        Inv_pz6 = Inv_pz ** 6
        P6 = (nx2 ** 3 - 15 * nx2 ** 2 * ny2 + 15 * nx2 * ny2 ** 2 - ny2 ** 3) * Inv_pz6
        P12 = 2 * P6 ** 2 - 1
        Q6 = 2 * nx * ny * (3 * nx2 - ny2) * (nx2 - 3 * ny2) * Inv_pz6

        # mask = (self.con[:, :, :] > 0.05) & (self.con[:, :, :] < 0.95)
        # Ep1 = np.ones_like(Ep)
        # Ep1[mask] = Ep[mask]

        # DEDx = (-ny * K1 - nx * nz * K2) * Inv_pz
        # DEDy = (nx * K1 - ny * nz * K2) * Inv_pz
        # DEDz = K2 * pz

        M0 = a6*P6 + a12*P12
        M1 = c1*nz + c2*nz**2 + c3*nz**3 + c4*nz**4 + c5*nz**5 + c6*nz**6
        M2 = c1 + 2*c2*nz + 3*c3*nz**2 + 4*c4*nz**3 + 5*c5*nz**4 + 6*c6*nz**5

        Ep = E0*(1 + M0*pz + M1)

        K1 = -6*(a6 + 4*a12*P6)*Q6
        K2 = -M0*nz + pz*M2

        # DEDn = E0 * np.stack([-ny * K1 - nx * nz * K2, nx * K1 - ny * nz * K2, K2 * pz ** 2], axis=0)

        DEDn = E0 * np.stack([-ny * K1, nx * K1, K2], axis=0)

        DEpDphi = Ep * np.einsum("i...,ij...->j...", DEDn, DnDphi)

        # DEpDphi *= DphiDxAbsInv * Inv_pz

        return Ep, DEpDphi
    
    def _calDnDphi(self, DphiDx):
        p = DphiDx                         # (3, ...)
        p2 = np.sum(p**2, axis=0)          # |∇φ|^2  -> (Nx,Ny,Nz)
        pp = p[:, None, ...] * p[None, :, ...]   # (3,3,Nx,Ny,Nz)
        I = np.eye(3)[:, :, None, None, None]    # (3,3,1,1,1)
        delta_p2 = I * p2                        # (3,3,Nx,Ny,Nz)
        DnDphi = (delta_p2 - pp)
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
        phi = self.con
        dF = mu * 5 * (1 - np.tanh(10 * phi - 5)**2)
        self.dcon = -data['iter']['dtime'] * self.mobi * dF

# ============================================================
# VTK exporter (legacy, multi-file)
# ============================================================
class VTKExporter:
    def export(self, con, vtkfile):
        Nx, Ny, Nz = con.shape
        dx, dy, dz = data['model']['dLen']

        with open(vtkfile, "w") as f:
            f.write(f'''# vtk DataFile Version 3.0
Phase field
ASCII
DATASET STRUCTURED_POINTS
DIMENSIONS {Nx} {Ny} {Nz}
ORIGIN 0 0 0
SPACING {dx} {dy} {dz}
POINT_DATA {Nx*Ny*Nz}
SCALARS con float 1
LOOKUP_TABLE default
''')

            con.ravel(order="F").tofile(f, sep="\n", format="%.6e")

# ============================================================
# Main run
# ============================================================
class run:
    def __init__(self):
        self.con, self.mobilitys = mesh().run()
        self.free = calEnergyFree()
        self.grad = calEnergyGradient()
        self.bulk = calEnergyBulk()
        # self.exporter = VTKExporter()

        self.loop()

    def loop(self):
        
        conSum = np.zeros((data['iter']['nstep'],))

        for istep in range(data['iter']['nstep']):
            self.step()

            if istep % 10 == 0:
                self.exporter.export(
                    self.con,
                    data['out']['vtk_file'].replace('*', f'{istep:04d}')
                )
                print(f"[saved] {data['out']['vtk_file'].replace('*', f'{istep:04d}')}")
            conSum[istep] = np.sum(self.con)
        np.savetxt('conSum.txt', conSum)

    def step(self):
        self.free.update(self.con, self.mobilitys['Free'])
        self.free.iter()

        self.grad.update(self.con, self.mobilitys['Gradient'])
        self.grad.iter()

        self.bulk.update(self.con, self.mobilitys['Bulk'])
        self.bulk.iter()

        self.con += self.free.dcon + self.grad.dcon + self.bulk.dcon
        self.con = np.clip(self.con, 1e-7, 1 - 1e-7)
    
# ============================================================
if __name__ == "__main__":
    run()
