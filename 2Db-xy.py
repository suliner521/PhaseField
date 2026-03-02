import numpy as np
import matplotlib.pyplot as plt
import os

# ============================================================
# 用户控制参数设置
# ============================================================

PLANE = "xy"          # 平面选择："xy", "xz", "yz"
EVOLUTION = "AC"      # 演化方程："AC" (Allen-Cahn) 或 "CH" (Cahn-Hilliard)
OUTDIR = f"./he/final/2D/png_xy"  # 输出目录
os.makedirs(OUTDIR, exist_ok=True)  # 创建输出目录，如果不存在

data = {
    'model': {
        'dN': [500, 500],  # 网格点数 [Nx, Ny]
        'dLen': [1.0, 1.0]  # 网格间距 [dx, dy]
    },
    'iter': {
        'dtime': 5e-4,  # 时间步长
        'nstep': 400,   # 总步数
        'theta': 0.5    # 隐式参数
    }
}

# ============================================================
# 2D 数学工具类
# ============================================================

class mathTools:
    def getGrad(self, f, dLen):
        # 计算梯度(周期结构)
        dx, dy = dLen
        fx = (np.roll(f, -1, 0) - np.roll(f, 1, 0)) / (2*dx)
        fy = (np.roll(f, -1, 1) - np.roll(f, 1, 1)) / (2*dy)
        return np.stack((fx, fy), axis=0)

    def getDiv(self, F, dLen):
        # 计算散度(周期结构)
        dx, dy = dLen
        Fx, Fy = F
        return (
            (np.roll(Fx, -1, 0) - np.roll(Fx, 1, 0)) / (2*dx) +
            (np.roll(Fy, -1, 1) - np.roll(Fy, 1, 1)) / (2*dy)
        )

mt = mathTools()

# ============================================================
# 网格和初始化（真实2D）
# ============================================================

class mesh:
    def run(self):
        Nx, Ny = data['model']['dN']  # 获取网格点数
        dx, dy = data['model']['dLen']  # 获取网格间距

        x = np.arange(Nx)*dx  # x坐标
        y = np.arange(Ny)*dy  # y坐标
        X, Y = np.meshgrid(x, y, indexing='ij')  # 创建网格

        cx = (Nx-1)*dx/2  # 中心x坐标
        cy = (Ny-1)*dy/2  # 中心y坐标

        R0 = 50.0  # 初始半径

        r2 = (X-cx)**2 + (Y-cy)**2  # 距离平方
        phi = np.exp(-r2/R0**2)  # 初始相场分布

        con = np.clip(phi, 0, 1)  # 限制在[0,1]

        mobi = np.ones_like(con)  # 均匀的迁移率

        return con, {
            'Free': mobi,  # 自由能迁移率
            'Gradient': mobi * 16,  # 梯度能迁移率
            'Bulk': mobi * 4  # 体积能迁移率
        }

# ============================================================
# 自由能计算
# ============================================================

class calEnergyFree:
    def update(self, con, mobi):
        self.con = con  # 相场变量
        self.mobi = mobi  # 迁移率

    def iter(self):
        A = 5  # 自由能参数
        phi = self.con
        dF = 2*A*phi*(1-phi)*(1-2*phi)  # 自由能导数
        self.dcon = -data['iter']['dtime'] * self.mobi * dF  # 时间演化

# ============================================================
# 2D Anisotropic Gradient Energy
# ============================================================

# ============================================================
# 2D 各向异性梯度能计算
# ============================================================

class calEnergyGradient:
    def __init__(self):
        self.dLen = data['model']['dLen']  # 网格间距
        self.theta = data['iter']['theta']  # 隐式参数
        self.old_dF = None  # 上一步的dF
        self.energy = 0  # 能量
        self.dcon = None  # 相场变化
        self.epsilon = 1e-12  # 小量避免除零
        self.grad_thresh = 1e-12  # 梯度阈值
        self.epsilon_field = None
        self.dcon_grad = None

    def update(self, con, mobility):
        self.con = con  # 相场变量
        self.mobility = mobility  # 迁移率

    def iter(self):
        dF = np.zeros_like(self.con)  # 自由能导数
        dE = np.zeros_like(self.con)  # 能量密度

        Dphi = mt.getGrad(self.con, self.dLen)  # 计算梯度

        gx, gy = Dphi
        grad_mag = np.sqrt(gx**2 + gy**2)  # 梯度幅值
        nx = np.zeros_like(grad_mag)  # 法向x分量
        ny = np.zeros_like(grad_mag)  # 法向y分量
        DphiDxAbsInv = np.zeros_like(grad_mag)  # 1/|∇φ|

        mask = grad_mag > self.grad_thresh  # 梯度大于阈值的区域
        nx[mask] = Dphi[0][mask] / grad_mag[mask]  # 单位法向
        ny[mask] = Dphi[1][mask] / grad_mag[mask]
        DphiDxAbsInv[mask] = 1 / grad_mag[mask]

        DnDphi = self._calDnDphi(Dphi)  # 计算DnDphi

        Ep, DEpDphi = self._cal_DEpDDdphi(DphiDxAbsInv, nx, ny, DnDphi)  # 计算各向异性参数
        dE = 0.5 * Ep**2 * grad_mag ** 2  # 梯度能密度
        dF = -mt.getDiv(Ep**2 * Dphi + DEpDphi, self.dLen)  # 梯度能导数

        if self.old_dF is not None:
            dF = self.theta * dF + (1 - self.theta) * self.old_dF  # 隐式时间积分

        if EVOLUTION == "AC":
            self.dcon = -data['iter']['dtime'] * self.mobility * dF  # Allen-Cahn方程
        else:  # CH
            self.dcon = data['iter']['dtime'] * mt.getDiv(
                self.mobility * mt.getGrad(dF, self.dLen),
                self.dLen
            )  # Cahn-Hilliard方程

        self.energy = np.sum(dE) * np.prod(self.dLen)  # 总能量
        self.old_dF = dF.copy()
        self.dcon_grad = self.dcon.copy()

    def _cal_DEpDDdphi(self, DphiDxAbsInv, nx, ny, DnDphi):
        # 计算各向异性参数Ep和其导数
        EParam = [2.0, -0.648935, 0.027655]  # 十二边形参数（简化版）
        EParam = [2.0, -6.48935000e-01, 2.76550000e-02, -9.04989914e+00, 6.65145020e+00, 
                  3.84564882e+01, -1.16793771e+01, -2.94860465e+01, 5.04480942e+00]  # 完整参数

        E0,a12,a6,c1,c2,c3,c4,c5,c6 = EParam  # 各向异性参数

        Inv_pz = np.ones_like(nx)  # 1/pz，pz=1
        nz = np.zeros_like(nx)  # z法向，2D中为0
        pz = np.ones_like(nx)  # pz=1
        
        nx2, ny2 = nx ** 2, ny ** 2  # nx^2, ny^2
        Inv_pz6 = Inv_pz ** 6  # (1/pz)^6
        P6 = (nx2 ** 3 - 15 * nx2 ** 2 * ny2 + 15 * nx2 * ny2 ** 2 - ny2 ** 3) * Inv_pz6  # 六次多项式
        P12 = 2 * P6 ** 2 - 1  # 十二次多项式
        Q6 = 2 * nx * ny * (3 * nx2 - ny2) * (nx2 - 3 * ny2) * Inv_pz6  # 导数相关

        M0 = a6*P6 + a12*P12  # 各向异性项
        M1 = c1*nz + c2*nz**2 + c3*nz**3 + c4*nz**4 + c5*nz**5 + c6*nz**6  # z方向多项式（2D中为0）
        M2 = c1 + 2*c2*nz + 3*c3*nz**2 + 4*c4*nz**3 + 5*c5*nz**4 + 6*c6*nz**5  # M1的导数

        Ep = E0*(1 + M0*pz + M1)  # 各向异性能量参数

        K1 = -6*(a6 + 4*a12*P6)*Q6  # 导数计算
        K2 = -M0*nz + pz*M2

        DEDn = E0 * np.stack([-ny * K1, nx * K1], axis=0)  # 对法向的导数

        DEpDphi = DphiDxAbsInv * Ep * np.einsum("i...,ij...->j...", DEDn, DnDphi) * Inv_pz  # 对梯度的导数

        return Ep, DEpDphi
    
    def _calDnDphi(self, DphiDx):
        # 计算DnDphi，用于导数计算
        p = DphiDx                        # (2, Nx, Ny) 梯度
        p2 = np.sum(p**2, axis=0)         # |∇φ|^2 -> (Nx, Ny)
        pp = p[:, None, ...] * p[None, :, ...]   # (2,2,Nx,Ny) 外积
        I = np.eye(2)[:, :, None, None]          # (2,2,1,1) 单位矩阵
        delta_p2 = I * p2                        # (2,2,Nx,Ny) 对角矩阵
        DnDphi = delta_p2 - pp                   # 分子部分 δ_ij |∇φ|^2 - ∂_i φ ∂_j φ
        return DnDphi


# ============================================================
# 体积能计算
# ============================================================

class calEnergyBulk:
    def update(self, con, mobi):
        self.con = con  # 相场变量
        self.mobi = mobi  # 迁移率

    def iter(self):
        mu = -5  # 化学势参数
        phi = self.con
        dF = mu * 5 * (1 - np.tanh(10*phi - 5)**2)  # 体积能导数
        self.dcon = -data['iter']['dtime'] * self.mobi * dF  # 时间演化

# ============================================================
# 运行模拟
# ============================================================

class run:
    def __init__(self):
        self.con, self.mobilitys = mesh().run()  # 初始化网格和相场
        self.free = calEnergyFree()  # 自由能对象
        self.grad = calEnergyGradient()  # 梯度能对象
        self.bulk = calEnergyBulk()  # 体积能对象
        self.loop()  # 开始循环

    def loop(self):
        for istep in range(data['iter']['nstep']):  # 循环总步数
            self.step()  # 执行一步

            if istep % 10 == 0:  # 每10步保存图像
                plt.figure(figsize=(5,5))
                plt.imshow(self.con.T, origin='lower',
                           cmap='jet', vmin=0, vmax=1)  # 绘制相场
                plt.colorbar(label="φ")  # 颜色条
                plt.title(f"2D Phase Field ({PLANE})  step={istep}")  # 标题
                plt.axis("equal")  # 等比例轴
                plt.axis("off")  # 关闭轴

                fname = f"{OUTDIR}/{PLANE}_{istep:04d}.png"  # 文件名
                plt.savefig(fname, dpi=200)  # 保存图像
                plt.close()  # 关闭图
                print("saved", fname)  # 输出保存信息

    def step(self):
        # 执行一步演化
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
        )  # 更新相场

        self.con = np.clip(self.con, 1e-7, 1-1e-7)  # 限制在[1e-7, 1-1e-7]

# ============================================================

if __name__ == "__main__":
    run()  # 运行模拟
