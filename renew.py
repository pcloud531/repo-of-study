import numpy as np
import matplotlib.pyplot as plt

def poisson_2d_fdm(h, f):
    # 创建网格
    n = int(1 / h) + 1
    x = np.linspace(0, 1, n)
    y = np.linspace(0, 1, n)
    X, Y = np.meshgrid(x, y)
    
    # 初始化u矩阵
    u = np.zeros((n, n))
    
    # 边界条件
    u[:, 0] = 0
    u[:, -1] = 0
    u[0, :] = 0
    u[-1, :] = 0
    
    # 内部点
    u[1:-1, 1:-1] = (h**2 * f[1:-1, 1:-1] + u[2:, 1:-1] + u[:-2, 1:-1] +
                     u[1:-1, 2:] + u[1:-1, :-2]) / 4

    return X, Y, u

# 精确解
def true_solution(x, y):
    return x * (1 - x) * y * (1 - y)

# 网格大小和源项
h = 0.1
f = np.ones((int(1/h) + 1, int(1/h) + 1)) * 4

# 计算数值解
X, Y, u_num = poisson_2d_fdm(h, f)

# 计算精确解
u_true = true_solution(X, Y)

# 画图比较
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(u_num, extent=[0, 1, 0, 1], origin='lower', cmap='viridis')
plt.colorbar()
plt.title('Numerical Solution')

plt.subplot(1, 2, 2)
plt.imshow(u_true, extent=[0, 1, 0, 1], origin='lower', cmap='viridis')
plt.colorbar()
plt.title('True Solution')

plt.tight_layout()
plt.show()
