import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt

# =====================================================
# 中文字体设置
# =====================================================
try:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
except:
    pass

# =====================================================
# 参数设定
# =====================================================

# 1. 家庭部门
# sigma: 相对风险厌恶系数。中国家庭储蓄意愿较强，通常设定较高，取2.0
sigma = 2.0
# rho_c: 消费习惯/惯性系数。反映消费的平滑特征，取0.7
rho_c = 0.7
# beta: 主观折现因子。对应季度无风险利率，0.98意味着年化利率约8%
beta = 0.98

# 2. 企业部门 (新凯恩斯Phillips曲线)
# theta: Calvo定价参数，价格粘性。中国的价格调整相较于全球基准更灵活，平均调价周期约2.5-3.3个季度
theta = 0.65
# kappa: Phillips曲线斜率 (由 theta 推导)
kappa = (1 - theta) * (1 - beta * theta) / theta

# 3. 能源与技术
# omega: 海洋能在能源投入中的权重
omega = 0.2
# delta_m: 海洋能专用资本折旧率。由于技术更新快、腐蚀等，折旧率略高于普通资本，取0.04
delta_m = 0.04

# 4. 货币当局 (央行泰勒规则)
# rho_i: 利率平滑参数。央行调整利率通常具有渐进性，取0.8
rho_i = 0.8
# phi_pi: 通胀反应系数。中国央行关注通胀，标准值1.5
phi_pi = 1.5
# phi_y: 产出缺口反应系数。中国央行同时也高度关注经济增长，取0.5(标准泰勒规则)
phi_y = 0.5

# 5. 财政当局与支出结构 (稳态比率)
# rho_g: 政府支出冲击的持续性
rho_g = 0.85
# 中国特征：高投资、相对较低的消费占比
# share_c: 居民消费占GDP比重
share_c = 0.55
# share_i: 投资占GDP比重 (约40%-45%，中国的高储蓄高投资特征)
share_i = 0.42
# share_g: 政府购买占GDP比重
share_g = 0.03

# 6. 资本与生产
# alpha: 资本产出弹性，资本在生产中的份额。中国是资本驱动型经济，alpha通常较高
alpha = 0.81
# delta_k: 普通资本折旧率 (季度)，0.025 对应年折旧率10%
delta_k = 0.025
# phi_inv: 投资调整成本/加速数效应。投资对产出变化的敏感度
phi_inv = 1.5

# 7. 外生冲击持续性
rho_z = 0.9  # 海洋能技术冲击 (技术通常具有很强的持续性)
rho_po = 0.7  # 国际化石能源价格冲击 (波动较大，持续性相对较弱)

# =====================================================
# 系统矩阵构建
# -----------------------------------------------------
# 线性系统形式: A * x_t = B * x_{t-1} + C * epsilon_t
# 变量列表 x (n=10):
# 0: c (消费)      1: y (产出)        2: pi (通胀)      3: i (名义利率)
# 4: k_m (海洋资本) 5: z (技术冲击)    6: p_o (油价)     7: k (普通资本)
# 8: inv (投资)    9: g (政府购买)
# =====================================================
n = 10
n_shocks = 4  # [货币, 技术, 能源价格, 财政]

A = np.eye(n)
B = np.zeros((n, n))
C = np.zeros((n, n_shocks))

# 1. IS 曲线 (欧拉方程)
# c_t = rho_c * c_{t-1} - (1/sigma) * (i_t - pi_t - r_star)
# 当期消费取决于习惯和实际利率
B[0, 0] = rho_c
B[0, 3] = - (1 - rho_c) / sigma  # 通常欧拉方程前系数受习惯形成影响
B[0, 2] = (1 - rho_c) / sigma

# 2. 资源约束 (国民收入恒等式)
# y_t = s_c * c_t + s_i * inv_t + s_g * g_t
A[1, 1] = 1
A[1, 0] = -share_c
A[1, 8] = -share_i
A[1, 9] = -share_g
B[1, 1] = 0  # 清除对角线默认值

# 3. 新凯恩斯 Phillips 曲线
# pi_t = beta * E_t[pi_{t+1}] + kappa * mc_t
# 简化版前瞻性 Phillips 曲线 (假设部分后顾以简化求解，或者视为静态预期)
# 这里的实现采用了混合形式：通胀取决于边际成本
# mc_t ~ y_t - alpha*k_{t-1} - omega*z_t ...
B[2, 1] = kappa  # 需求拉动
B[2, 7] = -kappa * alpha  # 资本存量增加降低边际成本
B[2, 5] = -kappa * omega  # 技术进步降低边际成本
B[2, 6] = kappa * (1 - omega)  # 能源成本推动

# 4. 泰勒规则 (货币政策)
# 央行根据上一期利率、当期通胀和产出缺口设定名义利率
B[3, 3] = rho_i
B[3, 2] = (1 - rho_i) * phi_pi
B[3, 1] = (1 - rho_i) * phi_y
C[3, 0] = 1  # 对应 shock_id=0 (货币政策冲击)，影响名义利率i

# 5. 海洋能资本积累
# 资本存量演变方程
A[4, 4] = 1
B[4, 4] = 1 - delta_m
B[4, 1] = delta_m * 0.1  # 假设部分产出转化为海洋能资本

# 6. 海洋能技术冲击过程 AR(1)
B[5, 5] = rho_z
C[5, 1] = 1  # 对应 shock_id=1(技术冲击)，影响海洋能技术Z

# 7. 化石能源价格冲击过程 AR(1)
B[6, 6] = rho_po
C[6, 2] = 1  # 对应 shock_id(价格冲击)，影响化石价格Po

# 8. 普通资本积累方程
# Time-to-build: k_t 取决于上一期资本和上一期投资
A[7, 7] = 1
B[7, 7] = 1 - delta_k
B[7, 8] = delta_k

# 9. 投资需求方程
# 简化加速数模型
# 投资取决于产出增长或水平
A[8, 8] = 1
B[8, 1] = phi_inv

# 10. 财政政策规则 AR(1)
A[9, 9] = 1
B[9, 9] = rho_g
C[9, 3] = 1  # 对应 shock_id=3(财政冲击)，影响政府购买G

# =====================================================
# 模型求解
# -----------------------------------------------------
# 将系统化简为简约形式:
# x_t = G * x_{t-1} + H * epsilon_t
# 其中 G = A^{-1} * B, H = A^{-1} * C
# =====================================================
G = la.solve(A, B)
H = la.solve(A, C)  # H 代表 Impact Matrix


# =====================================================
# 模拟功能函数
# =====================================================
def calculate_irf(G, H, shock_idx, periods=40):
    """
    计算脉冲响应
    在t=0时刻发生一次性的标准差冲击，观察系统随后的回归路径
    """
    n_vars = G.shape[0]
    irf_data = np.zeros((periods, n_vars))

    # 初始冲击：在t=0时刻，冲击向量为单位冲击
    shock_vec = np.zeros(H.shape[1])
    shock_vec[shock_idx] = 1.0

    # x_0 = G*x_{-1} + H*shock
    # 假设 x_{-1} 为稳态 0
    irf_data[0] = H @ shock_vec

    # 迭代计算后续时期
    for t in range(1, periods):
        irf_data[t] = G @ irf_data[t - 1]

    return irf_data

def simulate_stochastic(G, H, periods=100, seed=None):
    """
    计算随机模拟路径
    模拟经济体每一期都受到随机的正态分布冲击序列
    """
    if seed is not None:
        np.random.seed(seed)

    n_vars = G.shape[0]
    n_shocks = H.shape[1]

    sim_data = np.zeros((periods, n_vars))

    # 生成随机冲击序列 (标准正态分布)
    shocks = np.random.randn(periods, n_shocks)

    # 初始状态设为稳态 0
    x_prev = np.zeros(n_vars)

    for t in range(periods):
        x_curr = G @ x_prev + H @ shocks[t]    # x_t = G * x_{t-1} + H * epsilon_t
        sim_data[t] = x_curr
        x_prev = x_curr

    return sim_data

# =====================================================
# 绘图与分析
# =====================================================
variables_labels = [
    '消费 (C)', '产出 (Y)', '通胀 (Pi)', '名义利率 (i)',
    '海洋能资本 (Km)', '海洋能技术 (Z)', '化石能源价格 (Po)',
    '普通资本 (K)', '普通投资 (Inv)', '政府购买 (G)'
]

def plot_simulation(data, title, T_range):
    n_vars = data.shape[1]
    n_cols = 4
    n_rows = int(np.ceil(n_vars / n_cols))

    plt.figure(figsize=(16, 3.5 * n_rows))  # 动态调整高度
    plt.suptitle(title, fontsize=16, y=0.98)

    for i in range(n_vars):
        plt.subplot(n_rows, n_cols, i + 1)
        plt.plot(range(T_range), data[:, i], linewidth=2)
        plt.axhline(0, linestyle='--', color='gray', alpha=0.6)
        plt.title(variables_labels[i])
        plt.grid(True, alpha=0.3)
        plt.ylabel('偏离稳态百分比' if i < 2 else '偏离稳态')

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.show()



shock_names = [
    "货币政策冲击",
    "海洋能技术冲击",
    "化石能源价格冲击",
    "财政政策冲击"
]

T_irf = 40

for i in range(4):
    irf_res = calculate_irf(G, H, shock_idx=i, periods=T_irf)
    plot_title = f"{shock_names[i]}的脉冲响应"
    plot_simulation(irf_res, plot_title, T_irf)
