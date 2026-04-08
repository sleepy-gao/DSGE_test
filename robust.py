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

# 全局变量标签
variables_labels = [
    '消费 (C)', '产出 (Y)', '通胀 (Pi)', '名义利率 (i)',
    '海洋能资本 (Km)', '海洋能技术 (Z)', '化石能源价格 (Po)',
    '普通资本 (K)', '普通投资 (Inv)', '政府购买 (G)'
]


# =====================================================
# 模型构建
# eta_adj: 能源替代弹性调整系数 (基准为 1.0)
# theta: 价格刚性参数 (基准为 0.65)
# rho_g: 财政政策持续性参数 (基准为 0.85)
# =====================================================
def build_and_solve_system(eta_adj=1.0, theta=0.65, rho_g=0.85):
    # 基准固定参数
    sigma, rho_c, beta = 2.0, 0.7, 0.98
    omega = 0.2 * eta_adj  # 用 omega 乘以调整系数来代理替代弹性的传导效应
    delta_m = 0.04
    rho_i, phi_pi, phi_y = 0.8, 1.5, 0.5
    share_c, share_i, share_g = 0.55, 0.42, 0.03
    alpha, delta_k, phi_inv = 0.81, 0.025, 1.5
    rho_z, rho_po = 0.9, 0.7

    # 推导参数
    kappa = (1 - theta) * (1 - beta * theta) / theta

    n = 10
    n_shocks = 4
    A = np.eye(n)
    B = np.zeros((n, n))
    C = np.zeros((n, n_shocks))

    # IS 曲线
    B[0, 0] = rho_c
    B[0, 3] = - (1 - rho_c) / sigma
    B[0, 2] = (1 - rho_c) / sigma
    # 资源约束
    A[1, 1] = 1
    A[1, 0] = -share_c
    A[1, 8] = -share_i
    A[1, 9] = -share_g
    B[1, 1] = 0
    # Phillips 曲线
    B[2, 1] = kappa
    B[2, 7] = -kappa * alpha
    B[2, 5] = -kappa * omega
    B[2, 6] = kappa * (1 - omega)
    # 泰勒规则
    B[3, 3] = rho_i
    B[3, 2] = (1 - rho_i) * phi_pi
    B[3, 1] = (1 - rho_i) * phi_y
    C[3, 0] = 1
    # 海洋能资本积累
    A[4, 4] = 1
    B[4, 4] = 1 - delta_m
    B[4, 1] = delta_m * 0.1
    # 技术冲击
    B[5, 5] = rho_z
    C[5, 1] = 1
    # 价格冲击
    B[6, 6] = rho_po
    C[6, 2] = 1
    # 普通资本积累
    A[7, 7] = 1
    B[7, 7] = 1 - delta_k
    B[7, 8] = delta_k
    # 投资需求
    A[8, 8] = 1
    B[8, 1] = phi_inv
    # 财政政策
    A[9, 9] = 1
    B[9, 9] = rho_g
    C[9, 3] = 1

    G = la.solve(A, B)
    H = la.solve(A, C)
    return G, H


def calculate_irf(G, H, shock_idx, periods=40):
    n_vars = G.shape[0]
    irf_data = np.zeros((periods, n_vars))
    shock_vec = np.zeros(H.shape[1])
    shock_vec[shock_idx] = 1.0
    irf_data[0] = H @ shock_vec
    for t in range(1, periods):
        irf_data[t] = G @ irf_data[t - 1]
    return irf_data


# =====================================================
# 可视化对比
# =====================================================
def plot_robustness_comparison(irf_dict, var_indices, shock_name, title, periods=40):
    n_vars = len(var_indices)
    plt.figure(figsize=(15, 4))
    plt.suptitle(title, fontsize=16, y=1.05)

    line_styles = ['-', '--', ':']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    for i, var_idx in enumerate(var_indices):
        plt.subplot(1, n_vars, i + 1)

        for j, (label, irf_data) in enumerate(irf_dict.items()):
            plt.plot(range(periods), irf_data[:, var_idx],
                     label=label, linestyle=line_styles[j % 3],
                     color=colors[j % 3], linewidth=2.5 if j == 0 else 2)

        plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
        plt.title(f"{shock_name} 对 {variables_labels[var_idx]} 的影响")
        plt.xlabel('季度')
        plt.grid(True, alpha=0.3)
        if i == 0:
            plt.legend()

    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    T = 40

    # -----------------------------------------------------
    # 检验 1: 能源替代弹性的敏感性分析
    # -----------------------------------------------------
    irf_eta = {}
    # 基准
    G_base, H_base = build_and_solve_system(eta_adj=1.0)
    irf_eta['基准参数 (eta=1.0)'] = calculate_irf(G_base, H_base, shock_idx=1, periods=T)
    # 上调20%
    G_high, H_high = build_and_solve_system(eta_adj=1.2)
    irf_eta['替代弹性上调20%'] = calculate_irf(G_high, H_high, shock_idx=1, periods=T)
    # 下调20%
    G_low, H_low = build_and_solve_system(eta_adj=0.8)
    irf_eta['替代弹性下调20%'] = calculate_irf(G_low, H_low, shock_idx=1, periods=T)

    # 绘制产出、消费、投资
    plot_robustness_comparison(irf_eta, [1, 0, 8], "海洋能技术冲击",
                               "检验 1：能源替代弹性(η)的敏感性分析")

    # -----------------------------------------------------
    # 检验 2: 价格刚性强度的稳健性
    # -----------------------------------------------------
    irf_theta = {}
    # 基准 theta=0.65
    irf_theta['基准刚性 (theta=0.65)'] = calculate_irf(G_base, H_base, shock_idx=1, periods=T)
    # 刚性减弱 (调整更频繁) theta=0.5
    G_flex, H_flex = build_and_solve_system(theta=0.50)
    irf_theta['刚性减弱 (theta=0.50)'] = calculate_irf(G_flex, H_flex, shock_idx=1, periods=T)
    # 刚性增强 theta=0.8
    G_rigid, H_rigid = build_and_solve_system(theta=0.80)
    irf_theta['刚性增强 (theta=0.80)'] = calculate_irf(G_rigid, H_rigid, shock_idx=1, periods=T)

    # 绘制通胀、名义利率
    plot_robustness_comparison(irf_theta, [2, 3], "海洋能技术冲击",
                               "检验 2：价格刚性强度(θ)的稳健性分析")

    # -----------------------------------------------------
    # 检验 3: 财政政策持续性的影响
    # -----------------------------------------------------
    irf_rho_g = {}
    # 基准 rho_g=0.85
    irf_rho_g['基准持续性 (rho_g=0.85)'] = calculate_irf(G_base, H_base, shock_idx=3, periods=T)
    # 持续性拉长 rho_g=0.95
    G_g_long, H_g_long = build_and_solve_system(rho_g=0.95)
    irf_rho_g['高持续性 (rho_g=0.95)'] = calculate_irf(G_g_long, H_g_long, shock_idx=3, periods=T)
    # 持续性缩短 rho_g=0.70
    G_g_short, H_g_short = build_and_solve_system(rho_g=0.70)
    irf_rho_g['低持续性 (rho_g=0.70)'] = calculate_irf(G_g_short, H_g_short, shock_idx=3, periods=T)

    # 绘制政府购买、产出、投资
    plot_robustness_comparison(irf_rho_g, [9, 1, 8], "财政政策冲击",
                               "检验 3：财政政策持续性(rho_g)的影响分析")
