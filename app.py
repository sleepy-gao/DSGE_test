import streamlit as st
import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt

# =====================================================
# 页面基本设置
# =====================================================
st.set_page_config(page_title="海洋能 DSGE 模型仿真平台", layout="wide")

# 中文字体处理 (尝试加载)
try:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
except:
    pass

st.title("🌊 海洋能 DSGE 模型交互式仿真平台")
st.markdown("本系统基于你提供的 DSGE 2.0 模型，支持实时参数调优与脉冲响应分析。")

# =====================================================
# 侧边栏：参数设定
# =====================================================
st.sidebar.header("⚙️ 模型参数设置")

with st.sidebar.expander("家庭与生产参数", expanded=True):
    sigma = st.slider("相对风险厌恶 (sigma)", 0.5, 5.0, 1.5)
    rho_c = st.slider("消费惯性 (rho_c)", 0.0, 0.95, 0.7)
    beta = st.slider("折现因子 (beta)", 0.9, 0.999, 0.99)
    theta = st.slider("Calvo 定价 (theta)", 0.1, 0.9, 0.75)

with st.sidebar.expander("能源与资本参数"):
    omega = st.slider("海洋能占比 (omega)", 0.05, 0.8, 0.3)
    delta_m = st.slider("资本折旧率 (delta_m)", 0.01, 0.1, 0.05)

with st.sidebar.expander("货币政策参数"):
    rho_i = st.slider("利率平滑 (rho_i)", 0.0, 0.95, 0.7)
    phi_pi = st.slider("通胀反应 (phi_pi)", 1.0, 3.0, 1.5)
    phi_y = st.slider("产出反应 (phi_y)", 0.0, 1.0, 0.2)

with st.sidebar.expander("外部冲击持久性"):
    rho_z = st.slider("技术冲击持续性 (rho_z)", 0.5, 0.99, 0.9)
    rho_po = st.slider("能源价格冲击持续性 (rho_po)", 0.5, 0.99, 0.8)


# =====================================================
# 核心计算逻辑
# =====================================================
def solve_and_irf(shock_type, T=40):
    # 计算派生参数
    kappa = (1 - theta) * (1 - beta * theta) / theta

    # 变量排序: [c, y, pi, i, k_m, z, p_o]
    n = 7
    A = np.eye(n)
    B = np.zeros((n, n))
    C = np.zeros((n, 3))

    # 1. IS 曲线
    B[0, 0] = rho_c
    B[0, 3] = -1 / sigma
    B[0, 2] = 1 / sigma

    # 2. 产出恒等式
    B[1, 0] = 1

    # 3. 菲利普斯曲线
    B[2, 1] = kappa
    B[2, 5] = -kappa * omega
    B[2, 6] = kappa * (1 - omega)

    # 4. 泰勒规则
    B[3, 3] = rho_i
    B[3, 2] = (1 - rho_i) * phi_pi
    B[3, 1] = (1 - rho_i) * phi_y
    C[3, 0] = 1  # 货币政策冲击

    # 5. 海洋能资本
    B[4, 4] = 1 - delta_m
    B[4, 1] = delta_m

    # 6. 技术冲击
    B[5, 5] = rho_z
    C[5, 1] = 1

    # 7. 价格冲击
    B[6, 6] = rho_po
    C[6, 2] = 1

    # 求解
    G = la.solve(A, B)
    impact_mat = la.solve(A, C)

    # 映射冲击 ID
    shock_map = {"货币政策冲击": 0, "海洋能技术冲击": 1, "化石能源价格冲击": 2}
    shock_id = shock_map[shock_type]

    # 生成 IRF
    x = np.zeros((T, n))
    x[0] = impact_mat[:, shock_id]
    for t in range(1, T):
        x[t] = G @ x[t - 1]
    return x


# =====================================================
# 前端展示逻辑
# =====================================================
col1, col2 = st.columns([1, 3])

with col1:
    st.write("### 冲击选择")
    selected_shock = st.radio(
        "选择一个外生冲击进行模拟：",
        ["海洋能技术冲击", "化石能源价格冲击", "货币政策冲击"]
    )
    periods = st.number_input("模拟期数", value=40, step=5)

with col2:
    resp = solve_and_irf(selected_shock, T=periods)
    labels = ['消费', '产出', '通胀', '名义利率', '海洋能资本', '海洋能技术', '化石能源价格']

    fig, axes = plt.subplots(3, 3, figsize=(15, 10))
    fig.patch.set_facecolor('#f0f2f6')

    for i in range(len(labels)):
        ax = axes.flatten()[i]
        ax.plot(resp[:, i], color='#1f77b4', linewidth=2)
        ax.axhline(0, color='black', linestyle='--', alpha=0.3)
        ax.set_title(labels[i], fontsize=12)
        ax.grid(True, alpha=0.3)

    # 隐藏多余的子图
    for j in range(len(labels), 9):
        axes.flatten()[j].axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    st.pyplot(fig)

# 数据导出功能
st.divider()
if st.checkbox("查看原始模拟数据"):
    import pandas as pd

    df = pd.DataFrame(resp, columns=labels)
    st.dataframe(df.style.highlight_max(axis=0))
    st.download_button("下载 CSV 数据", df.to_csv(index=False), "irf_data.csv")
