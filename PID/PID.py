import matplotlib.pyplot as plt
import warnings

# --- 基础配置 ---
warnings.filterwarnings("ignore")
plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文
plt.rcParams['axes.unicode_minus'] = False

# --------- 控制器定义 ---------
class PController:
    def __init__(self, kp, name="P"):
        self.kp = kp
        self.name = name

    def compute(self, setpoint, current_val, dt=1.0):
        e = setpoint - current_val
        u = self.kp * e
        return max(0, min(100, u))


class PIController:
    def __init__(self, kp, ki, name="PI", out_min=0, out_max=100):
        self.kp, self.ki = kp, ki
        self.name = name
        self.out_min, self.out_max = out_min, out_max
        self.integral = 0.0

    def compute(self, setpoint, current_val, dt=1.0):
        e = setpoint - current_val
        self.integral += e * dt
        u = self.kp * e + self.ki * self.integral
        return max(self.out_min, min(self.out_max, u))


class PIDControllerBalanced:
    """
    均衡版 PID：
    1) D on measurement（对测量值微分）避免设定值阶跃造成 derivative kick
    2) Anti-windup（积分抗饱和）减少限幅导致的振荡/过冲
    3) 可选设定值权重 beta（略小于1可更稳）
    """
    def __init__(self, kp, ki, kd, name="PID(均衡)",
                 out_min=0, out_max=100, beta=0.9):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.name = name
        self.out_min, self.out_max = out_min, out_max
        self.beta = beta

        self.integral = 0.0
        self.last_val = None  # 用于测量值微分

    def compute(self, setpoint, current_val, dt=1.0):
        e = setpoint - current_val

        # P：设定值加权（beta<1 更稳，减少超调倾向）
        p_out = self.kp * (self.beta * setpoint - current_val)

        # D：对测量值微分（避免设定值阶跃带来的导数冲击）
        if self.last_val is None:
            d_meas = 0.0
        else:
            d_meas = (current_val - self.last_val) / dt
        d_out = - self.kd * d_meas
        self.last_val = current_val

        # 先计算不含积分的输出
        u_no_i = p_out + d_out

        # 预测包含积分后的输出，用于 anti-windup 判定
        u_pred = u_no_i + self.ki * self.integral

        # Anti-windup：如果已经在上限且误差>0（还想升），别继续积分；下限同理
        if not ((u_pred >= self.out_max and e > 0) or (u_pred <= self.out_min and e < 0)):
            self.integral += e * dt

        u = u_no_i + self.ki * self.integral
        return max(self.out_min, min(self.out_max, u))


# --------- 仿真环境 ---------
def run_simulation(controller, steps=150, target=100.0, dt=1.0):
    current_temp = 20.0  # 初始温度
    temp_records = []

    for _ in range(steps):
        power = controller.compute(target, current_temp, dt=dt)

        # 简单热模型：加热 + 与环境温差相关的散热
        heat_gain = power * 0.4
        heat_loss = (current_temp - 20) * 0.1
        current_temp += (heat_gain - heat_loss)

        temp_records.append(current_temp)

    return temp_records


# --------- 运行对比 ---------
if __name__ == "__main__":
    target_val = 100
    steps = 150
    dt = 1.0

    # P / PI / PID（均衡参数起点：可再微调）
    ctrl_p = PController(kp=1.2, name="P（有静差）")
    ctrl_pi = PIController(kp=1.2, ki=0.2, name="PI（快且稳，可能轻微超调）")
    ctrl_pid = PIDControllerBalanced(kp=1.2, ki=0.18, kd=0.20, beta=0.9, name="PID（均衡：更抑制超调）")

    history_p = run_simulation(ctrl_p, steps=steps, target=target_val, dt=dt)
    history_pi = run_simulation(ctrl_pi, steps=steps, target=target_val, dt=dt)
    history_pid = run_simulation(ctrl_pid, steps=steps, target=target_val, dt=dt)

    # --------- 绘图并保存 ---------
    plt.figure(figsize=(12, 6))
    plt.axhline(target_val, color='black', linestyle='--', alpha=0.5, label='目标温度')

    plt.plot(history_p, label=ctrl_p.name, linewidth=2)
    plt.plot(history_pi, label=ctrl_pi.name, linewidth=2)
    plt.plot(history_pid, label=ctrl_pid.name, linewidth=2, color='green')

    plt.title("P vs PI vs PID 控制效果对比（模拟加热系统）")
    plt.xlabel("时间步 (Time Steps)")
    plt.ylabel("实际温度 (°C)")
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)

    save_path = "pid_comparison.png"
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"对比图已保存为: {save_path}")

    plt.show()