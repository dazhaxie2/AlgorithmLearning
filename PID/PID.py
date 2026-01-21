import matplotlib.pyplot as plt
import numpy as np
import warnings

# --- 基础配置 ---
warnings.filterwarnings("ignore")
plt.rcParams['font.sans-serif'] = ['SimHei']  # 解决中文显示问题
plt.rcParams['axes.unicode_minus'] = False 

class PIDController:
    def __init__(self, kp, ki, kd, name):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.name = name
        self.integral = 0
        self.last_error = 0
        self.history = []

    def compute(self, setpoint, current_val, dt=1):
        error = setpoint - current_val
        
        # P 项
        p_out = self.kp * error
        
        # I 项
        self.integral += error * dt
        i_out = self.ki * self.integral
        
        # D 项 (预测误差变化趋势)
        derivative = (error - self.last_error) / dt
        d_out = self.kd * derivative
        
        self.last_error = error
        output = p_out + i_out + d_out
        
        # 限制输出功率 0-100%
        return max(0, min(100, output))

def run_simulation(controller, steps=150, target=100.0):
    current_temp = 20.0 # 初始温度
    temp_records = []
    
    # 为了让 D 项有用，我们模拟一个具有延迟感和散热的物理环境
    for _ in range(steps):
        power = controller.compute(target, current_temp)
        
        # 物理模型：升温效率 - 散热损失 (散热随温差增大)
        # 模拟系统惯性：温度变化不是瞬间的
        heat_gain = power * 0.4
        heat_loss = (current_temp - 20) * 0.1 
        
        current_temp += (heat_gain - heat_loss)
        temp_records.append(current_temp)
        
    return temp_records

# --- 设置三组控制器 ---
# 1. 只有 P: 快速反应，但有静差
ctrl_p = PIDController(kp=1.2, ki=0.0, kd=0.0, name="纯 P 控制")

# 2. PI: 消除静差，但容易产生超调（冲过头）
ctrl_pi = PIDController(kp=1.2, ki=0.2, kd=0.0, name="PI 控制")

# 3. PID: 既消除了静差，又利用 D 项抑制了超调
ctrl_pid = PIDController(kp=1.2, ki=0.2, kd=1.5, name="完整 PID 控制")

# --- 运行实验 ---
target_val = 100
history_p = run_simulation(ctrl_p, target=target_val)
history_pi = run_simulation(ctrl_pi, target=target_val)
history_pid = run_simulation(ctrl_pid, target=target_val)

# --- 绘图对比 ---
plt.figure(figsize=(12, 6))
plt.axhline(target_val, color='black', linestyle='--', alpha=0.5, label='目标温度')

plt.plot(history_p, label='P (存在静差)', linewidth=2)
plt.plot(history_pi, label='PI (消除静差，但有明显超调)', linewidth=2)
plt.plot(history_pid, label='PID (完美平衡：无静差、低超调)', linewidth=2, color='green')

plt.title("P vs PI vs PID 控制效果深度对比 (模拟加热系统)")
plt.xlabel("时间步 (Time Steps)")
plt.ylabel("实际温度 (°C)")
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)
plt.savefig('pid_comparison.png', bbox_inches='tight', dpi=300)
print("PID 控制效果对比图已保存为 'pid_comparison.png'")
