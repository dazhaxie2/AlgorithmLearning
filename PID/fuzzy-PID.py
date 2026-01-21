import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 设置中文字体支持
rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False

class FuzzyPID:
    def __init__(self, kp=1.0, ki=0.1, kd=0.01):
        """
        初始化模糊PID控制器
        kp, ki, kd: 初始PID参数
        """
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral = 0
        self.prev_error = 0
        
    def fuzzy_rules(self, error, delta_error):
        """
        模糊规则:根据误差和误差变化率调整PID参数
        """
        # 归一化误差和误差变化率到[-1, 1]
        e = np.clip(error / 10.0, -1, 1)
        de = np.clip(delta_error / 5.0, -1, 1)
        
        # 模糊规则调整系数
        # 根据误差大小和变化率动态调整PID参数
        if abs(e) > 0.5:  # 误差大
            kp_adj = 1.5
            ki_adj = 0.5
            kd_adj = 1.2
        elif abs(e) > 0.2:  # 误差中等
            kp_adj = 1.2
            ki_adj = 0.8
            kd_adj = 1.0
        else:  # 误差小
            kp_adj = 1.0
            ki_adj = 1.2
            kd_adj = 0.8
            
        # 根据误差变化率进一步调整
        if de > 0.3:  # 误差快速增加
            kp_adj *= 1.3
            kd_adj *= 1.5
        elif de < -0.3:  # 误差快速减小
            ki_adj *= 1.2
            
        return kp_adj, ki_adj, kd_adj
    
    def compute(self, setpoint, measured_value, dt=0.01):
        """
        计算模糊PID控制输出
        """
        error = setpoint - measured_value
        delta_error = (error - self.prev_error) / dt
        
        # 模糊规则调整PID参数
        kp_adj, ki_adj, kd_adj = self.fuzzy_rules(error, delta_error)
        
        # 积分项(带抗积分饱和)
        self.integral += error * dt
        self.integral = np.clip(self.integral, -100, 100)
        
        # 计算PID输出
        p_term = self.kp * kp_adj * error
        i_term = self.ki * ki_adj * self.integral
        d_term = self.kd * kd_adj * delta_error
        
        output = p_term + i_term + d_term
        
        self.prev_error = error
        
        return output, error

def simulate_system(controller, setpoint, duration=10, dt=0.01):
    """
    模拟一个一阶系统的响应
    """
    steps = int(duration / dt)
    time = np.linspace(0, duration, steps)
    
    measured = np.zeros(steps)
    control = np.zeros(steps)
    errors = np.zeros(steps)
    
    # 系统初始状态
    measured[0] = 0
    
    for i in range(1, steps):
        # PID控制计算
        u, e = controller.compute(setpoint, measured[i-1], dt)
        control[i] = u
        errors[i] = e
        
        # 一阶系统模型: dy/dt = -y + u
        # 添加一些噪声使其更真实
        noise = np.random.normal(0, 0.1)
        measured[i] = measured[i-1] + dt * (-measured[i-1] + u + noise)
    
    return time, measured, control, errors

# 主程序
if __name__ == "__main__":
    # 创建模糊PID控制器
    fuzzy_pid = FuzzyPID(kp=1.5, ki=0.3, kd=0.05)
    
    # 创建传统PID控制器作为对比
    class TraditionalPID:
        def __init__(self, kp, ki, kd):
            self.kp = kp
            self.ki = ki
            self.kd = kd
            self.integral = 0
            self.prev_error = 0
            
        def compute(self, setpoint, measured, dt):
            error = setpoint - measured
            self.integral += error * dt
            self.integral = np.clip(self.integral, -100, 100)
            derivative = (error - self.prev_error) / dt
            output = self.kp * error + self.ki * self.integral + self.kd * derivative
            self.prev_error = error
            return output, error
    
    traditional_pid = TraditionalPID(kp=1.5, ki=0.3, kd=0.05)
    
    # 设定目标值
    setpoint = 10.0
    
    # 运行仿真
    print("运行模糊PID仿真...")
    time_fuzzy, measured_fuzzy, control_fuzzy, errors_fuzzy = simulate_system(
        fuzzy_pid, setpoint, duration=10, dt=0.01
    )
    
    print("运行传统PID仿真...")
    time_trad, measured_trad, control_trad, errors_trad = simulate_system(
        traditional_pid, setpoint, duration=10, dt=0.01
    )
    
    # 创建图表
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # 图1: 系统响应
    axes[0].plot(time_fuzzy, measured_fuzzy, 'b-', label='模糊PID', linewidth=2)
    axes[0].plot(time_trad, measured_trad, 'r--', label='传统PID', linewidth=2)
    axes[0].axhline(y=setpoint, color='g', linestyle=':', label='目标值', linewidth=2)
    axes[0].set_xlabel('时间 (s)', fontsize=12)
    axes[0].set_ylabel('输出值', fontsize=12)
    axes[0].set_title('系统响应对比', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    # 图2: 控制信号
    axes[1].plot(time_fuzzy, control_fuzzy, 'b-', label='模糊PID', linewidth=1.5)
    axes[1].plot(time_trad, control_trad, 'r--', label='传统PID', linewidth=1.5)
    axes[1].set_xlabel('时间 (s)', fontsize=12)
    axes[1].set_ylabel('控制信号', fontsize=12)
    axes[1].set_title('控制信号对比', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)
    
    # 图3: 误差
    axes[2].plot(time_fuzzy, errors_fuzzy, 'b-', label='模糊PID', linewidth=1.5)
    axes[2].plot(time_trad, errors_trad, 'r--', label='传统PID', linewidth=1.5)
    axes[2].axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    axes[2].set_xlabel('时间 (s)', fontsize=12)
    axes[2].set_ylabel('误差', fontsize=12)
    axes[2].set_title('跟踪误差对比', fontsize=14, fontweight='bold')
    axes[2].legend(fontsize=11)
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图表到本地
    output_filename = 'fuzzy_pid_results.png'
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"\n图表已保存到: {output_filename}")
    
    # 计算性能指标
    settling_time_fuzzy = time_fuzzy[np.where(np.abs(errors_fuzzy) < 0.5)[0][0]] if len(np.where(np.abs(errors_fuzzy) < 0.5)[0]) > 0 else None
    settling_time_trad = time_trad[np.where(np.abs(errors_trad) < 0.5)[0][0]] if len(np.where(np.abs(errors_trad) < 0.5)[0]) > 0 else None
    
    print("\n性能指标:")
    print(f"模糊PID - 最大超调: {(np.max(measured_fuzzy) - setpoint):.2f}")
    print(f"传统PID - 最大超调: {(np.max(measured_trad) - setpoint):.2f}")
    print(f"模糊PID - 稳定时间: {settling_time_fuzzy:.2f}s" if settling_time_fuzzy else "模糊PID - 稳定时间: N/A")
    print(f"传统PID - 稳定时间: {settling_time_trad:.2f}s" if settling_time_trad else "传统PID - 稳定时间: N/A")
    print(f"模糊PID - 平均绝对误差: {np.mean(np.abs(errors_fuzzy)):.4f}")
    print(f"传统PID - 平均绝对误差: {np.mean(np.abs(errors_trad)):.4f}")
    
    plt.show()