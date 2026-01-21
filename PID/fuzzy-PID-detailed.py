import numpy as np
import matplotlib.pyplot as plt
import os

class FuzzyPID:
    def __init__(self, target, dt, min_max_output=(-100, 100)):
        """
        初始化模糊PID控制器
        :param target: 设定目标值
        :param dt: 采样时间
        :param min_max_output: 输出限幅
        """
        self.target = target
        self.dt = dt
        self.min_output, self.max_output = min_max_output
        
        # PID 基础参数 (修正：增强积分和比例)
        self.kp0 = 1.5  # 提高比例系数
        self.ki0 = 0.08  # 显著提高积分系数
        self.kd0 = 0.3
        
        # 状态变量
        self.error_last = 0
        self.integral = 0
        
        # 模糊控制的量化因子 (修正：根据实际误差范围调整)
        # 误差范围预计 -50 到 50 -> 映射到 -3 到 3
        self.Ke = 3.0 / 50.0  # 修正量化因子
        self.Kec = 3.0 / 10.0  # 误差变化率范围
        
        # 模糊输出的比例因子 (增加调整幅度)
        self.Ku_p = 0.1  # 增大
        self.Ku_i = 0.005  # 增大
        self.Ku_d = 0.05

        # 定义模糊规则表 (7x7) - 保持原有规则
        self.Kp_rules = np.array([
            [ 3,  3,  2,  2,  1,  0,  0],
            [ 3,  3,  2,  1,  1,  0, -1],
            [ 2,  2,  2,  1,  0, -1, -1],
            [ 2,  1,  1,  0, -1, -1, -2],
            [ 1,  1,  0, -1, -1, -2, -2],
            [ 0,  0, -1, -1, -2, -2, -3],
            [ 0, -1, -2, -2, -2, -3, -3]
        ])

        self.Ki_rules = np.array([
            [-3, -3, -2, -2, -1,  0,  0],
            [-3, -3, -2, -1, -1,  0,  0],
            [-3, -2, -1, -1,  0,  1,  1],
            [-2, -2, -1,  0,  1,  2,  2],
            [-2, -1,  0,  1,  1,  2,  3],
            [ 0,  0,  1,  1,  2,  3,  3],
            [ 0,  0,  1,  2,  2,  3,  3]
        ])

        self.Kd_rules = np.array([
            [ 1, -1, -2, -2, -2, -1,  1],
            [ 1, -1, -2, -2, -1,  0,  1],
            [ 0, -1, -1, -1,  0,  1,  1],
            [ 0,  0,  0,  0,  0,  0,  0],
            [ 0,  0,  1,  2,  2,  1,  1],
            [ 1,  0,  1,  2,  2,  1,  1],
            [ 1,  1,  2,  2,  2,  1,  1]
        ])

    def _trimf(self, x, abc):
        """三角隶属度函数"""
        a, b, c = abc
        if x <= a or x >= c:
            return 0
        elif a < x <= b:
            return (x - a) / (b - a)
        elif b < x < c:
            return (c - x) / (c - b)
        return 0

    def _get_membership(self, x):
        """计算输入值 x 的隶属度"""
        x = np.clip(x, -3, 3)
        
        mf_params = {
            'NB': [-4, -3, -2],
            'NM': [-3, -2, -1],
            'NS': [-2, -1, 0],
            'ZO': [-1, 0, 1],
            'PS': [0, 1, 2],
            'PM': [1, 2, 3],
            'PB': [2, 3, 4]
        }
        
        memberships = {}
        for level, params in mf_params.items():
            memberships[level] = self._trimf(x, params)
            
        return memberships

    def _defuzzify(self, e_grade, ec_grade, rule_matrix):
        """解模糊 (加权平均法)"""
        levels = ['NB', 'NM', 'NS', 'ZO', 'PS', 'PM', 'PB']
        level_idx = {l: i for i, l in enumerate(levels)}
        
        numerator = 0.0
        denominator = 0.0
        
        for e_level, e_val in e_grade.items():
            if e_val == 0: continue
            for ec_level, ec_val in ec_grade.items():
                if ec_val == 0: continue
                
                weight = min(e_val, ec_val)
                row = level_idx[ec_level]
                col = level_idx[e_level]
                rule_out = rule_matrix[row, col]
                
                numerator += weight * rule_out
                denominator += weight
        
        if denominator == 0:
            return 0
        return numerator / denominator

    def compute(self, current_value):
        """计算PID输出"""
        error = self.target - current_value
        ec = (error - self.error_last) / self.dt
        
        # 1. 模糊化
        e_norm = error * self.Ke
        ec_norm = ec * self.Kec
        
        e_grade = self._get_membership(e_norm)
        ec_grade = self._get_membership(ec_norm)
        
        # 2. 模糊推理 + 解模糊
        dkp = self._defuzzify(e_grade, ec_grade, self.Kp_rules)
        dki = self._defuzzify(e_grade, ec_grade, self.Ki_rules)
        dkd = self._defuzzify(e_grade, ec_grade, self.Kd_rules)
        
        # 3. 调整 PID 参数
        kp = self.kp0 + dkp * self.Ku_p
        ki = self.ki0 + dki * self.Ku_i
        kd = self.kd0 + dkd * self.Ku_d
        
        kp = max(0.1, kp)  # 防止过小
        ki = max(0.01, ki)
        kd = max(0, kd)
        
        # 4. 计算 PID 输出
        self.integral += error * self.dt
        
        # 抗积分饱和
        self.integral = np.clip(self.integral, -300, 300)

        output = (kp * error) + (ki * self.integral) + (kd * ec)
        output = np.clip(output, self.min_output, self.max_output)
        
        self.error_last = error
        
        return output, (kp, ki, kd)

def main():
    # 仿真参数
    dt = 0.1
    sim_time = 100
    steps = int(sim_time / dt)
    target = 50.0
    
    # 初始化控制器
    pid = FuzzyPID(target=target, dt=dt, min_max_output=(-80, 80))
    
    # 修正系统模型：增大控制增益
    y = 0.0
    y_last = 0.0
    y_last2 = 0.0
    
    # 记录数据
    time_list = []
    output_list = []
    target_list = []
    kp_list = []
    ki_list = []
    kd_list = []
    error_list = []
    
    print("开始仿真...")
    
    for i in range(steps):
        t = i * dt
        
        # 1. 计算控制量
        u, gains = pid.compute(y)
        kp_curr, ki_curr, kd_curr = gains
        
        # 2. 系统响应 (修正：增大控制增益)
        # 原始: 0.02 太小 -> 改为 0.1
        y = 1.9 * y_last - 0.92 * y_last2 + 0.1 * u
        
        y_last2 = y_last
        y_last = y
        
        # 记录
        time_list.append(t)
        output_list.append(y)
        target_list.append(target)
        kp_list.append(kp_curr)
        ki_list.append(ki_curr)
        kd_list.append(kd_curr)
        error_list.append(target - y)

    # 绘制结果
    print("仿真结束，正在绘图...")
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 12))
    
    # 图1: 系统响应
    axes[0].plot(time_list, target_list, 'r--', label='Target', linewidth=2)
    axes[0].plot(time_list, output_list, 'b-', label='Fuzzy PID Output', linewidth=2)
    axes[0].set_title('Fuzzy PID Control Response', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Value')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    # 图2: PID 参数自适应变化
    axes[1].plot(time_list, kp_list, label='Kp (Adaptive)', color='green', linewidth=2)
    axes[1].plot(time_list, ki_list, label='Ki (Adaptive)', color='orange', linewidth=2)
    axes[1].plot(time_list, kd_list, label='Kd (Adaptive)', color='purple', linewidth=2)
    axes[1].set_title('PID Parameters Adaptation', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Gain Value')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    # 图3: 跟踪误差
    axes[2].plot(time_list, error_list, 'r-', label='Tracking Error', linewidth=2)
    axes[2].axhline(y=0, color='k', linestyle='--', linewidth=0.8)
    axes[2].set_title('Tracking Error', fontsize=14, fontweight='bold')
    axes[2].set_xlabel('Time (s)')
    axes[2].set_ylabel('Error')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()
    
    plt.tight_layout()
    
    # 保存图片
    save_path = 'fuzzy_pid_fixed_result.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n图表已保存至: {save_path}")
    
    # 性能指标
    final_error = abs(error_list[-1])
    max_overshoot = max(output_list) - target
    settling_idx = next((i for i, e in enumerate(error_list) if abs(e) < 2 and all(abs(error_list[j]) < 2 for j in range(i, min(i+50, len(error_list))))), None)
    settling_time = time_list[settling_idx] if settling_idx else "未稳定"
    
    print(f"\n=== 性能指标 ===")
    print(f"最终稳态误差: {final_error:.2f}")
    print(f"最大超调量: {max_overshoot:.2f}")
    print(f"稳定时间 (±2范围): {settling_time}")
    print(f"平均绝对误差: {np.mean(np.abs(error_list)):.2f}")
    
    plt.show()

if __name__ == "__main__":
    main()