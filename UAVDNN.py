import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import csv
import os

# 创建CSV文件并写入表头（只执行一次）
if not os.path.exists("training_data.csv"):
    with open("training_data.csv", mode="w", newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
            "self_x", "self_y", "self_z",
            "self_speed",
            "other_x", "other_y", "other_z",
            "distance",
            "action"  # 0=maintain original speed, 1=decelerate, 2=accelerate
        ])

class Drone:
    def __init__(self, start, end, speed, name):
        self.start = np.array(start, dtype=float)
        self.end = np.array(end, dtype=float)
        self.speed = speed
        self.name = name
        self.position = self.start.copy()
        self.path_vector = self.end - self.start
        self.total_distance = np.linalg.norm(self.path_vector)
        self.unit_vector = self.path_vector / self.total_distance
        self.distance_covered = 0
        self.original_speed = speed
        self.path_history = [self.position.copy()]
        self.avoid_timer = 0         # 用于恢复原速
        self.avoid_state = False     # 标记是否正在避障

    def update(self, dt):
        # 恢复原速逻辑
        if self.avoid_state:
            self.avoid_timer -= dt
            if self.avoid_timer <= 0:
                self.speed = self.original_speed
                self.avoid_state = False

        # 飞行推进
        self.distance_covered += self.speed * dt
        self.distance_covered = min(self.distance_covered, self.total_distance)
        self.position = self.start + self.unit_vector * self.distance_covered
        self.path_history.append(self.position.copy())

    def time_to_point(self):
        remaining = self.total_distance - self.distance_covered
        return remaining / self.speed if self.speed > 0 else float('inf')

    def predict_position(self, t_future):
        d = self.distance_covered + self.speed * t_future
        d = min(d, self.total_distance)
        return self.start + self.unit_vector * d
    def predict_conflict(self, other, t_horizon=3.0, dt=0.1, threshold=2.0, time_margin=1.0):
        for t in np.arange(0, t_horizon, dt):
            p1 = self.predict_position(t)
            p2 = other.predict_position(t)
            dist = np.linalg.norm(p1 - p2)
            if dist < threshold:
                t1 = self.time_to_point()
                t2 = other.time_to_point()
                if abs(t1 - t2) < time_margin:
                    return True
        return False
    def avoid_conflict(self, slow=True, duration=3.0):
        if not self.avoid_state:
            self.speed *= 0.8 if slow else 1.2
            self.avoid_timer = duration
            self.avoid_state = True

# 初始化无人机（设置两个会相遇）
drones = [
    Drone((0, 0, 0), (10, 10, 10), 1.0, "Drone A"),
    Drone((10, 0, 0), (0, 10, 10), 1.0, "Drone B"),
    Drone((0, 10, 0), (10, 0, 10), 1.2, "Drone C")
]

# 可视化设置
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(121, projection='3d')  # 三维图在左边
info_ax = fig.add_subplot(122)              # 信息显示在右边
info_ax.axis('off')

colors = ['r', 'g', 'b']
markers = ['o', '^', 's']

# 无人机可视化元素
drone_scatters = [ax.plot([], [], [], markers[i], label=drones[i].name, color=colors[i])[0] for i in range(len(drones))]
trails = [ax.plot([], [], [], linestyle='--', color=colors[i])[0] for i in range(len(drones))]

# 状态文本区域
info_texts = [info_ax.text(0.05, 0.9 - i * 0.3, "", fontsize=11, color=colors[i], transform=info_ax.transAxes) for i in range(len(drones))]

# 坐标轴设置
ax.set_xlim(0, 12)
ax.set_ylim(0, 12)
ax.set_zlim(0, 12)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()

def update(frame):
    # 预测并规避冲突
    for i, drone in enumerate(drones):
        for j, other in enumerate(drones):
            if i != j:
                # 构造状态向量
                state = [
                    *drone.position,  # 自己位置
                    drone.speed,      # 自己速度
                    *other.position,  # 其他无人机位置
                    np.linalg.norm(drone.position - other.position)  # 与他人距离
                ]

                # 仅记录有预测冲突的样本
                if drone.predict_conflict(other):
                    if i < j:
                        action = 1  # 减速
                        drone.avoid_conflict(slow=True)
                        other.avoid_conflict(slow=False)
                    else:
                        action = 2  # 加速

                    # 保存样本
                    with open("training_data.csv", mode="a", newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow(state + [action])


    # 更新飞行状态
    for i, drone in enumerate(drones):
        drone.update(0.1)
        pos = drone.position

        # 更新轨迹
        drone_scatters[i].set_data(pos[0], pos[1])
        drone_scatters[i].set_3d_properties(pos[2])

        trail_data = np.array(drone.path_history)
        trails[i].set_data(trail_data[:, 0], trail_data[:, 1])
        trails[i].set_3d_properties(trail_data[:, 2])

        # 信息面板显示
        state = "Normal"
        if drone.avoid_state:
            state = "Slowing" if drone.speed < drone.original_speed else "Speeding"
        info_texts[i].set_text(
            f"{drone.name}:\n"
            f"  Speed: {drone.speed:.2f}\n"
            f"  Pos: ({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f})\n"
            f"  State: {state}"
        )

    return drone_scatters + trails + info_texts

ani = FuncAnimation(fig, update, frames=300, interval=100, blit=False)
plt.suptitle("3D Drone Collision Avoidance with Prediction", fontsize=14)
plt.tight_layout()
plt.show()
