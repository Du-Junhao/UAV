
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Button

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
        self.avoid_timer = 0
        self.avoid_state = False
        self.collision_info = None
        self.passed_collision_point = False

    def update(self, dt):
        if self.avoid_state:
            self.avoid_timer -= dt
            if self.avoid_timer <= 0:
                self.speed = self.original_speed
                self.avoid_state = False
                if self.collision_info and self.distance_covered > self.collision_info['self_dist']:
                    self.passed_collision_point = True

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
                    self_dist = self.distance_covered + self.speed * t
                    other_dist = other.distance_covered + other.speed * t
                    return {
                        'other': other,
                        'time': t,
                        'point': (p1 + p2) / 2,
                        'self_dist': self_dist,
                        'other_dist': other_dist
                    }
        return None

    def avoid_conflict(self, slow=True, duration=3.0):
        if not self.avoid_state:
            self.speed *= 0.8 if slow else 1.2
            self.avoid_timer = duration
            self.avoid_state = True
            self.passed_collision_point = False

def run_simulation(drone_params):
    drones = [Drone(d['start'], d['end'], d['speed'], d['name']) for d in drone_params]

    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(121, projection='3d')
    info_ax = fig.add_subplot(122)
    info_ax.axis('off')

    colors = ['r', 'g', 'b', 'm', 'c', 'y']
    markers = ['o', '^', 's', 'v', '*', 'D']

    drone_scatters = [ax.plot([], [], [], markers[i % len(markers)], label=drones[i].name, color=colors[i % len(colors)])[0] for i in range(len(drones))]
    trails = [ax.plot([], [], [], linestyle='--', color=colors[i % len(colors)])[0] for i in range(len(drones))]
    collision_points = [ax.plot([], [], [], 'X', color='black', markersize=10, alpha=0)[0] for _ in range(len(drones))]

    # 每架无人机预留足够行高，防止碰撞信息重叠
    max_lines_per_drone = 5  # 从4增加到5，为碰撞信息预留更多空间
    line_spacing = 0.04  # 从0.035增加到0.04，增加行间距
    start_y = 0.9
    info_texts = []
    for i in range(len(drones)):
        y_pos = start_y - i * (max_lines_per_drone * line_spacing)
        text = info_ax.text(0.05, y_pos, "", fontsize=10,
                            color=colors[i % len(colors)], transform=info_ax.transAxes)
        info_texts.append(text)

    ax.set_xlim(0, 12)
    ax.set_ylim(0, 12)
    ax.set_zlim(0, 12)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()

    ax_pause = plt.axes([0.8, 0.02, 0.1, 0.05])
    pause_button = Button(ax_pause, 'Pause/Resume')
    paused = [False]

    def pause_resume(event):
        paused[0] = not paused[0]

    pause_button.on_clicked(pause_resume)

    def update(frame):
        if paused[0]:
            return drone_scatters + trails + collision_points + info_texts

        for i, drone in enumerate(drones):
            for j, other in enumerate(drones):
                if i != j:
                    collision_info = drone.predict_conflict(other)
                    if collision_info and not drone.collision_info:
                        drone.collision_info = collision_info
                        other.collision_info = {
                            'other': drone,
                            'time': collision_info['time'],
                            'point': collision_info['point'],
                            'self_dist': collision_info['other_dist'],
                            'other_dist': collision_info['self_dist']
                        }
                        if i < j:
                            drone.avoid_conflict(slow=True)
                            other.avoid_conflict(slow=False)

        for i, drone in enumerate(drones):
            drone.update(0.1)
            pos = drone.position

            drone_scatters[i].set_data(pos[0], pos[1])
            drone_scatters[i].set_3d_properties(pos[2])

            trail_data = np.array(drone.path_history)
            trails[i].set_data(trail_data[:, 0], trail_data[:, 1])
            trails[i].set_3d_properties(trail_data[:, 2])

            if drone.collision_info and not drone.passed_collision_point:
                collision_points[i].set_data(drone.collision_info['point'][0], drone.collision_info['point'][1])
                collision_points[i].set_3d_properties(drone.collision_info['point'][2])
                collision_points[i].set_alpha(1)
            else:
                collision_points[i].set_alpha(0)
                if drone.passed_collision_point:
                    drone.collision_info = None

            state = "Normal"
            if drone.avoid_state:
                state = "Slowing" if drone.speed < drone.original_speed else "Speeding"

            collision_text = ""
            if drone.collision_info and not drone.passed_collision_point:
                other_name = drone.collision_info['other'].name
                point = drone.collision_info['point']
                collision_text = f"  Collision with {other_name} at: ({point[0]:.1f}, {point[1]:.1f}, {point[2]:.1f})\n"

            info_texts[i].set_text(
                f"{drone.name}:\n"
                f"  Speed: {drone.speed:.2f}\n"
                f"  Pos: ({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f})\n"
                f"  State: {state}\n"
                f"{collision_text}"
            )

        return drone_scatters + trails + collision_points + info_texts

    ani = FuncAnimation(fig, update, frames=300, interval=100, blit=False)
    plt.suptitle("3D Drone Collision Avoidance", fontsize=14, y=0.98)  # 提高标题位置
    plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.88, wspace=0.3)  # 调整整体布局
    plt.show()
