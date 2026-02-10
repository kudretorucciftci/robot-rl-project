import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image

class RobotEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, world_size=10.0, render_mode=None):
        super(RobotEnv, self).__init__()

        # --- Fizik ve Dünya Parametreleri ---
        self.world_size = world_size
        self.dt = 0.1
        self.max_speed = 10.0
        self.max_acceleration = 4.0
        self.max_angular_velocity = np.pi
        self.drag = 0.98
        self.robot_radius = 0.5
        self.target_radius = 0.4
        self.obstacle_radius = 0.5
        # Dünya köşegeninin uzunluğu, maksimum olası mesafe için
        self.max_world_diagonal = np.linalg.norm([self.world_size, self.world_size])

        # --- Sensör Parametreleri (LIDAR) ---
        self.num_lidar_rays = 8
        self.lidar_range = 5.0  # Sensör menzili
        
        # --- Gözlem ve Aksiyon Uzayları (Sürekli) ---
        # Gözlem: [robot_hizi_x, robot_hizi_y, robot_acisi, hedefe_goreli_x, hedefe_goreli_y,
        #          ...8 adet LIDAR verisi...]
        # Hız limitleri
        obs_low = np.array([-self.max_speed, -self.max_speed, -np.pi,
                            -self.world_size, -self.world_size] + [0.0] * self.num_lidar_rays)
        obs_high = np.array([self.max_speed, self.max_speed, np.pi,
                             self.world_size, self.world_size] + [self.lidar_range] * self.num_lidar_rays)
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

        # Aksiyon: [ivmelenme, donus_hizi] - Her ikisi de -1 ve 1 arasında normalize
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)

        # --- Render ve Durum Değişkenleri ---
        self.render_mode = render_mode
        self.position = None
        self.velocity = None
        self.angle = None
        self.target_position = None
        self._obstacle_locations = []
        self.current_step = 0
        self.max_steps = 300

        self.fig, self.ax = (None, None)
        self.robot_img = None
        if self.render_mode:
            try:
                # PIL kullanarak resmi yükle ve RGBA formatına çevir
                self.robot_img = Image.open("robot.png").convert("RGBA")
                print("robot.png başarıyla yüklendi.")
            except Exception as e:
                print(f"UYARI: 'robot.png' yüklenemedi. Hata: {e}")
                print("Görselleştirme için basit bir daire kullanılacak.")
                self.robot_img = None

    def _get_lidar_readings(self):
        lidar_data = []
        angles = np.linspace(0, 2 * np.pi, self.num_lidar_rays, endpoint=False)
        
        for ray_angle in angles:
            abs_angle = self.angle + ray_angle
            ray_dir = np.array([np.cos(abs_angle), np.sin(abs_angle)])
            
            # Başlangıçta maksimum menzil
            min_dist = self.lidar_range
            
            # Engellerle çarpışma kontrolü
            for obs_pos in self._obstacle_locations:
                # Robot pozisyonundan engele olan vektör
                to_obs = obs_pos - self.position
                # Işın (ray) üzerindeki izdüşüm
                projection = np.dot(to_obs, ray_dir)
                
                if projection > 0:
                    # Işından engele olan en yakın mesafe (dik mesafe)
                    dist_to_ray = np.linalg.norm(to_obs - projection * ray_dir)
                    if dist_to_ray < self.obstacle_radius:
                        # Çember ve doğru kesişimi (basitleştirilmiş)
                        intersect_dist = projection - np.sqrt(self.obstacle_radius**2 - dist_to_ray**2)
                        if 0 < intersect_dist < min_dist:
                            min_dist = intersect_dist
            
            # Duvarlarla çarpışma kontrolü
            for i in range(2): # x ve y eksenleri
                if ray_dir[i] != 0:
                    dist_to_wall = (self.world_size if ray_dir[i] > 0 else 0) - self.position[i]
                    wall_intersect = dist_to_wall / ray_dir[i]
                    if 0 < wall_intersect < min_dist:
                        min_dist = wall_intersect
            
            lidar_data.append(min_dist)
            
        return np.array(lidar_data, dtype=np.float32)

    def _get_obs(self):
        relative_target = self.target_position - self.position
        lidar_readings = self._get_lidar_readings()

        # [robot_hizi_x, robot_hizi_y, robot_acisi, hedefe_goreli_x, hedefe_goreli_y, LIDARx8]
        observation = np.concatenate([
            self.velocity,                      # 2 elements
            [self.angle],                       # 1 element
            relative_target,                    # 2 elements
            lidar_readings                      # 8 elements
        ]).astype(np.float32)

        return observation

    def _get_info(self):
        return {"distance": np.linalg.norm(self.position - self.target_position)}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.position = self.np_random.uniform(self.robot_radius, self.world_size - self.robot_radius, size=2)
        self.velocity = np.zeros(2)
        self.angle = self.np_random.uniform(-np.pi, np.pi)

        while True:
            self.target_position = self.np_random.uniform(self.target_radius, self.world_size - self.target_radius, size=2)
            if np.linalg.norm(self.position - self.target_position) > self.robot_radius + self.target_radius + 1.0:
                break
        
        self._obstacle_locations = []
        for _ in range(int(self.world_size / 2)):
            while True:
                obs_pos = self.np_random.uniform(self.obstacle_radius, self.world_size - self.obstacle_radius, size=2)
                if (np.linalg.norm(self.position - obs_pos) > self.robot_radius + self.obstacle_radius + 0.5 and
                    np.linalg.norm(self.target_position - obs_pos) > self.target_radius + self.obstacle_radius + 0.5):
                    self._obstacle_locations.append(obs_pos)
                    break

        if self.render_mode == "human":
            self._render_frame()
        return self._get_obs(), self._get_info()

    def step(self, action):
        self.current_step += 1
        acceleration = action[0] * self.max_acceleration
        angular_velocity = action[1] * self.max_angular_velocity

        self.angle += angular_velocity * self.dt
        self.angle = (self.angle + np.pi) % (2 * np.pi) - np.pi
        
        forward_vector = np.array([np.cos(self.angle), np.sin(self.angle)])
        self.velocity += forward_vector * acceleration * self.dt
        self.velocity *= self.drag

        speed = np.linalg.norm(self.velocity)
        if speed > self.max_speed:
            self.velocity = (self.velocity / speed) * self.max_speed
            
        prev_pos = self.position.copy()
        self.position += self.velocity * self.dt

        terminated = False
        reward = 0

        if not (self.robot_radius <= self.position[0] <= self.world_size - self.robot_radius and
                self.robot_radius <= self.position[1] <= self.world_size - self.robot_radius):
            self.position = prev_pos
            self.velocity = -self.velocity * 0.5
            reward = -10
        
        for obs_pos in self._obstacle_locations:
            if np.linalg.norm(self.position - obs_pos) < self.robot_radius + self.obstacle_radius:
                terminated = True
                reward = -20
                break
        if terminated:
             return self._get_obs(), reward, terminated, self.current_step >= self.max_steps, self._get_info()

        if np.linalg.norm(self.position - self.target_position) < self.robot_radius + self.target_radius:
            terminated = True
            reward = 100
        else:
            prev_dist = np.linalg.norm(prev_pos - self.target_position)
            current_dist = np.linalg.norm(self.position - self.target_position)
            reward += (prev_dist - current_dist) * 10
        
        reward -= 0.1
        reward -= np.sum(np.square(action)) * 0.05
        
        truncated = self.current_step >= self.max_steps

        if self.render_mode == "human":
            self._render_frame()

        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def render(self):
        if self.render_mode in ["human", "rgb_array"]:
            return self._render_frame()

    def _render_frame(self):
        # rgb_array modu için her seferinde yeni bir figür oluştur (sağlamlık için)
        if self.render_mode == "rgb_array":
            fig, ax = plt.subplots(figsize=(12, 12))
        # human modu için mevcut figürü kullan (verimlilik için)
        else:
            if self.fig is None:
                plt.ion()
                # Pencere boyutunu 12x12 yaparak büyütüyoruz
                self.fig, self.ax = plt.subplots(figsize=(12, 12))
                # Pencere başlığını ayarla
                self.fig.canvas.manager.set_window_title('Robot RL Simülasyonu')
                # Pencereyi öne getir (backend destekliyorsa)
                try:
                    mngr = plt.get_current_fig_manager()
                    mngr.window.attributes("-topmost", True)
                    mngr.window.attributes("-topmost", False)
                except:
                    pass
            fig, ax = self.fig, self.ax

        ax.clear()
        ax.set_xlim(0, self.world_size)
        ax.set_ylim(0, self.world_size)
        ax.set_aspect('equal')
        
        # Izgara ekle (mesafe algısı için)
        ax.grid(True, linestyle='--', alpha=0.6)
        
        for obs_pos in self._obstacle_locations:
            ax.add_patch(Circle(obs_pos, self.obstacle_radius, color='black', zorder=5))
            
        ax.add_patch(Circle(self.target_position, self.target_radius, color='green', zorder=5))

        # --- LIDAR Render ---
        lidar_readings = self._get_lidar_readings()
        angles = np.linspace(0, 2 * np.pi, self.num_lidar_rays, endpoint=False)
        for dist, ray_angle in zip(lidar_readings, angles):
            abs_angle = self.angle + ray_angle
            ray_end = self.position + np.array([np.cos(abs_angle), np.sin(abs_angle)]) * dist
            ax.plot([self.position[0], ray_end[0]], [self.position[1], ray_end[1]], 
                    color='red', linestyle=':', alpha=0.4, zorder=4)

        # --- Robot Çizimi ---
        if self.robot_img:
            rotated_img = self.robot_img.rotate(-np.degrees(self.angle), resample=Image.BICUBIC)
            # 32px olan robot görselini belirgin hale getirmek için zoom değerini 1.5 yapıyorum
            # Bu yaklaşık 48 piksel boyutunda bir görüntü oluşturur
            zoom = 1.5
            oi = OffsetImage(rotated_img, zoom=zoom)
            ab = AnnotationBbox(oi, self.position, frameon=False, zorder=10)
            ax.add_artist(ab)
        else:
            # Görsel yüklenemezse büyük bir mavi daire çiz
            ax.add_patch(Circle(self.position, self.robot_radius, color='blue', zorder=10, alpha=0.8))
            # Ön tarafı gösteren daha kalın bir çizgi
            forward_line = np.array([self.position, self.position + np.array([np.cos(self.angle), np.sin(self.angle)]) * (self.robot_radius + 0.5)])
            ax.plot(forward_line[:, 0], forward_line[:, 1], 'r-', linewidth=5, zorder=11)

        if self.render_mode == "human":
            plt.draw()
            plt.pause(0.001)
        else: # rgb_array
            fig.canvas.draw()
            buf = fig.canvas.tostring_argb()
            img = np.frombuffer(buf, dtype=np.uint8)
            img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))
            img = np.roll(img, shift=-1, axis=2)
            plt.close(fig) # Figürü kapat
            return img

    def close(self):
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None
            if self.render_mode == "human":
                plt.ioff()
