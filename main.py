import os
import time
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from robot_env import RobotEnv

# --- Kurulum ---
# Loglar ve modeller için klasörler oluştur
log_dir = "logs"
model_dir = "models"
os.makedirs(log_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)

# Yeni model için yeni bir isim kullanalım
model_path = os.path.join(model_dir, "ppo_robot_continuous_model.zip")

# --- 1. Ortamı Oluşturma ---
print("Sürekli ortam oluşturuluyor...")
env = RobotEnv(world_size=10.0)
print("Ortam oluşturuldu.")

# --- 2. PPO Ajanını Oluşturma ---
print("PPO ajanı oluşturuluyor (MlpPolicy ile)...")
# Sürekli aksiyon uzayı için MlpPolicy kullanılıyor.
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_dir)
print("Ajan oluşturuldu.")

# --- 3. Ajanı Eğitme ---
# Sürekli uzayda öğrenmek daha zordur, bu yüzden adım sayısını artıralım.
TIMESTEPS = 200000 
print(f"{TIMESTEPS} adım boyunca eğitim başlatılıyor...")
model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="PPO_Robot_Continuous")
print("Eğitim tamamlandı.")

# --- 4. Eğitilmiş Modeli Kaydetme ---
print(f"Model şuraya kaydediliyor: {model_path}")
model.save(model_path)
print("Model kaydedildi.")

del model  # Kaydetme ve yükleme işlemini göstermek için modeli siliyoruz

# --- 5. Eğitilmiş Modeli Değerlendirme ---
print("\n--- Eğitilmiş model değerlendiriliyor ---")

# Değerlendirme için yeni bir ortam (render_mode='human') oluşturalım
eval_env = RobotEnv(world_size=10.0, render_mode="human")
# Eğitilmiş modeli bu yeni ortama yükle
model = PPO.load(model_path, env=eval_env)


episodes = 5
for ep in range(episodes):
    obs, _ = eval_env.reset()
    done = False
    total_reward = 0
    while not done:
        # deterministik=True, ajanın en olası eylemi seçmesini sağlar
        action, _states = model.predict(obs, deterministic=True)
        # Sürekli aksiyon bir vektör olduğu için .item() KULLANILMAZ
        obs, reward, terminated, truncated, info = eval_env.step(action)
        total_reward += reward
        done = terminated or truncated
    print(f"Bölüm {ep + 1}: Toplam Ödül: {total_reward:.2f}")
    time.sleep(2)  # Bölümler arasında 2 saniye bekle

eval_env.close()

print("\n--- Değerlendirme Tamamlandı ---")
print("\nEğitim sonuçlarını görmek için terminalde aşağıdaki komutu çalıştırın:")
print(f"tensorboard --logdir={log_dir}")