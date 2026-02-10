import os
import time
from stable_baselines3 import PPO
from robot_env import RobotEnv

# --- Model ve Ortam Kurulumu ---
MODEL_NAME = "ppo_robot_continuous_model.zip"
model_path = os.path.join("models", MODEL_NAME)

print("--- Test Başlatılıyor ---")

# 1. Modelin var olup olmadığını kontrol et
if not os.path.exists(model_path):
    print(f"Hata: Model dosyası bulunamadı: {model_path}")
    print("Lütfen önce modeli eğitmek için main.py dosyasını çalıştırın.")
    exit()

# 2. Değerlendirme için 'human' modunda bir ortam oluştur
print("Ortam 'human' modunda oluşturuluyor... (Birazdan bir pencere açılacak)")
try:
    # Ortamı başlat. PNG dosyası varsa onu kullanır, yoksa daire çizer.
    eval_env = RobotEnv(world_size=10.0, render_mode="human")
except Exception as e:
    print(f"Ortam oluşturulurken bir hata oluştu: {e}")
    exit()


# 3. Eğitilmiş modeli bu ortama yükle
print(f"Eğitilmiş model yükleniyor: {MODEL_NAME}")
try:
    model = PPO.load(model_path, env=eval_env)
except Exception as e:
    print(f"Model yüklenirken bir hata oluştu: {e}")
    exit()


# 4. Birden fazla bölümü çalıştır ve görselleştir
print("\n--- Test Bölümleri Çalıştırılıyor ---")
print("Robotun hareketini izlemek için açılan pencereye bakın.")
print("(Not: Pencerenin açılması birkaç saniye sürebilir)")

num_episodes_to_test = 5 # Kaç farklı senaryoda test edileceği
time.sleep(2) # Başlamadan önce kullanıcıya hazırlanma süresi

try:
    for episode_num in range(num_episodes_to_test):
        print(f"\n--- Bölüm {episode_num + 1} Başlıyor ---")
        obs, _ = eval_env.reset() # Her bölüm için ortamı sıfırla
        done = False
        total_reward = 0
        step_count = 0

        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            
            total_reward += reward
            step_count += 1
            done = terminated or truncated
            
            # Simülasyonu biraz yavaşlat ki gözle görülebilsin
            time.sleep(0.05) 

        print(f"Bölüm {episode_num + 1}: {step_count} adımda tamamlandı. Toplam Ödül: {total_reward:.2f}")
        time.sleep(2) # Bölümler arasında daha uzun bekleme

except Exception as e:
    if "Tcl_AsyncDelete: async handler deleted by the wrong thread" not in str(e):
         print(f"\nTest sırasında bir hata oluştu: {e}")
   
finally:
    print("\nPencere 5 saniye içinde kapanacak...")
    time.sleep(5)
    eval_env.close()
    print("\n--- Test Tamamlandı ---")
