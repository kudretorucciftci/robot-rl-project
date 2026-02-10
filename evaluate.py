import imageio
import numpy as np
from stable_baselines3 import PPO
from robot_env import RobotEnv
import os

# Sürekli model için doğru dosya yolunu kullan
model_path = os.path.join("models", "ppo_robot_continuous_model.zip")
gif_path = "robot_evaluation_continuous.gif"

print("Eğitilmiş sürekli model yükleniyor...")
# Modeli yüklemek için kullanılan ortamın render özelliğine ihtiyacı yok
env_for_load = RobotEnv()

# Modelin var olup olmadığını kontrol et
if not os.path.exists(model_path):
    print(f"Hata: Model dosyası bulunamadı: {model_path}")
    print("Lütfen önce modeli eğitmek ve kaydetmek için main.py dosyasını çalıştırın.")
else:
    model = PPO.load(model_path, env=env_for_load)

    print("Değerlendirme ortamı 'rgb_array' modunda oluşturuluyor...")
    # Değerlendirme ortamı, kareleri bir dizi olarak render edebilmeli
    eval_env = RobotEnv(world_size=10.0, render_mode="rgb_array")

    images = []
    obs, _ = eval_env.reset()

    print("Bir bölüm çalıştırılıyor ve kareler yakalanıyor...")
    # Maksimum adım sayısını artıralım ki bölüm tamamlansın
    for i in range(eval_env.max_steps):
        action, _ = model.predict(obs, deterministic=True)
        # Sürekli aksiyon bir vektördür, .item() kullanılmaz
        
        obs, _, terminated, truncated, _ = eval_env.step(action)
        
        # Ortamı bir RGB dizisine render et ve listeye ekle
        img = eval_env.render()
        if img is not None:
            images.append(img)
        
        if terminated or truncated:
            print(f"Bölüm {i+1} adımda tamamlandı.")
            break
            
    eval_env.close()

    if images:
        print(f"{len(images)} adet kare {gif_path} dosyasına kaydediliyor...")
        # Yakalanan kareleri daha akıcı bir GIF olarak kaydet
        imageio.mimsave(gif_path, images, fps=30)
        print(f"Başarılı! Değerlendirme GIF'i şuraya kaydedildi: {gif_path}")
        print("Robot ajanınızı çalışırken görmek için bu dosyayı şimdi açabilirsiniz.")
    else:
        print("Hiçbir kare oluşturulmadı. Render işleminde bir sorun olabilir.")
