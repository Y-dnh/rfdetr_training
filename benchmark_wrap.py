import os
import sys
import time
from pathlib import Path

# Додаємо корінь проєкту в шлях для імпорту track
BASE_DIR = Path(__file__).parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import track

# =============================================================================
# КОНФІГУРАЦІЯ БЕНЧМАРКУ
# =============================================================================

# 1. Список моделей для тестування
# "path" — повний шлях до .engine файлу
# "size" — розмір моделі ('m', 'l' і т.д.) для правильного imgsz та завантаження
MODELS = [
    {
        "name": "rfdetr_dpsu_v8",
        "path": "D:/projects_yaroslav/rfdetr_training/runs/rfdetr_dpsu_v8/baseline/weights/inference_model.sim.engine",
        "size": "m"
    },
    {
        "name": "rfdetr_large",
        "path": "D:/projects_yaroslav/rfdetr_training/runs/rfdetr_large/baseline/weights/inference_model.sim.engine",
        "size": "l"
    },
    {
        "name": "rfdetr_medium",
        "path": "D:/projects_yaroslav/rfdetr_training/runs/rfdetr_medium/baseline/weights/inference_model.sim.engine",
        "size": "m"
    },
]

# 2. Список конфігурацій SAHI
# "use": True/False
# "size": розмір слайса (720, 1080 і т.д.)
SAHI_CONFIGS = [
    {"use": False, "size": None},
    {"use": True,  "size": 576},
    {"use": True,  "size": 720},
]

# 3. Список відео або папок з відео
VIDEOS = [
    "D:\\videos_for_test\\dpsu",
]

# Куди зберігати результати бенчмарку
BENCHMARK_OUTPUT_DIR = BASE_DIR / "tracked_videos"

# =============================================================================
# RUNNER
# =============================================================================

def main():
    print(f"\n{'='*80}")
    print(f" ЗАПУСК МУЛЬТІ-БЕНЧМАРКУ (RF-DETR + SAHI)")
    print(f"{'='*80}\n")

    for video_source in VIDEOS:
        source_path = Path(video_source)
        video_group_name = source_path.name
        
        # Збираємо всі відео з папки або беремо один файл
        if source_path.is_dir():
            video_list = track.collect_videos_from_folder(str(source_path))
        else:
            video_list = [str(source_path)]
        
        if not video_list:
            print(f">>> [WARN] У папці {video_source} не знайдено відео. Пропускаємо.")
            continue

        for model in MODELS:
            for sahi in SAHI_CONFIGS:
                sahi_label = f"sahi_{sahi['size']}" if sahi["use"] else "no_sahi"
                test_name = f"{model['name']}_{sahi_label}"
                
                # Створюємо папку для цієї конфігурації
                output_dir = BENCHMARK_OUTPUT_DIR / video_group_name / test_name
                output_dir.mkdir(parents=True, exist_ok=True)

                print(f"\n>>> [ТЕСТ] Група: {video_group_name} | Модель: {model['name']} | SAHI: {sahi_label}")
                
                # ОНОВЛЕННЯ ГЛОБАЛЬНИХ ЗМІННИХ В track.py
                track.MODEL_SIZE = model["size"]
                track.PROJECT_NAME = model["name"]
                track.USE_SAHI = sahi["use"]
                if sahi["use"]:
                    track.SAHI_SLICE_WIDTH = sahi["size"]
                    track.SAHI_SLICE_HEIGHT = sahi["size"]
                
                # Запускаємо обробку кожного відео по черзі
                for i, video_path in enumerate(video_list, 1):
                    print(f"    [{i}/{len(video_list)}] Обробка: {Path(video_path).name}")
                    
                    try:
                        start_time = time.time()
                        track.run_tracking(
                            video_input_path=video_path,
                            model_path=model["path"],
                            output_base_dir=output_dir,
                            detection_interval=track.DETECTION_INTERVAL,
                            benchmark_mode=True
                        )
                        elapsed = time.time() - start_time
                        # print(f"    --- Завершено за {elapsed:.1f} сек.")
                        
                    except Exception as e:
                        print(f"    --- [ERROR] Помилка на відео {Path(video_path).name}: {e}")
                        import traceback
                        traceback.print_exc()
                        continue

    print(f"\n{'='*80}")
    print(f" УСІ ТЕСТИ ЗАВЕРШЕНО!")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    main()
