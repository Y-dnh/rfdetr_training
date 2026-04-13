import json
import os
import sys
import time
from datetime import datetime
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
        "name": "rfdetr_medium_overfit",
        "path": "D:/projects_yaroslav/rfdetr_training/runs/rfdetr_medium_overfit/baseline/weights/inference_model.sim.engine",
        "size": "m"
    },
        {
        "name": "rfdetr_medium",
        "path": "D:/projects_yaroslav/rfdetr_training/runs/rfdetr_medium/baseline/weights/inference_model.sim.engine",
        "size": "m"
    },
        {
        "name": "rfdetr_dpsu_v8",
        "path": "D:/projects_yaroslav/rfdetr_training/runs/rfdetr_dpsu_v8/baseline/weights/inference_model.sim.engine",
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
    "D:\\videos_for_test\\zir",
]

# Куди зберігати результати бенчмарку
BENCHMARK_OUTPUT_DIR = BASE_DIR / "tracked_videos"


# =============================================================================
# СЕРІАЛІЗАЦІЯ BENCHMARK STATS → JSON-FRIENDLY DICT
# =============================================================================

def _serialize_benchmark_stats(bs):
    """Конвертує BenchmarkStats у словник для JSON (без масивів per-frame)."""
    if bs is None or bs.n == 0:
        return None

    n = bs.n
    phases = {
        "frame_read":  bs.frame_read_times,
        "detection":   bs.detection_times,
        "tracker":     bs.tracker_update_times,
        "drawing":     bs.drawing_times,
        "frame_write": bs.frame_write_times,
    }

    phase_summary = {}
    for name, arr in phases.items():
        t_total = sum(arr)
        phase_summary[name] = {
            "total_sec": round(t_total, 4),
            "avg_ms": round(t_total / n * 1000, 2) if n > 0 else 0,
            "min_ms": round(min(arr) * 1000, 2) if arr else 0,
            "max_ms": round(max(arr) * 1000, 2) if arr else 0,
        }

    measured_total = sum(sum(arr) for arr in phases.values())

    return {
        "frames_processed": n,
        "detection_frame_count": bs.detection_frame_count,
        "tracking_only_frame_count": bs.tracking_only_frame_count,
        "total_detections": bs.total_detections,
        "total_tracks_drawn": bs.total_tracks_drawn,
        "model_load_ms": round(bs.model_load_time * 1000, 2),
        "warmup_ms": round(bs.warmup_time * 1000, 2),
        "measured_total_sec": round(measured_total, 4),
        "phases": phase_summary,
    }


def _serialize_result(r):
    """Конвертує один результат run_tracking у JSON-friendly dict."""
    return {
        "video_stem": r["video_stem"],
        "video_path": r["video_path"],
        "video_info": r.get("video_info", {}),
        "detection_config": r.get("detection_config", {}),
        "tracking_config": r.get("tracking_config", {}),
        "sahi_config": r.get("sahi_config", {}),
        "performance": {
            "elapsed_sec": round(r["elapsed_sec"], 3),
            "fps_processed": round(r["fps_processed"], 2),
            "total_detections": r.get("total_detections", 0),
            "total_tracks": r.get("total_tracks", 0),
            "benchmark": _serialize_benchmark_stats(r.get("benchmark_stats")),
        },
    }


def _generate_model_json(model_info: dict, test_entries: list, output_path: Path):
    """
    Генерує один зведений JSON-файл для моделі.
    test_entries: список dict з ключами sahi_label, video_group, results.
    """
    tests = []
    for entry in test_entries:
        sahi_label = entry["sahi_label"]
        video_group = entry["video_group"]
        results = entry["results"]

        serialized_videos = [_serialize_result(r) for r in results]

        # Агрегована статистика по групі
        total_frames = sum(r.get("frames_processed", 0) for r in results)
        total_time = sum(r.get("elapsed_sec", 0) for r in results)
        avg_fps = total_frames / total_time if total_time > 0 else 0

        tests.append({
            "sahi_label": sahi_label,
            "sahi_config": results[0].get("sahi_config", {}) if results else {},
            "video_group": video_group,
            "videos": serialized_videos,
            "aggregate": {
                "total_frames": total_frames,
                "total_time_sec": round(total_time, 3),
                "avg_fps": round(avg_fps, 2),
                "videos_count": len(results),
            },
        })

    data = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model": {
            "name": model_info["name"],
            "path": model_info["path"],
            "size": model_info["size"],
        },
        "tests": tests,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)

    print(f"\n>>> [JSON] Зведений бенчмарк моделі збережено: {output_path}")


# =============================================================================
# RUNNER
# =============================================================================

def main():
    print(f"\n{'='*80}")
    print(f" ЗАПУСК МУЛЬТІ-БЕНЧМАРКУ (RF-DETR + SAHI)")
    print(f"{'='*80}\n")

    # Збір результатів по кожній моделі: model_name -> list of test entries
    all_model_results = {}

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
                
                results = []
                
                # Запускаємо обробку кожного відео по черзі
                for i, video_path in enumerate(video_list, 1):
                    print(f"    [{i}/{len(video_list)}] Обробка: {Path(video_path).name}")
                    
                    try:
                        start_time = time.time()
                        out = track.run_tracking(
                            video_input_path=video_path,
                            model_path=model["path"],
                            output_base_dir=output_dir,
                            detection_interval=track.DETECTION_INTERVAL,
                            benchmark_mode=True
                        )
                        if out:
                            results.append(out)
                        elapsed = time.time() - start_time
                        # print(f"    --- Завершено за {elapsed:.1f} сек.")
                        
                    except Exception as e:
                        print(f"    --- [ERROR] Помилка на відео {Path(video_path).name}: {e}")
                        import traceback
                        traceback.print_exc()
                        continue

                if results:
                    track._generate_global_reports(results, output_dir)

                # Зберігаємо для загального JSON по моделі
                if results:
                    model_name = model["name"]
                    if model_name not in all_model_results:
                        all_model_results[model_name] = {"model": model, "entries": []}
                    all_model_results[model_name]["entries"].append({
                        "sahi_label": sahi_label,
                        "video_group": video_group_name,
                        "results": results,
                    })

    # Генеруємо один зведений JSON-файл для кожної моделі
    for model_name, data in all_model_results.items():
        json_path = BENCHMARK_OUTPUT_DIR / f"{model_name}_benchmark.json"
        _generate_model_json(data["model"], data["entries"], json_path)

    print(f"\n{'='*80}")
    print(f" УСІ ТЕСТИ ЗАВЕРШЕНО!")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    main()
