"""
Повна перевірка пайплайну аугментацій: на якому етапі втрачаються трансформи.
Запуск (у тому ж середовищі, що й train/preview): python -m tests.verify_augmentation_pipeline
"""
from __future__ import annotations

import os
import sys
from typing import Any, List

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def _resolve(t: Any) -> Any:
    # Розгортати тільки фабрики (function/lambda); інстанси A.* не викликати — t() дає dict/KeyError
    if type(t).__name__ == "function":
        try:
            return t()
        except Exception as e:
            return f"<resolve_error: {e!r}>"
    return t


def _has_available_keys(t: Any) -> bool:
    return getattr(t, "available_keys", None) is not None


def _get_name(t: Any) -> str:
    if t is None:
        return "None"
    if isinstance(t, str) and t.startswith("<"):
        return t
    name = getattr(getattr(t, "__class__", None), "__name__", None)
    return name or type(t).__name__


def main() -> None:
    from rfdetr.training.albumentation_config import get_default_albu_config

    imgsz = 640
    print("=" * 70)
    print("  ВЕРИФІКАЦІЯ ПАЙПЛАЙНУ АУГМЕНТАЦІЙ")
    print("=" * 70)

    # --- ЕТАП 1: get_default_albu_config ---
    raw_list: List[Any] = get_default_albu_config(imgsz)
    n_raw = len(raw_list)
    print(f"\n[ЕТАП 1] get_default_albu_config(imgsz={imgsz})")
    print(f"  Повернуто елементів: {n_raw}")

    if n_raw == 0:
        print("  ПОМИЛКА: список порожній. Перевірте albumentation_config та import A.")
        return

    # Типи: callable vs інстанс
    n_callable = sum(1 for t in raw_list if callable(t) and not isinstance(t, type))
    n_instance = sum(1 for t in raw_list if not (callable(t) and not isinstance(t, type)))
    print(f"  З них: callable(лямбда/функція)={n_callable}, інстанс={n_instance}")

    # Після resolve: скільки успішних, скільки з available_keys
    resolved_ok: List[Any] = []
    resolved_fail: List[str] = []
    with_available_keys = 0
    names_after_resolve: List[str] = []
    for i, t in enumerate(raw_list):
        r = _resolve(t)
        if isinstance(r, str) and r.startswith("<"):
            resolved_fail.append(f"  [{i}] {_get_name(t)} -> {r}")
            continue
        if r is None:
            resolved_fail.append(f"  [{i}] {_get_name(t)} -> None")
            continue
        resolved_ok.append(r)
        names_after_resolve.append(_get_name(r))
        if _has_available_keys(r):
            with_available_keys += 1

    print(f"\n  Після resolve (лямбди -> інстанси):")
    print(f"    успішно: {len(resolved_ok)}, помилки/None: {len(resolved_fail)}")
    print(f"    з атрибутом available_keys: {with_available_keys} / {len(resolved_ok)}")
    if resolved_fail and len(resolved_fail) <= 15:
        for line in resolved_fail:
            print(line)
    elif resolved_fail:
        print(f"    (перші 10 помилок resolve:)")
        for line in resolved_fail[:10]:
            print(line)

    # Унікальні назви після resolve (без dict/function/type)
    valid_names = [n for n in names_after_resolve if n and n not in ("function", "type", "dict")]
    print(f"\n  Назви трансформ після resolve (без 'dict'/'function'/'type'): {len(valid_names)}")
    if len(valid_names) <= 25:
        print("    " + ", ".join(valid_names))
    else:
        print("    " + ", ".join(valid_names[:20]) + " ... " + ", ".join(valid_names[-5:]))

    # --- ЕТАП 2: симуляція wrapper (старий фільтр available_keys vs новий без нього) ---
    print(f"\n[ЕТАП 2] Симуляція AlbumentationsWrapper")
    # Як раніше (з фільтром available_keys): скільки б потрапило в wrapper
    would_pass_old_filter = sum(1 for r in resolved_ok if _has_available_keys(r))
    # Після виправлення (без фільтра): усі resolved_ok
    would_pass_new_filter = len(resolved_ok)
    print(f"  Якщо фільтр available_keys УВІМКНЕНО: у wrapper потрапляє {would_pass_old_filter} трансформ.")
    print(f"  Якщо фільтр available_keys ВИМКНЕНО: у wrapper потрапляє {would_pass_new_filter} трансформ.")

    if would_pass_old_filter == 0 and len(resolved_ok) > 0:
        print("  ПОМИЛКА: жоден трансформ не має available_keys (версія albumentations?).")
    elif would_pass_old_filter < len(resolved_ok):
        print(f"  УВАГА: фільтр available_keys відсікав {len(resolved_ok) - would_pass_old_filter} трансформ.")

    # --- ЕТАП 3: справжній wrapper (якщо є numpy/torch) ---
    n_wrapper = 0
    try:
        from rfdetr.training.augmentations.albumentations_wrapper import AlbumentationsWrapper
        wrapper = AlbumentationsWrapper(transforms=raw_list)
        n_wrapper = len(getattr(wrapper, "_transforms", None) or [])
        print(f"\n[ЕТАП 3] Справжній AlbumentationsWrapper(transforms=raw_list)")
        print(f"  wrapper._transforms: {n_wrapper} елементів.")
        if n_wrapper > 0:
            wrapper_names = [_get_name(t) for t in wrapper._transforms]
            print(f"  Назви: {wrapper_names[:15]}{' ...' if len(wrapper_names) > 15 else ''}")
    except Exception as e:
        print(f"\n[ЕТАП 3] Імпорт/створення wrapper пропущено (потрібні numpy/torch): {e!r}")

    print("\n" + "=" * 70)
    if n_raw > 0 and would_pass_old_filter == 0 and len(resolved_ok) > 0:
        print("  ВИСНОВОК: Пропуск на ЕТАПІ 2 — фільтр available_keys відкидав УСІ трансформи.")
        print("  Рішення: у albumentations_wrapper.py прибрано фільтр available_keys.")
    elif would_pass_old_filter < len(resolved_ok):
        print("  ВИСНОВОК: Частина трансформ втрачалася через фільтр available_keys (виправлено).")
    if n_wrapper == would_pass_new_filter and n_wrapper > 0:
        print("  Перевірка: wrapper тепер містить усі resolved трансформи.")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
