"""python -m pribor_ai input.json --output cards.json --markdown cards.md"""

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

from .core import InputError, PlanError, generate, markdown, parse_products
from .provider import ProviderError


def load_env(path):
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, sep, value = line.partition("=")
        if key.strip() not in {"OPENROUTER_API_KEY", "OPENROUTER_MODEL"} or not sep:
            raise InputError("В .env допустимы только OPENROUTER_API_KEY и OPENROUTER_MODEL в формате ИМЯ=значение.")
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def atomic_write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as f:
        temporary = Path(f.name)
        f.write(text)
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Черновики карточек приборов через бесплатные модели OpenRouter.")
    parser.add_argument("input", type=Path, help="JSON с товаром или массивом товаров")
    parser.add_argument("--output", type=Path, help="Файл результата JSON; иначе stdout")
    parser.add_argument("--markdown", type=Path, help="Дополнительно сохранить читаемый Markdown")
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--demo", action="store_true", help="Локальная демонстрация БЕЗ LLM, явно маркируется")
    args = parser.parse_args(argv)
    try:
        targets = [p.resolve() for p in (args.input, args.output, args.markdown, args.env_file) if p]
        if len(targets) != len(set(targets)):
            raise InputError("Входной JSON, .env и выходные файлы должны иметь разные пути.")
        if not args.demo:
            load_env(args.env_file)
        if args.input.stat().st_size > 1_000_000:
            raise InputError("Входной JSON больше 1 МБ.")
        products = parse_products(args.input.read_text(encoding="utf-8-sig"))
        cards = []
        for i, product in enumerate(products, 1):
            print(f"Товар {i}/{len(products)}: {'демонстрация' if args.demo else 'вызов OpenRouter'}…", file=sys.stderr)
            cards.append(generate(product, demo=args.demo))
        # Do not replace prior outputs with a partial batch when a later API call fails.
        result = json.dumps(cards, ensure_ascii=False, indent=2) + "\n"
        if args.output:
            atomic_write(args.output, result)
        else:
            print(result)
        if args.markdown:
            atomic_write(args.markdown, markdown(cards))
        print(f"Готово: {len(cards)} карточки. Режим: {'demo (без API)' if args.demo else 'live' }.", file=sys.stderr)
        return 0
    except (InputError, PlanError, ProviderError) as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        return 2
    except (OSError, UnicodeError):
        print("Ошибка чтения/записи файла. Проверьте пути, права доступа и кодировку UTF-8.", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
