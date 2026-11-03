"""Render a read-only HTML report from saved cards; no API or third-party packages."""

import argparse
import html
import json
import re
from pathlib import Path
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parents[1]


def escape(value):
    return html.escape(str(value), quote=True)


def render_card(card, index):
    product = card["product_name"]
    card_id = re.sub(r"[^a-z0-9]+", "-", product.lower()).strip("-") or f"card-{index}"
    generation = card["generation"]
    mode_note = "настоящий вызов API" if generation["mode"] == "live" else "локальная демонстрация без API"
    checks = card["checks"]
    cost = generation.get("usage", {}).get("cost")
    cost_text = "не указана" if cost is None else f"${cost:g}"
    status = "Нужно уточнение" if card["status"] == "needs_data" else "Черновик"
    rows = "".join(
        f'<tr><th scope="row">{escape(spec["name"])}</th><td>{escape(spec["value"])}</td></tr>'
        for spec in card["key_specs"]
    )
    queries = "".join(f'<li>{escape(query)}</li>' for query in card["search_queries"])
    tasks = "".join(f'<li>{escape(task)}</li>' for task in card["commercial"]["tasks"])
    missing = "".join(
        f'<li><strong>{escape(item["field"].removeprefix("specs."))}</strong>: {escape(item["value"])}</li>'
        for item in card["missing_fields"]
    ) or "<li>Входные поля заполнены.</li>"
    warnings = "".join(f'<p>{escape(warning)}</p>' for warning in card.get("warnings", []))
    url = card.get("source", {}).get("product_url", "")
    source = escape(url)
    if urlparse(url).scheme in {"https", "http"}:
        source = f'<a href="{escape(url)}">{escape(url)}</a>'
    return f"""<section class="card" id="{escape(card_id)}" aria-label="{escape(product)}">
      <header>
        <div class="eyebrow"><span class="brand">ПРИБОР AI</span><span>Результаты генерации</span><span class="number">{index:02d}</span></div>
        <div class="title-row"><h1>{escape(card['seo_title'])}</h1><span class="status">{status}</span></div>
        <p class="subtitle">Сохранённая карточка товара · Производитель: {escape(card['manufacturer'])}</p>
        <div class="metrics">
          <div><strong>{escape(checks['title_characters'])} <small>/ 70</small></strong><span>символов в заголовке</span></div>
          <div><strong>{escape(checks['description_characters'])}</strong><span>символов в SEO-описании</span></div>
          <div><strong>{len(card['search_queries'])}</strong><span>поисковых запросов</span></div>
          <div><strong>{escape(cost_text)}</strong><span>стоимость по ответу API</span></div>
        </div>
      </header>
      <div class="columns">
        <div class="left-column">
          <div class="block"><h2>SEO-описание</h2><p class="description">{escape(card['seo_description'])}</p></div>
          <div class="block search"><h2>Поисковые запросы</h2><ul>{queries}</ul></div>
          <div class="block manager"><h2>Для менеджера</h2><p>{escape(card['commercial']['audience'])}</p><ul>{tasks}</ul><p class="manager-note">{escape(card['commercial']['manager_note'])}</p></div>
        </div>
        <div class="right-column">
          <div class="block"><h2>Ключевые характеристики</h2><table><tbody>{rows}</tbody></table></div>
          <aside class="missing"><h2>Что уточнить</h2><ul>{missing}</ul>{warnings}<p>Неподтверждённые параметры не добавлены в карточку.</p></aside>
        </div>
      </div>
      <footer>
        <div><span>Режим</span><strong>{escape(generation['mode'])} · {mode_note}</strong></div>
        <div><span>Фактическая модель</span><strong class="model">{escape(generation['model'])}</strong></div>
        <div class="source"><span>Источник товара</span>{source}</div>
        <p class="report-note">Статический просмотр сохранённого результата · examples/cards.live.json · перед публикацией требуется проверка менеджером</p>
      </footer>
    </section>"""


CSS = """
*{box-sizing:border-box}body{margin:0;background:#edf1f6;color:#172b42;font-family:"Segoe UI",Arial,sans-serif;-webkit-font-smoothing:antialiased}
main{max-width:1440px;margin:auto;padding:32px 40px;display:grid;gap:32px}.card{background:#fff;border:1px solid #d9e2ed;border-radius:18px;padding:36px 40px;box-shadow:0 8px 32px #243c5d08}
.eyebrow{display:flex;align-items:center;gap:18px;color:#5c6f84;font-size:14px;letter-spacing:.02em}.brand{font-weight:750;color:#176052;letter-spacing:.10em}.number{margin-left:auto;color:#a1b0bf;font-weight:650;font-size:20px}
.title-row{display:flex;align-items:center;gap:24px;justify-content:space-between;margin-top:15px}h1{font-size:32px;line-height:1.2;font-weight:720;margin:0;letter-spacing:-.025em}.status{color:#80561b;background:#fff5e1;border:1px solid #f1dbae;padding:7px 12px;border-radius:6px;font-size:13px;white-space:nowrap;font-weight:600}
.subtitle{color:#64758a;margin:12px 0 22px;font-size:14px}.metrics{display:grid;grid-template-columns:repeat(4,1fr);background:#f6f9fc;border:1px solid #e1e8ef;border-radius:10px;padding:17px 0;margin-bottom:27px}.metrics>div{display:flex;flex-direction:column;gap:4px;padding:0 22px;border-right:1px solid #dde6ee}.metrics>div:last-child{border-right:0}.metrics strong{font-size:26px;font-weight:700;line-height:1.1}.metrics small{font-size:18px;color:#8293a5;font-weight:500}.metrics span{font-size:12px;color:#667b90}
.columns{display:grid;grid-template-columns:1.05fr 1fr;gap:35px}.block{margin-bottom:24px}h2{font-size:17px;line-height:1.3;margin:0 0 12px;font-weight:700;color:#1f425c}.description,p,li{font-size:15px;line-height:1.52}p{margin:0 0 11px}.description{margin:0;color:#2d4357}.search ul{display:flex;flex-wrap:wrap;gap:7px;padding:0;margin:0;list-style:none}.search li{font-size:13px;padding:5px 10px;background:#edf5f7;border:1px solid #d8e8ed;border-radius:5px;color:#295568}.manager ul{margin:8px 0 12px;padding-left:19px}.manager li{margin-bottom:6px}.manager-note{border-left:3px solid #accbc2;padding-left:12px;font-size:13px;color:#617589}.manager{margin-bottom:0}
table{width:100%;border-collapse:collapse;font-size:14px;line-height:1.4}th,td{vertical-align:top;padding:9px 0;border-bottom:1px solid #e5ebf1}th{width:43%;text-align:left;color:#61768a;font-weight:500;padding-right:19px}td{font-weight:500;color:#253d52}tr:first-child th,tr:first-child td{padding-top:0}.missing{margin-top:21px;background:#fff9ef;border:1px solid #efdfbf;border-radius:9px;padding:17px 19px}.missing h2{color:#7b5c2d;font-size:16px;margin-bottom:9px}.missing ul{margin:0 0 8px;padding-left:18px}.missing li{font-size:14px}.missing p{font-size:12px;color:#857354;margin:0}
footer{border-top:1px solid #dce6ef;margin-top:25px;padding-top:18px;display:grid;grid-template-columns:1fr 1.55fr;gap:11px 26px;font-size:12px}footer>div{display:flex;gap:9px;align-items:baseline}footer span{color:#718498;white-space:nowrap}footer strong{font-weight:550;color:#385168}.model{font-size:12px}footer a{color:#226883;text-decoration:none}.source{grid-column:1/-1}.report-note{grid-column:1/-1;font-size:11px;color:#8797a7;margin:0}
@media(max-width:900px){main{padding:16px}.card{padding:24px}.columns{grid-template-columns:1fr}.title-row{align-items:flex-start;flex-direction:column;gap:12px}h1{font-size:26px}.metrics{grid-template-columns:repeat(2,1fr);row-gap:18px}.metrics>div:nth-child(2){border:0}footer{grid-template-columns:1fr}footer>div{flex-wrap:wrap}.source a{overflow-wrap:anywhere}}
"""


def main():
    parser = argparse.ArgumentParser(description="Build a read-only HTML view from saved card results.")
    parser.add_argument("--input", type=Path, default=ROOT / "examples/cards.live.json")
    parser.add_argument("--output", type=Path, default=ROOT / "examples/result-preview.html")
    args = parser.parse_args()
    cards = json.loads(args.input.read_text(encoding="utf-8-sig"))
    content = "\n".join(render_card(card, i) for i, card in enumerate(cards, 1))
    page = f'<!doctype html><html lang="ru"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Прибор AI — результаты генерации</title><style>{CSS}</style></head><body><main>{content}</main></body></html>'
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(page, encoding="utf-8")
    print(f"Rendered {len(cards)} cards: {args.output}")


if __name__ == "__main__":
    main()
