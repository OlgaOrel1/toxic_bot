# bot.py
import os
import json
from dotenv import load_dotenv
from aiogram import Bot, Dispatcher, F
from aiogram.types import Message
from aiogram.filters import CommandStart, Command
from aiogram.enums import ParseMode
from openai import OpenAI
from collections import defaultdict
from aiogram.enums import ChatType
import re


load_dotenv()
TG_TOKEN = os.getenv("TELEGRAM_TOKEN")
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
CHAT_MODE = defaultdict(lambda: "assistive")  # per-chat: "assistive" або "strict"
STATS = defaultdict(lambda: {"seen": 0, "flagged": 0})


bot = Bot(token=TG_TOKEN)
dp = Dispatcher()
client = OpenAI(api_key=OPENAI_KEY)

TOXICITY_THRESHOLD = 0.6  # підкручуй за політикою
# --- i18n ----
I18N = {
    "uk": {
        "toxic_hdr": "⚠️ Є ризик токсичності / нездорових патернів.",
        "ok_hdr": "✅ Явної токсичності не виявлено.",
        "brief": "• Коротко:",
        "patterns": "• Патерни:",
        "softer": "• Мʼякіший варіант:",
        "reply": "• Як відповісти на це повідомлення:",
        "none": "—",
    },
    "en": {
        "toxic_hdr": "⚠️ Possible toxicity / unhealthy patterns.",
        "ok_hdr": "✅ No clear toxicity detected.",
        "brief": "• Summary:",
        "patterns": "• Patterns:",
        "softer": "• Softer alternative:",
        "reply": "• How to reply to this message:",
        "none": "—",
    },
}

PATTERN_LABELS = {
    "uk": {
        "sarcasm": "сарказм",
        "dismissive": "зневажливість",
        "condescension": "поблажливий тон",
        "humiliation": "приниження",
        "passive_aggression": "пасивна агресія",
        "manipulation": "маніпуляція",
        "gaslighting": "газлайтинг",
    },
    "en": {
        "sarcasm": "sarcasm",
        "dismissive": "dismissive",
        "condescension": "condescension",
        "humiliation": "humiliation",
        "passive_aggression": "passive aggression",
        "manipulation": "manipulation",
        "gaslighting": "gaslighting",
    },
}

# --- ZONING ---
Z_RED_SCORES = {           # дуже високі бали -> червона
    "insult": 0.75,
    "toxicity": 0.75,
}
Z_RED_CRITICAL = {         # критичні категорії -> червона
    "threat": 0.30,
    "hate": 0.50,
    "sexual_harassment": 0.50,
}
Z_ORANGE_SCORES = {        # середні бали -> помаранчева
    "insult": 0.40,
    "toxicity": 0.40,
    "harassment": 0.40,
}

Z_RED_PATTERNS = {"humiliation", "gaslighting", "manipulation"}  # важкі патерни
Z_ORANGE_PATTERNS = {"sarcasm", "dismissive", "condescension", "passive_aggression"}  # м’якші ризики

I18N["uk"].update({
    "zone_green": "🟢 Зелена зона",
    "zone_orange": "🟠 Помаранчева зона",
    "zone_red": "🔴 Червона зона",
})
I18N["en"].update({
    "zone_green": "🟢 Green zone",
    "zone_orange": "🟠 Orange zone",
    "zone_red": "🔴 Red zone",
})

def pick_lang(code: str | None) -> str:
    code = (code or "").lower()
    if code.startswith("uk"): return "uk"
    if code.startswith("en"): return "en"
    # fallback
    return "uk"

def check_moderation(text: str):
    resp = client.moderations.create(
        model="omni-moderation-latest",
        input=text
    )
    r = resp.results[0]
    return {"flagged": r.flagged, "categories": r.categories}


def decide_zone(detail: dict, mod_flagged: bool) -> tuple[str, list[str]]:
    """
    Повертає ("green"|"orange"|"red", reasons[])
    """
    reasons = []

    # 0) Якщо Moderation вже позначив — червона
    if mod_flagged:
        return "red", ["moderation flagged"]

    # 1) Червона за критичними категоріями
    for k, th in Z_RED_CRITICAL.items():
        v = float(detail.get(k, 0) or 0)
        if v >= th:
            reasons.append(f"{k}>={th}")
            return "red", reasons

    # 2) Червона за дуже високими балами образ/токсичності
    for k, th in Z_RED_SCORES.items():
        v = float(detail.get(k, 0) or 0)
        if v >= th:
            reasons.append(f"{k}>={th}")
            return "red", reasons

    # 3) Червона за важкими патернами разом із помірною інтенсивністю
    pats = set(detail.get("unhealthy_patterns") or [])
    if pats & Z_RED_PATTERNS:
        # якщо є хоча б середня інтенсивність (будь-який із цих балах >= 0.4)
        if any(float(detail.get(k, 0) or 0) >= 0.4 for k in ["toxicity", "insult", "harassment"]):
            reasons.append(f"patterns:{','.join(sorted(pats & Z_RED_PATTERNS))}")
            return "red", reasons

    # 4) Помаранчева за середніми балами
    orange_hit = False
    for k, th in Z_ORANGE_SCORES.items():
        v = float(detail.get(k, 0) or 0)
        if v >= th:
            orange_hit = True
            reasons.append(f"{k}>={th}")

    # 5) Помаранчева за «м’якшими» патернами (навіть якщо бали низькі)
    if pats & Z_ORANGE_PATTERNS:
        orange_hit = True
        reasons.append(f"patterns:{','.join(sorted(pats & Z_ORANGE_PATTERNS))}")

    if orange_hit:
        return "orange", reasons

    # 6) Інакше зелена
    return "green", ["no clear toxicity"]

def _lang_name(lang: str | None) -> str:
    return {"uk": "Ukrainian", "en": "English"}.get((lang or "").lower(), "Ukrainian")

def _looks_like_lang(text: str, lang: str | None) -> bool:
    if not lang:
        return True
    has_cyr = re.search(r"[\u0400-\u04FF]", text) is not None
    has_lat = re.search(r"[A-Za-z]", text) is not None
    if lang == "uk":
        return has_cyr  # українська — кирилиця
    if lang == "en":
        return has_lat and not has_cyr  # англ — латинка без кирилиці
    return True

def rephrase_non_toxic(text: str, lang: str | None = None) -> str:
    """
    Перефраз тим самим змістом, але нетоксично.
    Гарантуємо мову через явну інструкцію + 1 ретрай, якщо скрипт не збігається.
    """
    ln = _lang_name(lang)
    sys = (
        f"You rewrite messages into respectful, non-violent language. "
        f"Answer ONLY in {ln}. Do NOT use any other language. "
        f"Return ONLY the rewritten sentence, WITHOUT quotation marks."
    )
    usr = (
        f"Rewrite the following message in {ln}, preserving the intent, "
        f"keeping emojis/punctuation if appropriate:\n\n{text}"
    )

    def _call(temp: float):
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=temp,
            messages=[{"role": "system", "content": sys},
                      {"role": "user",   "content": usr}],
        )
        return resp.choices[0].message.content.strip().strip('“”"')

    out = _call(0.1)
    if not _looks_like_lang(out, lang):
        # жорсткіша вказівка і повтор
        sys_retry = sys + " STRICT: Use EXACTLY this language. If you cannot, output nothing."
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
            messages=[{"role": "system", "content": sys_retry},
                      {"role": "user",   "content": usr}],
        )
        out = resp.choices[0].message.content.strip().strip('“”"')
    return out

def classify_unhealthy(text: str) -> dict:
    system = (
        "You are a communication classifier. Detect the input language and ALWAYS respond in that language. "
        "Return ONLY JSON with keys: "
        "{\"lang\": \"bcp47 like 'uk' or 'en'\", "
        "\"toxicity\": float, \"insult\": float, \"harassment\": float, \"hate\": float, "
        "\"sexual_harassment\": float, \"threat\": float, "
        "\"unhealthy_patterns\": [\"sarcasm\",\"dismissive\",\"condescension\",\"humiliation\","
        "\"passive_aggression\",\"manipulation\",\"gaslighting\"], "
        "\"summary\": \"1–2 sentence neutral summary in the input language\"} "
        "Scoring rubric (0..1, round to 0.1): "
        "- Direct insult or condescending rhetorical question (e.g., \"you are stupid\", "
        "\"you don't understand simple things\") → insult ≥ 0.6, toxicity ≥ 0.6."
    )
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": f"Text:\n{text}"},
        ],
        response_format={"type": "json_object"}
    )
    import json
    return json.loads(resp.choices[0].message.content)



def suggest_reply(text: str, flagged: bool, lang: str | None = None) -> str:
    """
    Коротка відповідь тією самою мовою.
    - flagged=True  → деескалація + межі + конструктивний крок.
    - flagged=False → кооперативна/інформативна відповідь без оборонного тону.
    Гарантуємо мову через явну інструкцію + 1 ретрай.
    """
    ln = _lang_name(lang)

    base_sys = (
        f"You craft concise replies. Answer ONLY in {ln}. Do NOT use any other language. "
        "Return ONLY the reply text, WITHOUT quotation marks or extra commentary. "
        "Keep it to 1–2 sentences. Match the input's tone/emoticons when appropriate."
    )

    if flagged:
        user_msg = (
            "Write a brief, respectful reply to the message below. "
            "Goals: de-escalate; use I-statements; set a clear boundary if there is an insult; "
            "offer a constructive next step (clarify, pause, or move to a calmer channel). "
            "Do NOT moralize, do NOT attack back.\n\n"
            f"Message:\n{text}\n"
            f"Language hint: {ln}"
        )
        temp = 0.2
    else:
        user_msg = (
            "Write a brief, cooperative reply to the message below. "
            "Assume positive intent. Be informative and friendly. "
            "If it is a question—answer directly; if it is a request—acknowledge and confirm the next step. "
            "Avoid defensive phrasing.\n\n"
            f"Message:\n{text}\n"
            f"Language hint: {ln}"
        )
        temp = 0.15

    def _call(sys_msg: str, temperature: float) -> str:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=temperature,
            messages=[
                {"role": "system", "content": sys_msg},
                {"role": "user",   "content": user_msg},
            ],
        )
        return resp.choices[0].message.content.strip().strip('“”"')

    out = _call(base_sys, temp)

    # якщо модель дала не ту мову — один жорсткіший ретрай
    if not _looks_like_lang(out, lang):
        sys_retry = base_sys + " STRICT: Use EXACTLY this language. If you cannot, output nothing."
        out = _call(sys_retry, 0)

    # останній страховочний fallback на випадок порожнього або знову не того алфавіту
    if not out or not _looks_like_lang(out, lang):
        FALLBACK = {
            "uk": "Дякую за повідомлення. Давай обговоримо це спокійно, щоб краще зрозуміти одне одного.",
            "en": "Thanks for your message. Let’s discuss this calmly so we can better understand each other.",
        }
        out = FALLBACK.get((lang or "uk").lower(), FALLBACK["uk"])

    return out



@dp.message(Command("mode"))
async def set_mode(m: Message):
    parts = (m.text or "").split()
    if len(parts) >= 2 and parts[1].lower() in {"assistive", "strict"}:
        CHAT_MODE[m.chat.id] = parts[1].lower()
        await m.answer(f"Mode set to: {CHAT_MODE[m.chat.id]}")
    else:
        await m.answer("Usage: /mode assistive | strict")

@dp.message(Command("threshold"))
async def set_threshold(m: Message):
    global TOXICITY_THRESHOLD
    parts = (m.text or "").split()
    if len(parts) >= 2:
        try:
            val = float(parts[1])
            TOXICITY_THRESHOLD = max(0.0, min(1.0, val))
            await m.answer(f"Threshold set to {TOXICITY_THRESHOLD}")
        except:
            await m.answer("Usage: /threshold 0.0..1.0")
    else:
        await m.answer(f"Current threshold: {TOXICITY_THRESHOLD}")

@dp.message(Command("stats"))
async def stats(m: Message):
    s = STATS[m.chat.id]
    rate = (s["flagged"] / s["seen"] * 100) if s["seen"] else 0.0
    await m.answer(f"Seen: {s['seen']}, flagged: {s['flagged']} ({rate:.1f}%). Mode: {CHAT_MODE[m.chat.id]}")

@dp.message(CommandStart())
async def start(m: Message):
    await m.answer(
        "Привіт! Надішли мені текст — я перевірю на токсичність і запропоную мʼякіший варіант ✨\n"
        "Команди: /help"
    )

@dp.message(Command("help"))
async def help_cmd(m: Message):
    await m.answer(
        "Просто надішли будь-який текст. Я перевірю його в два етапи:\n"
        "1) Швидка модерація (омні-модель OpenAI)\n"
        "2) Тонкий аналіз патернів + мʼякий перефраз"
    )
@dp.message(F.text)
async def analyze(m: Message):
    text = (m.text or "").strip()
    STATS[m.chat.id]["seen"] += 1

    # 1) Moderation
    mod = check_moderation(text)

    # 2) Детальний аналіз (із визначенням мови)
    detail = classify_unhealthy(text)

    # (опційно) підлога політик/регекси
    if 'apply_policy_floors' in globals():
        try:
            detail = apply_policy_floors(text, detail)
        except Exception:
            pass

    # Мова та локалізація
    lang = pick_lang(detail.get("lang"))
    t = I18N.get(lang, I18N["uk"])
    pat_map = PATTERN_LABELS.get(lang, PATTERN_LABELS["uk"])

    # Дані для відображення
    summary = detail.get("summary", t["none"])
    raw_patterns = detail.get("unhealthy_patterns") or []
    patterns_txt = ", ".join(pat_map.get(p, p) for p in raw_patterns) if raw_patterns else t["none"]

    # 3) Визначення зони (зелена/помаранчева/червона)
    zone, _reasons = decide_zone(detail, mod["flagged"])
    flagged = (zone != "green")  # для сумісності з рештою логіки

    zone_hdr = {
        "green": I18N[lang]["zone_green"],
        "orange": I18N[lang]["zone_orange"],
        "red": I18N[lang]["zone_red"],
    }[zone]

    # 4) Генерація варіантів текстів (тією ж мовою)
    softer = rephrase_non_toxic(text, lang)
    reply_to_msg = suggest_reply(text, flagged, lang)

    STATS[m.chat.id]["flagged"] += int(bool(flagged))

    # --- Відповідь залежно від типу чату і режиму ---
    is_group = m.chat.type in {ChatType.GROUP, ChatType.SUPERGROUP}
    mode = CHAT_MODE[m.chat.id]

    # Технічний короткий dbg
    dbg = (
        f"\n\n<code>zone={zone}, flagged={flagged}, "
        f"tox={detail.get('toxicity')}, insult={detail.get('insult')}, "
        f"harassment={detail.get('harassment')}, threat={detail.get('threat')}</code>"
    )

    if not is_group:
        header = t["toxic_hdr"] if flagged else t["ok_hdr"]
        body = (
            f"{zone_hdr}\n"
            f"{header}\n\n"
            f"{t['brief']} {summary}\n"
            f"{t['patterns']} {patterns_txt}\n"
            f"{t['softer']}\n{softer}\n"
            f"{t['reply']}\n{reply_to_msg}"
        )
        await m.answer(body + dbg, parse_mode=ParseMode.HTML)
        return

    # --- ГРУПИ ---
    if mode == "strict":
        if zone == "red":
            # Спроба видалити токсичне (потрібні адмін-права + privacy off у BotFather)
            try:
                await m.delete()
                warn = (
                    f"{zone_hdr}\n"
                    f"{t['toxic_hdr']}\n"
                    f"{t['brief']} {summary}\n"
                    f"{t['patterns']} {patterns_txt}\n"
                    f"{t['softer']}\n{softer}\n"
                    f"{t['reply']}\n{reply_to_msg}"
                )
                await bot.send_message(chat_id=m.chat.id, text=warn, parse_mode=ParseMode.HTML)
            except Exception:
                await m.reply(
                    f"{zone_hdr}\n{t['toxic_hdr']}\n"
                    f"{t['softer']}\n{softer}\n"
                    f"{t['reply']}\n{reply_to_msg}",
                    parse_mode=ParseMode.HTML
                )
        elif zone == "orange":
            await m.reply(
                f"{zone_hdr}\n{t['toxic_hdr']}\n\n"
                f"{t['brief']} {summary}\n"
                f"{t['patterns']} {patterns_txt}\n"
                f"{t['softer']}\n{softer}\n"
                f"{t['reply']}\n{reply_to_msg}",
                parse_mode=ParseMode.HTML
            )
        else:
            # green — щоб не засмічувати групу, мовчимо
            pass
    else:
        # assistive: відповідаємо лише коли є ризик
        if zone in ("orange", "red"):
            await m.reply(
                f"{zone_hdr}\n{t['toxic_hdr']}\n\n"
                f"{t['brief']} {summary}\n"
                f"{t['patterns']} {patterns_txt}\n"
                f"{t['softer']}\n{softer}\n"
                f"{t['reply']}\n{reply_to_msg}",
                parse_mode=ParseMode.HTML
            )
        else:
            pass  # green — тихо

if __name__ == "__main__":
    import asyncio, logging
    logging.basicConfig(level=logging.INFO)

    async def main():
        # гарантуємо, що long polling отримує апдейти
        await bot.delete_webhook(drop_pending_updates=True)
        await dp.start_polling(bot)

    asyncio.run(main())

