# webhook_server.py
import os, logging
from fastapi import FastAPI, Request
from aiogram.types import Update
from bot_toxicity_guard import bot, dp  # імпортуємо ТВОГО бота і маршрути

logging.basicConfig(level=logging.INFO)

WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET", "supersecret123")
PUBLIC_URL = os.getenv("PUBLIC_URL")  # типу https://<your-service>.onrender.com

app = FastAPI()

@app.on_event("startup")
async def on_startup():
    # Якщо відомий публічний URL — ставимо вебхук автоматично
    if PUBLIC_URL:
        url = PUBLIC_URL.rstrip("/") + f"/webhook/{WEBHOOK_SECRET}"
        await bot.set_webhook(url=url, drop_pending_updates=True, allowed_updates=["message"])
        logging.info(f"Webhook set to {url}")
    else:
        logging.warning("PUBLIC_URL is not set; set webhook manually after deploy.")

@app.post("/webhook/{secret}")
async def telegram_webhook(secret: str, request: Request):
    if secret != WEBHOOK_SECRET:
        return {"ok": False}
    data = await request.json()
    update = Update.model_validate(data)  # pydantic v2
    await dp.feed_update(bot, update)
    return {"ok": True}

@app.get("/health")
async def health():
    return {"status": "ok"}
