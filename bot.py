import os
import socket
import json
import urllib.request
import logging
import io
import asyncio
import time
import requests

# Подробное логирование
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# =====================================================================
# 1. АВТОМАТИЧЕСКИЙ DNS-OVER-HTTPS (DoH) ДЛЯ ОБХОДА СБОЕВ DNS НА RENDER
# =====================================================================
DNS_CACHE = {}

def resolve_via_google_doh(hostname: str) -> str:
    """Прямое получение IP-адреса через Google DNS (8.8.8.8) без участия DNS Render"""
    if hostname in DNS_CACHE:
        return DNS_CACHE[hostname]
    try:
        url = f"https://8.8.8.8/resolve?name={hostname}&type=A"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=5) as response:
            data = json.loads(response.read().decode())
            if data.get("Status") == 0 and "Answer" in data:
                for ans in data["Answer"]:
                    if ans.get("type") == 1:  # Запись A (IPv4)
                        ip = ans.get("data")
                        DNS_CACHE[hostname] = ip
                        logger.info(f"🌐 DoH успешно распознал {hostname} -> {ip}")
                        return ip
    except Exception as e:
        logger.warning(f"Не удалось распознать {hostname} через DoH: {e}")
    return None

old_getaddrinfo = socket.getaddrinfo

def custom_getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
    try:
        return old_getaddrinfo(host, port, socket.AF_INET, type, proto, flags)
    except socket.gaierror:
        ip = resolve_via_google_doh(host)
        if ip:
            return old_getaddrinfo(ip, port, socket.AF_INET, type, proto, flags)
        raise

socket.getaddrinfo = custom_getaddrinfo

# =====================================================================
# 2. НАСТРОЙКИ КЛИЕНТОВ И ПЕРЕМЕННЫХ
# =====================================================================
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes
from qdrant_client import QdrantClient
from groq import Groq
from huggingface_hub import InferenceClient

logger.info("=== СТАРТ BOT.PY (INFERENCE CLIENT + DOH) ===")

# Токен берется из переменных Render, а если там пусто — используется ваш новый токен
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "8901191309:AAF4UuKO5RIZX7_Z2mj7PKp7K-chKZJdvE8")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")

QDRANT_URL = "https://18545c10-4b80-4ed2-9304-4ba636a29618.eu-west-1-0.aws.cloud.qdrant.io"
COLLECTION_NAME = "knowledge_base"

# =====================================================================
# 3. ПОЛУЧЕНИЕ ЭМБЕДДИНГОВ
# =====================================================================
def get_cloud_embedding(text: str) -> list:
    last_error = None
    for attempt in range(3):
        try:
            result = _hf_client.feature_extraction(
                text,
                model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
            )

            # Конвертация numpy array -> обычный список
            data = result.tolist() if hasattr(result, "tolist") else result

            # Разворачиваем вложенные списки [[...]]
            while isinstance(data, list) and len(data) > 0 and isinstance(data[0], list):
                data = data[0]

            if isinstance(data, list) and len(data) > 0 and isinstance(data[0], (int, float)):
                return [float(x) for x in data]

            raise Exception(f"Неожиданный формат ответа от HF: {data}")

        except Exception as e:
            last_error = str(e)
            logger.warning(f"Попытка {attempt + 1}/3 не удалась: {last_error}")
            time.sleep(2)

    raise Exception(f"Ошибка получения вектора: {last_error}")

# =====================================================================
# 4. ОБРАБОТЧИКИ СООБЩЕНИЙ TELEGRAM
# =====================================================================
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.info(f"Команда /start от: @{update.effective_user.username or update.effective_user.id}")
    await update.message.reply_text(
        "👋 Здравствуйте! Я ваш виртуальный корпоративный ассистент.\n\n"
        "Задайте мне любой вопрос текстом или отправьте голосовое сообщение!"
    )

def search_rag_answer(query_text: str) -> str:
    try:
        query_vector = get_cloud_embedding(query_text)
        
        response = qdrant.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=3
        )
        search_results = response.points
        max_score = max([hit.score for hit in search_results]) if search_results else 0.0

        if not search_results or max_score < 0.35:
            return "К сожалению, в корпоративной базе знаний пока нет подробных инструкций по этому вопросу."

        context_chunks = [
            f"[Файл: {hit.payload.get('source_file', 'Документ')}]\n{hit.payload.get('text', '')}"
            for hit in search_results
        ]
        context = "\n\n---\n\n".join(context_chunks)

        llm_prompt = f"""Ты — вежливый виртуальный ассистент корпоративной базы знаний.
Ответь на вопрос пользователя, используя ТОЛЬКО предоставленную ниже информацию.

--- ИНФОРМАЦИЯ ИЗ БАЗЫ ЗНАНИЙ ---
{context}

--- ВОПРОС ПОЛЬЗОВАТЕЛЯ ---
{query_text}

--- ОТВЕТ ---"""

        res = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": llm_prompt}],
            temperature=0.2
        )
        return res.choices[0].message.content
    except Exception as e:
        logger.error(f"Ошибка RAG: {e}", exc_info=True)
        return f"⚠️ Произошла ошибка при поиске: {e}"

async def handle_text_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        user_text = update.message.text
        logger.info(f"Вопрос из Telegram: '{user_text}'")
        await update.message.reply_chat_action("typing")
        
        loop = asyncio.get_running_loop()
        answer = await loop.run_in_executor(None, search_rag_answer, user_text)
        await update.message.reply_text(answer)
    except Exception as e:
        logger.error(f"Ошибка обработки текста: {e}", exc_info=True)
        await update.message.reply_text(f"⚠️ Ошибка: {e}")

async def handle_voice_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        logger.info("Обработка голосового сообщения...")
        await update.message.reply_chat_action("typing")
        
        voice_file = await context.bot.get_file(update.message.voice.file_id)
        voice_bytes = await voice_file.download_as_bytearray()
        
        audio_io = io.BytesIO(voice_bytes)
        audio_io.name = "voice.ogg"

        transcription = groq_client.audio.transcriptions.create(
            file=audio_io,
            model="whisper-large-v3-turbo",
            prompt="Запрос на русском языке по базе знаний",
            response_format="text"
        )
        prompt_text = str(transcription).strip()
        logger.info(f"Голос распознан: '{prompt_text}'")
        
        await update.message.reply_text(f"🎙️ *Распознано:* `{prompt_text}`", parse_mode="Markdown")
        
        loop = asyncio.get_running_loop()
        answer = await loop.run_in_executor(None, search_rag_answer, prompt_text)
        await update.message.reply_text(answer)
    except Exception as e:
        logger.error(f"Ошибка голосового ввода: {e}", exc_info=True)
        await update.message.reply_text(f"⚠️ Ошибка при обработке аудио: {e}")

if __name__ == '__main__':
    if not TELEGRAM_BOT_TOKEN:
        logger.critical("КРИТИЧЕСКАЯ ОШИБКА: TELEGRAM_BOT_TOKEN не задан!")
        exit(1)
        
    logger.info("Запуск слушателя Telegram...")
    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_text_message))
    app.add_handler(MessageHandler(filters.VOICE, handle_voice_message))
    
    logger.info("🤖 УСПЕХ: Бот запущен!")
    # drop_pending_updates=True сбрасывает старые подсоединения и устраняет ошибку 409 Conflict
    app.run_polling(drop_pending_updates=True)
