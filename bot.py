import os
import io
import asyncio
import logging
import json
import urllib.request
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes
from qdrant_client import QdrantClient
from groq import Groq

# Подробное логирование
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("=== СТАРТ ЛЕГКОГО BOT.PY ===")

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

QDRANT_URL = "https://18545c10-4b80-4ed2-9304-4ba636a29618.eu-west-1-0.aws.cloud.qdrant.io"
COLLECTION_NAME = "knowledge_base"

# Облачные клиенты
qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, port=443, https=True, check_compatibility=False)
groq_client = Groq(api_key=GROQ_API_KEY)

def get_cloud_embedding(text: str) -> list:
    """Получение векторного представления через HuggingFace API (потребляет 0 МБ RAM сервера)"""
    url = "https://api-inference.huggingface.co/models/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    headers = {"Content-Type": "application/json"}
    payload = json.dumps({"inputs": text, "options": {"wait_for_model": True}}).encode("utf-8")
    
    req = urllib.request.Request(url, data=payload, headers=headers)
    with urllib.request.urlopen(req, timeout=12) as response:
        res = json.loads(response.read().decode("utf-8"))
        if isinstance(res, list):
            return res if isinstance(res[0], float) else res[0]
        raise Exception(f"Неожиданный ответ HF API: {res}")

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.info(f"Команда /start от: @{update.effective_user.username or update.effective_user.id}")
    await update.message.reply_text(
        "👋 Здравствуйте! Я ваш виртуальный корпоративный ассистент.\n\n"
        "Задайте мне любой вопрос текстом или отправьте голосовое сообщение!"
    )

def search_rag_answer(query_text: str) -> str:
    try:
        # Получаем вектор текста через облако (без нагрузки на память)
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
        logger.error(f"Ошибка в RAG: {e}", exc_info=True)
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
        
    logger.info("Запуск Telegram бота...")
    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_text_message))
    app.add_handler(MessageHandler(filters.VOICE, handle_voice_message))
    
    logger.info("🤖 УСПЕХ: Легкий Telegram-бот успешно запущен!")
    app.run_polling()
