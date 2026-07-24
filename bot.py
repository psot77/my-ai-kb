import os
import io
import asyncio
import logging
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes
from fastembed import TextEmbedding
from qdrant_client import QdrantClient
from groq import Groq

# Настройка логирования
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)

# Ключи берем из переменных окружения
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

QDRANT_URL = "https://18545c10-4b80-4ed2-9304-4ba636a29618.eu-west-1-0.aws.cloud.qdrant.io"
COLLECTION_NAME = "knowledge_base"

# Инициализация клиентов
qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, port=443, https=True, check_compatibility=False)
groq_client = Groq(api_key=GROQ_API_KEY)
embedding_model = TextEmbedding(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "👋 Здравствуйте! Я ваш виртуальный корпоративный ассистент.\n\n"
        "Задайте мне любой вопрос текстом или **отправьте голосовое сообщение**, и я найду ответ в базе знаний!"
    )

def search_rag_answer(query_text: str) -> str:
    query_vector = list(embedding_model.embed([query_text]))[0].tolist()
    
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

async def handle_text_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_text = update.message.text
    await update.message.reply_chat_action("typing")
    
    loop = asyncio.get_event_loop()
    answer = await loop.run_in_executor(None, search_rag_answer, user_text)
    await update.message.reply_text(answer)

async def handle_voice_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_chat_action("typing")
    
    # Скачивание голосового файла из Telegram
    voice_file = await context.bot.get_file(update.message.voice.file_id)
    voice_bytes = await voice_file.download_as_bytearray()
    
    audio_io = io.BytesIO(voice_bytes)
    audio_io.name = "voice.ogg"

    # Распознавание голоса через Groq Whisper API
    transcription = groq_client.audio.transcriptions.create(
        file=audio_io,
        model="whisper-large-v3-turbo",
        prompt="Запрос на русском языке по базе знаний",
        response_format="text"
    )
    prompt_text = str(transcription).strip()
    
    await update.message.reply_text(f"🎙️ *Распознано:* `{prompt_text}`", parse_mode="Markdown")
    
    # Поиск ответа
    loop = asyncio.get_event_loop()
    answer = await loop.run_in_executor(None, search_rag_answer, prompt_text)
    await update.message.reply_text(answer)

if __name__ == '__main__':
    if not TELEGRAM_BOT_TOKEN:
        print("Ошибка: TELEGRAM_BOT_TOKEN не задан!")
        exit(1)
        
    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_text_message))
    app.add_handler(MessageHandler(filters.VOICE, handle_voice_message))
    
    print("🤖 Telegram-бот успешно запущен и готов к работе!")
    app.run_polling()
