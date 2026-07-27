import os
import socket
import json
import urllib.request
import logging
import io
import asyncio
import time
import uuid
import requests

# Чтение PDF и DOCX
import pypdf
import docx

# Настройка логирования
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# =====================================================================
# 1. КОНСТАНТЫ И ПАРАМЕТРЫ ИНДЕКСАЦИИ
# =====================================================================
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
VECTOR_DIM = 384
QDRANT_URL = "https://18545c10-4b80-4ed2-9304-4ba636a29618.eu-west-1-0.aws.cloud.qdrant.io"
COLLECTION_NAME = "knowledge_base"

# =====================================================================
# 2. DNS-OVER-HTTPS (DoH)
# =====================================================================
DNS_CACHE = {}

def resolve_via_google_doh(hostname: str) -> str:
    if hostname in DNS_CACHE:
        return DNS_CACHE[hostname]
    try:
        url = f"https://8.8.8.8/resolve?name={hostname}&type=A"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=5) as response:
            data = json.loads(response.read().decode())
            if data.get("Status") == 0 and "Answer" in data:
                for ans in data["Answer"]:
                    if ans.get("type") == 1:
                        ip = ans.get("data")
                        DNS_CACHE[hostname] = ip
                        return ip
    except Exception as e:
        logger.warning(f"DoH error: {e}")
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
# 3. НАСТРОЙКИ КЛИЕНТОВ
# =====================================================================
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from groq import Groq
from huggingface_hub import InferenceClient

logger.info("=== СТАРТ BOT.PY (С ДЕТАЛИЗАЦИЕЙ ЗАГРУЗКИ) ===")

def clean_env(var_name: str, fallback: str = None) -> str:
    val = os.getenv(var_name, fallback)
    if val:
        val = val.strip().strip("'").strip('"')
    return val

TELEGRAM_BOT_TOKEN = clean_env("TELEGRAM_BOT_TOKEN", "8901191309:AAF4UuKO5RIZX7_Z2mj7PKp7K-chKZJdvE8")
GROQ_API_KEY = clean_env("GROQ_API_KEY")
QDRANT_API_KEY = clean_env("QDRANT_API_KEY")
HF_TOKEN = clean_env("HF_TOKEN")

qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, port=443, https=True, check_compatibility=False)
groq_client = Groq(api_key=GROQ_API_KEY)

# =====================================================================
# 4. ВЕКТОРИЗАЦИЯ И ИЗВЛЕЧЕНИЕ ТЕКСТА
# =====================================================================
def get_cloud_embedding(text: str) -> list:
    client = InferenceClient(token=HF_TOKEN)
    last_error = None
    for attempt in range(3):
        try:
            result = client.feature_extraction(
                text,
                model=EMBEDDING_MODEL
            )
            data = result.tolist() if hasattr(result, "tolist") else result
            while isinstance(data, list) and len(data) > 0 and isinstance(data[0], list):
                data = data[0]
            if isinstance(data, list) and len(data) > 0 and isinstance(data[0], (int, float)):
                return [float(x) for x in data]
            raise Exception(f"Неожиданный формат HF: {data}")
        except Exception as e:
            last_error = str(e)
            time.sleep(2)
    raise Exception(f"Ошибка вектора: {last_error}")

def extract_text_from_docx_bytes(file_bytes: bytes) -> str:
    """Извлекает текст из абзацев и таблиц файла Word"""
    doc_obj = docx.Document(io.BytesIO(file_bytes))
    full_text = []

    for p in doc_obj.paragraphs:
        if p.text.strip():
            full_text.append(p.text.strip())

    for table in doc_obj.tables:
        for row in table.rows:
            row_text = [cell.text.strip() for cell in row.cells if cell.text.strip()]
            if row_text:
                full_text.append(" | ".join(row_text))

    return "\n".join(full_text)

def split_text_into_chunks(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    if not text:
        return []
    chunks = []
    start = 0
    text_len = len(text)
    while start < text_len:
        end = start + chunk_size
        chunk = text[start:end]
        if end < text_len and not text[end].isspace():
            last_space = chunk.rfind(' ')
            if last_space != -1:
                end = start + last_space
                chunk = text[start:end]
        chunks.append(chunk.strip())
        start += (chunk_size - overlap)
    return [c for c in chunks if len(c) > 20]

# =====================================================================
# 5. ОБРАБОТЧИКИ TELEGRAM
# =====================================================================
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "👋 Здравствуйте! Я ваш виртуальный корпоративный ассистент.\n\n"
        "📄 Отправьте мне файл (.docx, .pdf, .txt), и я добавлю его в базу знаний!\n"
        "💬 Или просто задайте мне любой вопрос."
    )

def search_rag_answer(query_text: str) -> str:
    try:
        query_vector = get_cloud_embedding(query_text)
        response = qdrant.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=5
        )
        search_results = response.points
        max_score = max([hit.score for hit in search_results]) if search_results else 0.0

        if not search_results or max_score < 0.20:
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
        await update.message.reply_chat_action("typing")
        loop = asyncio.get_running_loop()
        answer = await loop.run_in_executor(None, search_rag_answer, user_text)
        await update.message.reply_text(answer)
    except Exception as e:
        await update.message.reply_text(f"⚠️ Ошибка: {e}")

async def handle_voice_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
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
        await update.message.reply_text(f"🎙️ *Распознано:* `{prompt_text}`", parse_mode="Markdown")
        
        loop = asyncio.get_running_loop()
        answer = await loop.run_in_executor(None, search_rag_answer, prompt_text)
        await update.message.reply_text(answer)
    except Exception as e:
        await update.message.reply_text(f"⚠️ Ошибка при обработке аудио: {e}")

async def handle_document_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик документов с подробной визуализацией параметров загрузки"""
    doc = update.message.document
    file_name = doc.file_name
    file_size_kb = round(doc.file_size / 1024, 1)
    ext = os.path.splitext(file_name)[1].lower()

    if ext not in [".docx", ".pdf", ".txt"]:
        await update.message.reply_text("⚠️ Поддерживаются только форматы `.docx`, `.pdf` и `.txt`", parse_mode="Markdown")
        return

    start_time = time.time()

    # Стартовое сообщение с параметрами
    msg = await update.message.reply_text(
        f"⚙️ **Запуск индексации файла...**\n\n"
        f"📁 **Файл:** `{file_name}` ({file_size_kb} KB)\n"
        f"⚙️ **Настройки чанкинга:** `{CHUNK_SIZE}` симв. / перекрытие `{CHUNK_OVERLAP}`\n"
        f"🧠 **Модель:** `{EMBEDDING_MODEL.split('/')[-1]}` ({VECTOR_DIM} dim)\n"
        f"🗄️ **Коллекция:** `{COLLECTION_NAME}`\n\n"
        f"⏳ *Идет чтение файла и генерация векторов...*",
        parse_mode="Markdown"
    )

    try:
        telegram_file = await context.bot.get_file(doc.file_id)
        file_bytes = await telegram_file.download_as_bytearray()
        
        extracted_text = ""
        if ext == ".txt":
            extracted_text = file_bytes.decode("utf-8", errors="ignore")
        elif ext == ".docx":
            extracted_text = extract_text_from_docx_bytes(file_bytes)
        elif ext == ".pdf":
            reader = pypdf.PdfReader(io.BytesIO(file_bytes))
            pages = [page.extract_text() for page in reader.pages if page.extract_text()]
            extracted_text = "\n".join(pages)

        if not extracted_text.strip():
            await msg.edit_text(f"⚠️ Не удалось извлечь текст из файла `{file_name}`.")
            return

        chunks = split_text_into_chunks(extracted_text)
        
        points = []
        for idx, chunk in enumerate(chunks):
            vector = get_cloud_embedding(chunk)
            points.append(
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vector,
                    payload={
                        "text": chunk,
                        "source_file": file_name,
                        "chunk_index": idx
                    }
                )
            )
            await asyncio.sleep(0.04)

        qdrant.upsert(collection_name=COLLECTION_NAME, points=points)

        elapsed_time = round(time.time() - start_time, 2)

        # Финальное подробное сообщение
        await msg.edit_text(
            f"✅ **Данные успешно проиндексированы и загружены!**\n\n"
            f"📊 **Статистика и параметры загрузки:**\n"
            f"• **Файл:** `{file_name}`\n"
            f"• **Размер:** `{file_size_kb} KB` (`{ext.upper()}`)\n"
            f"• **Сгенерировано чанков:** `{len(chunks)}` шт.\n"
            f"• **Размер чанка:** `{CHUNK_SIZE}` символов (перекрытие `{CHUNK_OVERLAP}`)\n"
            f"• **Векторная модель:** `{EMBEDDING_MODEL.split('/')[-1]}` (`{VECTOR_DIM}` dim)\n"
            f"• **Хранилище:** Qdrant Cloud (`{COLLECTION_NAME}`)\n"
            f"• **Время обработки:** `{elapsed_time} сек`\n\n"
            f"💡 *Теперь можете задавать вопросы по содержанию этого документа!*",
            parse_mode="Markdown"
        )
    except Exception as e:
        logger.error(f"Ошибка загрузки файла: {e}", exc_info=True)
        await msg.edit_text(f"⚠️ Ошибка при индексации файла: {e}")

# =====================================================================
# 6. ЗАПУСК
# =====================================================================
if __name__ == '__main__':
    if not TELEGRAM_BOT_TOKEN:
        logger.critical("TELEGRAM_BOT_TOKEN не задан!")
        exit(1)
        
    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.Document.ALL, handle_document_message))
    app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_text_message))
    app.add_handler(MessageHandler(filters.VOICE, handle_voice_message))
    
    logger.info("🤖 УСПЕХ: Бот запущен!")
    app.run_polling(drop_pending_updates=True)
