import uuid
import time
import hashlib
import json
import urllib.request
import io
import os
from datetime import datetime
import pandas as pd
import streamlit as st
from fastembed import TextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, Filter, 
    FieldCondition, MatchValue, MatchAny, PayloadSchemaType
)
from langchain_text_splitters import MarkdownHeaderTextSplitter
from groq import Groq

# Парсеры документов
from pypdf import PdfReader
from docx import Document

# Генерация PDF
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# =====================================================================
# 1. НАСТРОЙКИ КЛЮЧЕЙ, СТРАНИЦЫ И ТАЙМ-АУТА
# =====================================================================
GROQ_API_KEY = str(st.secrets.get("GROQ_API_KEY", "")).strip().strip("'").strip('"')
QDRANT_API_KEY = str(st.secrets.get("QDRANT_API_KEY", "")).strip().strip("'").strip('"')

QDRANT_URL = "https://18545c10-4b80-4ed2-9304-4ba636a29618.eu-west-1-0.aws.cloud.qdrant.io"
COLLECTION_NAME = "knowledge_base"
LOGS_COLLECTION = "audit_logs"
ANALYTICS_COLLECTION = "analytics_logs"
CONFIG_COLLECTION = "system_config"

SESSION_TIMEOUT_MINUTES = 15

st.set_page_config(page_title="Enterprise AI Knowledge Base", page_icon="🛡️", layout="wide")

st.markdown(
    """
    <style>
    div[role="dialog"], div[data-testid="stModal"] {
        display: none !important;
        visibility: hidden !important;
        pointer-events: none !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

# =====================================================================
# 2. ФУНКЦИИ ГЕНЕРАЦИИ PDF-ОТЧЕТОВ
# =====================================================================
def generate_pdf_report(project_name: str, messages: list) -> bytes:
    font_path = "DejaVuSans.ttf"
    if not os.path.exists(font_path):
        try:
            urllib.request.urlretrieve(
                "https://cdn.jsdelivr.net/font-dejavu/2.37/ttf/DejaVuSans.ttf", 
                font_path
            )
        except Exception:
            pass

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=30, leftMargin=30, topMargin=30, bottomMargin=30)
    styles = getSampleStyleSheet()

    if os.path.exists(font_path):
        pdfmetrics.registerFont(TTFont('DejaVu', font_path))
        font_name = 'DejaVu'
    else:
        font_name = 'Helvetica'

    style_title = ParagraphStyle('DocTitle', fontName=font_name, fontSize=16, leading=20, spaceAfter=12)
    style_meta = ParagraphStyle('DocMeta', fontName=font_name, fontSize=9, leading=12, textColor='#555555', spaceAfter=18)
    style_msg = ParagraphStyle('DocMsg', fontName=font_name, fontSize=10, leading=14, spaceAfter=10)

    story = []
    story.append(Paragraph(f"<b>Отчет по диалогу: {project_name}</b>", style_title))
    story.append(Paragraph(f"Дата генерации: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Сгенерировано системой Enterprise AI", style_meta))

    for msg in messages:
        role_label = "👤 <b>Пользователь</b>" if msg["role"] == "user" else "🤖 <b>AI Ассистент</b>"
        clean_content = msg["content"].replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br/>")
        story.append(Paragraph(f"{role_label}:<br/>{clean_content}", style_msg))

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()

# =====================================================================
# 3. ФУНКЦИИ ИЗВЛЕЧЕНИЯ ТЕКСТА ИЗ ФАЙЛОВ
# =====================================================================
def extract_text_from_file(uploaded_file) -> str:
    fname = uploaded_file.name.lower()
    try:
        if fname.endswith(".pdf"):
            pdf_reader = PdfReader(uploaded_file)
            text_pages = [page.extract_text() or "" for page in pdf_reader.pages]
            return "\n\n".join(text_pages)
        elif fname.endswith(".docx"):
            doc = Document(uploaded_file)
            paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
            for table in doc.tables:
                for row in table.rows:
                    row_text = [cell.text.strip() for cell in row.cells if cell.text.strip()]
                    if row_text:
                        paragraphs.append(" | ".join(row_text))
            return "\n\n".join(paragraphs)
        else:
            return uploaded_file.read().decode("utf-8", errors="ignore")
    except Exception as e:
        st.error(f"Ошибка при чтении файла {uploaded_file.name}: {e}")
        return ""

def split_text_into_chunks(text: str, chunk_size: int = 600) -> list:
    paragraphs = text.split("\n\n")
    chunks = []
    current_chunk = ""
    
    for p in paragraphs:
        p_clean = p.strip()
        if not p_clean:
            continue
        if len(current_chunk) + len(p_clean) < chunk_size:
            current_chunk += p_clean + "\n\n"
        else:
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
            current_chunk = p_clean + "\n\n"
            
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
        
    return chunks if chunks else [text]

# =====================================================================
# 4. GeoIP ФУНКЦИИ
# =====================================================================
def get_client_ip() -> str:
    try:
        if hasattr(st, "context") and hasattr(st.context, "headers"):
            headers = st.context.headers
            if "X-Forwarded-For" in headers:
                return headers["X-Forwarded-For"].split(",")[0].strip()
            if "X-Real-Ip" in headers:
                return headers["X-Real-Ip"]
    except Exception:
        pass
    return "127.0.0.1"

def get_geoip_details(ip: str) -> dict:
    default_res = {"country": "Неизвестно", "city": "Неизвестно", "lat": None, "lon": None}
    if ip in ["127.0.0.1", "localhost"] or ip.startswith("192.168.") or ip.startswith("10."):
        return {"country": "Локальная сеть", "city": "Dev", "lat": 50.4501, "lon": 30.5234}
    try:
        url = f"http://ip-api.com/json/{ip}?fields=country,city,lat,lon,status"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=1.5) as response:
            data = json.loads(response.read().decode())
            if data.get("status") == "success":
                return {
                    "country": data.get("country", "Неизвестно"),
                    "city": data.get("city", "Неизвестно"),
                    "lat": data.get("lat"),
                    "lon": data.get("lon")
                }
    except Exception:
        pass
    return default_res

# =====================================================================
# 5. ИНИЦИАЛИЗАЦИЯ СЕРВИСОВ
# =====================================================================
@st.cache_resource(max_entries=1)
def init_services():
    qdrant = QdrantClient(
        url=QDRANT_URL, 
        api_key=QDRANT_API_KEY, 
        port=443, 
        https=True, 
        check_compatibility=False
    )
    
    collections = [c.name for c in qdrant.get_collections().collections]
    
    if COLLECTION_NAME not in collections:
        qdrant.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE)
        )
    
    if LOGS_COLLECTION not in collections:
        qdrant.create_collection(
            collection_name=LOGS_COLLECTION,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE)
        )

    if ANALYTICS_COLLECTION not in collections:
        qdrant.create_collection(
            collection_name=ANALYTICS_COLLECTION,
            vectors_config=VectorParams(size=1, distance=Distance.COSINE)
        )

    if CONFIG_COLLECTION not in collections:
        qdrant.create_collection(
            collection_name=CONFIG_COLLECTION,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE)
        )

    for field in ["section", "project", "source_file"]:
        try:
            qdrant.create_payload_index(
                collection_name=COLLECTION_NAME,
                field_name=field,
                field_schema=PayloadSchemaType.KEYWORD
            )
        except Exception:
            pass
        
    groq_client = Groq(api_key=GROQ_API_KEY)
    
    embed_model = TextEmbedding(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        threads=1
    )
    return qdrant, groq_client, embed_model

qdrant, groq_client, embedding_model = init_services()

# =====================================================================
# 6. ФУНКЦИИ ЛОГИРОВАНИЯ И КОНФИГУРАЦИИ
# =====================================================================
def load_system_config():
    try:
        scroll_res, _ = qdrant.scroll(collection_name=CONFIG_COLLECTION, limit=5, with_payload=True)
        for pt in scroll_res:
            if pt.payload and "projects" in pt.payload:
                return pt.payload.get("projects"), pt.payload.get("sections")
    except Exception:
        pass
    return None, None

def save_system_config(projects, sections):
    try:
        point = PointStruct(
            id="00000000-0000-0000-0000-000000000001",
            vector=[0.0] * 384,
            payload={
                "projects": projects,
                "sections": list(set(sections))
            }
        )
        qdrant.upsert(collection_name=CONFIG_COLLECTION, points=[point])
    except Exception as e:
        print(f"Ошибка сохранения конфига: {e}")

def log_event(action: str, details: str, ip: str = None, username: str = None, role: str = None):
    try:
        user_info = st.session_state.get("current_user") or {}
        
        req_username = username if username else user_info.get("username", "Гость")
        req_role = role if role else user_info.get("role", "guest")
        req_ip = ip if ip else user_info.get("ip", get_client_ip())
        
        geo = get_geoip_details(req_ip)
        
        log_point = PointStruct(
            id=uuid.uuid4().hex,
            vector=[0.0] * 384,
            payload={
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "username": req_username,
                "role": req_role,
                "action": action,
                "details": details,
                "ip": req_ip,
                "country": geo.get("country", "Неизвестно"),
                "city": geo.get("city", "Неизвестно"),
                "lat": geo.get("lat"),
                "lon": geo.get("lon")
            }
        )
        qdrant.upsert(collection_name=LOGS_COLLECTION, points=[log_point])
    except Exception as e:
        print(f"Ошибка логирования: {e}")

def log_analytics(source: str, user_id: str, username: str, event_type: str, query: str = "", score: float = 0.0, status: str = "Успешно", details: str = ""):
    try:
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        point = PointStruct(
            id=uuid.uuid4().hex,
            vector=[0.0],
            payload={
                "timestamp": now_str,
                "source": source,
                "user_id": str(user_id),
                "username": username or "Аноним",
                "event_type": event_type,
                "query": query[:300],
                "score": float(score),
                "found_in_kb": bool(score >= 0.20),
                "status": status,
                "details": details
            }
        )
        qdrant.upsert(collection_name=ANALYTICS_COLLECTION, points=[point])
    except Exception as e:
        print(f"Ошибка логирования аналитики: {e}")

def get_audit_logs():
    try:
        scroll_res, _ = qdrant.scroll(
            collection_name=LOGS_COLLECTION,
            limit=1000,
            with_payload=True,
            with_vectors=False
        )
        logs = []
        for pt in scroll_res:
            p = pt.payload or {}
            logs.append({
                "timestamp": p.get("timestamp", ""),
                "username": p.get("username", "Неизвестно"),
                "role": p.get("role", "guest"),
                "ip": p.get("ip", "127.0.0.1"),
                "country": p.get("country", "Неизвестно"),
                "city": p.get("city", "Неизвестно"),
                "lat": p.get("lat"),
                "lon": p.get("lon"),
                "action": p.get("action", "UNKNOWN"),
                "details": p.get("details", "")
            })
        return sorted(logs, key=lambda x: x.get("timestamp", ""), reverse=True)
    except Exception:
        return []

def get_db_files_summary():
    try:
        scroll_res, _ = qdrant.scroll(
            collection_name=COLLECTION_NAME,
            limit=10000,
            with_payload=["source_file", "section"],
            with_vectors=False
        )
        files_by_section = {}
        for point in scroll_res:
            sec = point.payload.get("section", "Общий раздел")
            src = point.payload.get("source_file", "Неизвестный файл")
            if sec not in files_by_section:
                files_by_section[sec] = {}
            files_by_section[sec][src] = files_by_section[sec].get(src, 0) + 1
        return files_by_section
    except Exception:
        return {}

def export_chat_history():
    text = f"# 📝 История диалога (Проект: {st.session_state.get('selected_project', 'Общий')})\n\n"
    for msg in st.session_state.get("messages", []):
        role = "👤 **Пользователь**" if msg["role"] == "user" else "🤖 **Ассистент**"
        text += f"{role}:\n{msg['content']}\n\n---\n\n"
    return text

# =====================================================================
# 7. ИНИЦИАЛИЗАЦИЯ СЕССИИ И БД ПОЛЬЗОВАТЕЛЕЙ
# =====================================================================
if "users_db" not in st.session_state:
    st.session_state.users_db = {
        "owner": {
            "password": hash_password("owner123"), 
            "role": "owner", 
            "name": "Собственник",
            "failed_attempts": 0,
            "is_blocked": False,
            "max_connections": 5,
            "active_sessions": 0
        },
        "admin": {
            "password": hash_password("admin123"), 
            "role": "admin", 
            "name": "Администратор",
            "failed_attempts": 0,
            "is_blocked": False,
            "max_connections": 3,
            "active_sessions": 0
        },
        "user": {
            "password": hash_password("user123"),  
            "role": "user",  
            "name": "Менеджер",
            "failed_attempts": 0,
            "is_blocked": False,
            "max_connections": 1,
            "active_sessions": 0
        }
    }

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "projects" not in st.session_state or "sections" not in st.session_state:
    db_projects, db_sections = load_system_config()
    if db_projects and db_sections:
        st.session_state.projects = db_projects
        st.session_state.sections = db_sections
    else:
        st.session_state.sections = ["Общий раздел", "Продажи и CRM", "Регламенты", "Техническая часть"]
        st.session_state.projects = {
            "Общий проект": ["Общий раздел"],
            "Отдел продаж": ["Продажи и CRM", "Общий раздел"],
            "IT и Разработка": ["Техническая часть", "Регламенты"]
        }
        save_system_config(st.session_state.projects, st.session_state.sections)

try:
    existing_files_summary = get_db_files_summary()
    for sec_key in existing_files_summary.keys():
        if sec_key and sec_key not in st.session_state.sections:
            st.session_state.sections.append(sec_key)
except Exception:
    pass

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Здравствуйте! Задайте вопрос текстом или записав голос через микрофон."}
    ]

if "metrics_history" not in st.session_state:
    st.session_state.metrics_history = []

if "voice_key_counter" not in st.session_state:
    st.session_state.voice_key_counter = 0

# =====================================================================
# 8. АВТОМАТИЧЕСКИЙ ТАЙМ-АУТ СЕССИИ
# =====================================================================
if st.session_state.logged_in:
    current_time = time.time()
    last_act = st.session_state.get("last_activity_time", current_time)
    
    if (current_time - last_act) > (SESSION_TIMEOUT_MINUTES * 60):
        c_user = st.session_state.get("current_user", {})
        u_rec = st.session_state.users_db.get(c_user.get("username", ""))
        if u_rec and u_rec.get("active_sessions", 0) > 0:
            u_rec["active_sessions"] -= 1
            
        log_event("SESSION_TIMEOUT", f"Сессия автоматически завершена из-за неактивности ({SESSION_TIMEOUT_MINUTES} мин)")
        st.session_state.logged_in = False
        st.session_state.current_user = None
        st.session_state.timeout_message = f"⏳ Ваша сессия завершена автоматически из-за отсутствия активности более {SESSION_TIMEOUT_MINUTES} минут."
        st.rerun()
    else:
        st.session_state.last_activity_time = current_time

# =====================================================================
# 9. ЭКРАН ВХОДА В СИСТЕМУ
# =====================================================================
if not st.session_state.logged_in:
    col_l1, col_l2, col_l3 = st.columns([1, 2, 1])
    with col_l2:
        st.markdown("<h1 style='text-align: center;'>🛡️ Вход в AI Базу Знаний</h1>", unsafe_allow_html=True)
        st.caption("Корпоративная авторизация с контролем безопасности.")
        
        if st.session_state.get("timeout_message"):
            st.warning(st.session_state["timeout_message"])
            st.session_state["timeout_message"] = None

        client_ip = get_client_ip()
        geo_info = get_geoip_details(client_ip)
        st.info(f"🌐 Ваш IP: `{client_ip}` | Страна: **{geo_info['country']}** ({geo_info['city']})")

        with st.form("login_form"):
            user_input = st.text_input("Логин:")
            pass_input = st.text_input("Пароль:", type="password")
            submit_login = st.form_submit_button("Войти в систему", use_container_width=True)

            if submit_login:
                clean_user = user_input.strip().lower()
                user_record = st.session_state.users_db.get(clean_user)

                if not user_record:
                    st.error("Неверный логин или пароль")
                    log_event("LOGIN_FAILED", f"Попытка входа с несуществующим логином '{clean_user}'", ip=client_ip, username=clean_user, role="guest")
                elif user_record.get("is_blocked", False):
                    st.error("❌ Ваш аккаунт заблокирован! Обратитесь к Собственнику.")
                    log_event("LOGIN_BLOCKED", f"Попытка входа в заблокированный аккаунт '{clean_user}'", ip=client_ip, username=clean_user, role=user_record.get("role", "guest"))
                elif user_record.get("active_sessions", 0) >= user_record.get("max_connections", 1):
                    st.error(f"❌ Превышен лимит одновременных подключений ({user_record['max_connections']})!")
                    log_event("LOGIN_LIMIT_EXCEEDED", f"Превышен лимит сессий для '{clean_user}'", ip=client_ip, username=clean_user, role=user_record.get("role", "guest"))
                elif user_record["password"] != hash_password(pass_input):
                    user_record["failed_attempts"] = user_record.get("failed_attempts", 0) + 1
                    attempts = user_record["failed_attempts"]
                    
                    if attempts >= 3:
                        user_record["is_blocked"] = True
                        log_event("AUTO_BLOCK", f"Автоматическая блокировка аккаунта '{clean_user}' после 3 ошибок", ip=client_ip, username=clean_user, role=user_record.get("role", "guest"))
                        st.error("❌ Аккаунт заблокирован из-за 3 неверных попыток ввода пароля!")
                    else:
                        log_event("LOGIN_FAILED", f"Неверный пароль для '{clean_user}' (попытка {attempts}/3)", ip=client_ip, username=clean_user, role=user_record.get("role", "guest"))
                        st.error(f"Неверный пароль! Осталось попыток: {3 - attempts}")
                else:
                    user_record["failed_attempts"] = 0
                    user_record["active_sessions"] = user_record.get("active_sessions", 0) + 1
                    
                    st.session_state.logged_in = True
                    st.session_state.last_activity_time = time.time()
                    st.session_state.current_user = {
                        "username": clean_user,
                        "role": user_record["role"],
                        "name": user_record["name"],
                        "ip": client_ip,
                        "country": geo_info["country"],
                        "city": geo_info["city"]
                    }
                    log_event("LOGIN_SUCCESS", f"Успешный вход пользователя '{user_record['name']}'", ip=client_ip, username=clean_user, role=user_record["role"])
                    st.success("Успешная авторизация!")
                    st.rerun()

        st.divider()
        with st.expander("🔑 Демо-учётные записи"):
            st.markdown("""
            * **👑 Собственник:** `owner` | `owner123` *(Max 5 подключений)*
            * **🛠️ Администратор:** `admin` | `admin123` *(Max 3 подключения)*
            * **👤 Пользователь:** `user` | `user123` *(Max 1 подключение)*
            """)
    st.stop()

# =====================================================================
# 10. БОКОВАЯ ПАНЕЛЬ С ВЫХОДОМ И ЭКСПОРТОМ
# =====================================================================
user_data = st.session_state.current_user
user_role = user_data["role"]

role_badges = {
    "owner": "👑 Собственник",
    "admin": "🛠️ Администратор",
    "user":  "👤 Пользователь"
}

with st.sidebar:
    st.markdown(f"### {user_data['name']}")
    st.caption(f"Роль: **{role_badges.get(user_role, user_role)}**")
    st.caption(f"IP: `{user_data.get('ip', '127.0.0.1')}` ({user_data.get('country', 'Неизвестно')})")
    st.caption(f"⏱️ Автовыход при неактивности: **{SESSION_TIMEOUT_MINUTES} мин**")
    
    if st.button("🚪 Выйти из аккаунта", use_container_width=True):
        u_rec = st.session_state.users_db.get(user_data["username"])
        if u_rec and u_rec.get("active_sessions", 0) > 0:
            u_rec["active_sessions"] -= 1
            
        log_event("LOGOUT", "Выход из системы")
        st.session_state.logged_in = False
        st.session_state.current_user = None
        st.rerun()

    st.divider()
    st.header("📂 Проекты")
    
    project_names = list(st.session_state.projects.keys())
    selected_project = st.selectbox("Активный проект:", project_names)
    st.session_state.selected_project = selected_project
    
    active_sections = st.session_state.projects.get(selected_project, [])
    st.caption(f"Разделы: **{', '.join(active_sections) if active_sections else 'Нет'}**")

    if user_role in ["admin", "owner"]:
        with st.expander("➕ Создать проект"):
            new_proj_name = st.text_input("Имя проекта:", key="input_new_proj_name")
            chosen_sections = st.multiselect(
                "Разделы:",
                options=st.session_state.sections,
                default=[st.session_state.sections[0]] if st.session_state.sections else [],
                key="ms_create_project"
            )
            if st.button("Сохранить проект", use_container_width=True):
                if new_proj_name and new_proj_name not in st.session_state.projects:
                    st.session_state.projects[new_proj_name] = chosen_sections
                    save_system_config(st.session_state.projects, st.session_state.sections)
                    log_event("CREATE_PROJECT", f"Создан проект '{new_proj_name}': {chosen_sections}")
                    st.success(f"Проект '{new_proj_name}' создан!")
                    st.rerun()

        with st.expander("⚙️ Изменить разделы проекта"):
            updated_sections = st.multiselect(
                f"Разделы для '{selected_project}':",
                options=st.session_state.sections,
                default=active_sections,
                key=f"ms_edit_project_{selected_project}"
            )
            if st.button("Обновить привязку", use_container_width=True):
                st.session_state.projects[selected_project] = updated_sections
                save_system_config(st.session_state.projects, st.session_state.sections)
                log_event("EDIT_PROJECT", f"Обновлены разделы проекта '{selected_project}': {updated_sections}")
                st.success("Обновлено!")
                st.rerun()

    st.divider()
    
    try:
        total_chunks_res = qdrant.count(collection_name=COLLECTION_NAME)
        doc_count = total_chunks_res.count
            
        col_stat1, col_stat2 = st.columns(2)
        col_stat1.metric("Чанков", doc_count)
        col_stat2.metric("Разделов", len(active_sections))
    except Exception:
        pass

    st.divider()
    st.markdown("### 📥 Экспорт истории")
    
    st.download_button(
        label="📥 Скачать историю (.md)",
        data=export_chat_history(),
        file_name=f"chat_{selected_project}.md",
        mime="text/markdown",
        use_container_width=True
    )

    pdf_bytes = generate_pdf_report(selected_project, st.session_state.get("messages", []))
    st.download_button(
        label="📄 Скачать отчет (.pdf)",
        data=pdf_bytes,
        file_name=f"report_{selected_project}.pdf",
        mime="application/pdf",
        use_container_width=True
    )
    
    if st.button("🗑️ Очистить диалог", use_container_width=True):
        st.session_state.messages = [
            {"role": "assistant", "content": f"Диалог очищен. Проект: '{selected_project}'."}
        ]
        st.session_state.metrics_history = []
        st.rerun()

# =====================================================================
# 11. ОСНОВНОЙ ИНТЕРФЕЙС И ВКЛАДКИ
# =====================================================================
st.title(f"🤖 AI Ассистент — [{selected_project}]")

tab_titles = ["💬 Чат по проекту"]

if user_role in ["admin", "owner"]:
    tab_titles.extend(["📁 Загрузка документов", "🗂️ Управление файлами", "📈 Аналитика"])

if user_role == "owner":
    tab_titles.append("📋 Журнал логов & Безопасность")

tabs = st.tabs(tab_titles)
tab_dict = {title: tab for title, tab in zip(tab_titles, tabs)}

# ---------------------------------------------------------------------
# ВКЛАДКА 1: ЧАТ И ГОЛОСОВОЙ ВВОД С ФОЛЛБЭК-ПОИСКОМ
# ---------------------------------------------------------------------
with tab_dict["💬 Чат по проекту"]:
    for msg_idx, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            
            if msg["role"] == "assistant" and msg_idx > 0:
                c_fb1, c_fb2, _ = st.columns([1, 1, 10])
                user_prompt_text = st.session_state.messages[msg_idx - 1]["content"] if msg_idx > 0 else "Вопрос не найден"
                
                with c_fb1:
                    if st.button("👍", key=f"pos_{msg_idx}"):
                        log_event("FEEDBACK_POSITIVE", f"Вопрос: '{user_prompt_text}' | Отклик: Отличный ответ")
                        st.toast("Спасибо за оценку! 👍", icon="✅")
                with c_fb2:
                    if st.button("👎", key=f"neg_{msg_idx}"):
                        st.session_state[f"show_dislike_form_{msg_idx}"] = True

                if st.session_state.get(f"show_dislike_form_{msg_idx}", False):
                    with st.form(key=f"dislike_form_{msg_idx}"):
                        st.caption("📝 **Опишите, что именно не так в ответе:**")
                        user_comment = st.text_input(
                            "Замечание:", 
                            placeholder="Например: устаревший регламент, неточная формулировка...", 
                            key=f"comment_input_{msg_idx}"
                        )
                        if st.form_submit_button("Отправить отзыв", use_container_width=True):
                            comment_text = user_comment.strip() if user_comment.strip() else "Без описания"
                            full_log_details = f"Вопрос: '{user_prompt_text}' | Комментарий: '{comment_text}'"
                            log_event("FEEDBACK_NEGATIVE", full_log_details)
                            st.toast("Спасибо! Замечание сохранено и передано администраторам 📝", icon="📝")
                            st.session_state[f"show_dislike_form_{msg_idx}"] = False
                            st.rerun()

    st.markdown("---")
    c_v1, c_v2 = st.columns([2, 5])
    with c_v1:
        st.write("🎙️ **Задать вопрос голосом:**")
        audio_value = st.audio_input(
            "Запись аудио", 
            key=f"voice_recorder_{st.session_state.voice_key_counter}", 
            label_visibility="collapsed"
        )
    
    prompt = None
    
    if audio_value is not None:
        with st.spinner("🎙️ Распознавание голоса через Groq Whisper..."):
            try:
                audio_bytes = audio_value.read()
                audio_file = io.BytesIO(audio_bytes)
                audio_file.name = "audio.wav"
                
                transcription = groq_client.audio.transcriptions.create(
                    file=audio_file,
                    model="whisper-large-v3-turbo",
                    prompt="Запрос на русском языке по базе знаний",
                    response_format="text"
                )
                prompt = str(transcription).strip()
                st.session_state.voice_key_counter += 1
                log_event("VOICE_INPUT", f"Распознано: '{prompt}'")
                
                log_analytics("Web", user_data.get("username"), user_data.get("name"), "Голосовой запрос", prompt)
                st.toast(f"🎙️ Голос распознан: '{prompt}'", icon="🗣️")
            except Exception as e:
                st.error(f"Ошибка распознавания голоса: {e}")

    text_prompt = st.chat_input(f"Или введите вопрос по проекту '{selected_project}'...")
    if text_prompt:
        prompt = text_prompt

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.spinner("Поиск ответа в базе знаний..."):
            t_start = time.perf_counter()

            query_vector = list(embedding_model.embed([prompt]))[0].tolist()

            t_qdrant_start = time.perf_counter()
            search_results = []
            
            if active_sections:
                search_filter = Filter(
                    must=[FieldCondition(key="section", match=MatchAny(any=active_sections))]
                )
                try:
                    response = qdrant.query_points(
                        collection_name=COLLECTION_NAME,
                        query=query_vector,
                        query_filter=search_filter,
                        limit=5
                    )
                    search_results = response.points
                except Exception:
                    search_results = []

            if not search_results:
                try:
                    response = qdrant.query_points(
                        collection_name=COLLECTION_NAME,
                        query=query_vector,
                        limit=5
                    )
                    search_results = response.points
                except Exception:
                    search_results = []

            t_qdrant = (time.perf_counter() - t_qdrant_start) * 1000
            max_score = max([hit.score for hit in search_results]) if search_results else 0.0

            log_analytics(
                source="Web",
                user_id=user_data.get("username"),
                username=user_data.get("name"),
                event_type="Текстовый запрос",
                query=prompt,
                score=max_score,
                status="Успешно" if max_score >= 0.20 else "Не найдено в БЗ"
            )

            if not search_results or max_score < 0.20:
                answer = "К сожалению, в базе знаний пока нет подробных инструкций по этому вопросу. Запрос передан администраторам."
                st.session_state.messages.append({"role": "assistant", "content": answer})
                log_event("KNOWLEDGE_GAP", f"Вопрос без ответа: '{prompt}' (Релевантность: {max_score*100:.1f}%)")
            else:
                context_chunks = [
                    f"[Раздел: {hit.payload.get('section', 'Общий')} | Файл: {hit.payload.get('source_file', 'Документ')}]\n{hit.payload.get('text', '')}"
                    for hit in search_results
                ]
                context = "\n\n---\n\n".join(context_chunks)

                llm_prompt = f"""Ты — высококвалифицированный корпоративный AI-ассистент базы знаний проекта "{selected_project}".
Твоя задача — давать точные, профессиональные и структурированные ответы.

--- ПРАВИЛА И ОГРАНИЧЕНИЯ ---
1. **Язык ответа:** Отвечай СТРОГО на том же языке, на котором написан «ВОПРОС ПОЛЬЗОВАТЕЛЯ».
2. **Строгая точность (БЕЗ ГАЛЛЮЦИНАЦИЙ):** Используй ТОЛЬКО информацию из блока «ИНФОРМАЦИЯ ИЗ БАЗЫ ЗНАНИЙ».
3. **Форматирование:** Используй списки (`•` или `1.`), выделяй ключевые термины и цифры **жирным шрифтом**. Избегай «воды».
4. **Указание источников:** В конце ответа укажи файлы-источники.

--- ИНФОРМАЦИЯ ИЗ БАЗЫ ЗНАНИЙ ---
{context}

--- ВОПРОС ПОЛЬЗОВАТЕЛЯ ---
{prompt}

--- ОТВЕТ ---"""

                t_llm_start = time.perf_counter()
                res = groq_client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[{"role": "user", "content": llm_prompt}],
                    temperature=0.1
                )
                t_llm = time.perf_counter() - t_llm_start
                t_total = time.perf_counter() - t_start

                answer = res.choices[0].message.content
                st.session_state.messages.append({"role": "assistant", "content": answer})

                log_event("QUERY", f"Проект '{selected_project}' | Вопрос: '{prompt[:40]}...' | Токены: {res.usage.total_tokens}")

                st.session_state.metrics_history.append({
                    "Запрос №": len(st.session_state.metrics_history) + 1,
                    "Входные токены": res.usage.prompt_tokens,
                    "Выходные токены": res.usage.completion_tokens,
                    "Всего токенов": res.usage.total_tokens,
                    "Время ответа (сек)": round(t_total, 2),
                    "Поиск Qdrant (мс)": round(t_qdrant, 0),
                    "Проект": selected_project
                })

        st.rerun()

# ---------------------------------------------------------------------
# ВКЛАДКА 2: ЗАГРУЗКА ДОКУМЕНТОВ
# ---------------------------------------------------------------------
if "📁 Загрузка документов" in tab_dict:
    with tab_dict["📁 Загрузка документов"]:
        st.subheader("📁 Пополнение Базы Знаний (PDF, Word, Text, Markdown)")
        col_up1, col_up2 = st.columns([2, 1])
        
        with col_up1:
            target_section = st.selectbox("Целевой раздел:", st.session_state.sections)
        with col_up2:
            new_sec_input = st.text_input("➕ Новый раздел:")
            if st.button("Добавить раздел", use_container_width=True):
                if new_sec_input and new_sec_input not in st.session_state.sections:
                    st.session_state.sections.append(new_sec_input)
                    save_system_config(st.session_state.projects, st.session_state.sections)
                    log_event("CREATE_SECTION", f"Создан раздел '{new_sec_input}'")
                    st.success(f"Раздел '{new_sec_input}' создан!")
                    st.rerun()

        st.divider()
        uploaded_files = st.file_uploader(
            "Перетащите файлы (`.pdf`, `.docx`, `.txt`, `.md`):", 
            type=["pdf", "docx", "txt", "md"], 
            accept_multiple_files=True
        )

        if uploaded_files and st.button(f"🚀 Векторизовать и загрузить в '{target_section}'", use_container_width=True):
            markdown_splitter = MarkdownHeaderTextSplitter(
                headers_to_split_on=[("#", "Header 1"), ("##", "Header 2"), ("###", "Header 3")], 
                strip_headers=False
            )

            all_points = []
            with st.spinner("Извлечение текста, нарезка на чанки и векторизация..."):
                for file in uploaded_files:
                    fname = file.name
                    extracted_text = extract_text_from_file(file)
                    
                    if not extracted_text.strip():
                        st.warning(f"Файл '{fname}' пуст или из него не удалось извлечь текст.")
                        continue

                    if fname.lower().endswith(".md"):
                        chunks_md = markdown_splitter.split_text(extracted_text)
                        texts = [c.page_content for c in chunks_md] if chunks_md else [extracted_text]
                        metadatas = [c.metadata for c in chunks_md] if chunks_md else [{}]
                    else:
                        texts = split_text_into_chunks(extracted_text)
                        metadatas = [{}] * len(texts)

                    embeddings = list(embedding_model.embed(texts))

                    for idx, emb in enumerate(embeddings):
                        all_points.append(
                            PointStruct(
                                id=uuid.uuid4().hex,
                                vector=emb.tolist(),
                                payload={
                                    "text": texts[idx],
                                    "source_file": fname,
                                    "section": target_section,
                                    **metadatas[idx]
                                }
                            )
                        )

                if all_points:
                    qdrant.upsert(collection_name=COLLECTION_NAME, points=all_points)
                    log_event("UPLOAD_FILES", f"Загружено {len(uploaded_files)} файлов ({len(all_points)} чанков) в раздел '{target_section}'")
                    log_analytics("Web", user_data.get("username"), user_data.get("name"), "Загрузка документа", f"Файлов: {len(uploaded_files)}", score=1.0, status="Загружено")
                    
                    st.success(f"🎉 Успешно векторизовано файлов: {len(uploaded_files)} (всего {len(all_points)} чанков)!")
                    st.rerun()

# ---------------------------------------------------------------------
# ВКЛАДКА 3: УПРАВЛЕНИЕ ФАЙЛАМИ
# ---------------------------------------------------------------------
if "🗂️ Управление файлами" in tab_dict:
    with tab_dict["🗂️ Управление файлами"]:
        st.subheader("🗂️ Управление документами")
        files_by_sec = get_db_files_summary()

        if not files_by_sec:
            st.info("Файлы отсутствуют.")
        else:
            for sec_name, files_dict in files_by_sec.items():
                with st.expander(f"📁 Раздел: **{sec_name}** ({len(files_dict)} файлов)", expanded=True):
                    for fname, chunk_cnt in files_dict.items():
                        c1, c2 = st.columns([3, 1])
                        with c1:
                            st.write(f"📄 **{fname}** (`{chunk_cnt} чанков`)")
                            other_secs = [s for s in st.session_state.sections if s != sec_name]
                            if other_secs:
                                dest_s = st.selectbox("Переместить в:", other_secs, key=f"s_{sec_name}_{fname}")
                                if st.button("🚚 Переместить", key=f"m_{sec_name}_{fname}"):
                                    pts, _ = qdrant.scroll(
                                        collection_name=COLLECTION_NAME,
                                        scroll_filter=Filter(must=[
                                            FieldCondition(key="source_file", match=MatchValue(value=fname)),
                                            FieldCondition(key="section", match=MatchValue(value=sec_name))
                                        ]),
                                        limit=10000, with_payload=False, with_vectors=False
                                    )
                                    p_ids = [p.id for p in pts]
                                    if p_ids:
                                        qdrant.set_payload(collection_name=COLLECTION_NAME, payload={"section": dest_s}, points=p_ids)
                                        log_event("MOVE_FILE", f"Файл '{fname}' из '{sec_name}' в '{dest_s}'")
                                        st.success("Перемещено!")
                                        st.rerun()

                        with c2:
                            if st.button("🗑️ Удалить", key=f"d_{sec_name}_{fname}", type="primary"):
                                pts, _ = qdrant.scroll(
                                    collection_name=COLLECTION_NAME,
                                    scroll_filter=Filter(must=[
                                        FieldCondition(key="source_file", match=MatchValue(value=fname)),
                                        FieldCondition(key="section", match=MatchValue(value=sec_name))
                                    ]),
                                    limit=10000, with_payload=False, with_vectors=False
                                )
                                p_ids = [p.id for p in pts]
                                if p_ids:
                                    qdrant.delete(collection_name=COLLECTION_NAME, points_selector=p_ids)
                                    log_event("DELETE_FILE", f"Файл '{fname}' удален из '{sec_name}'")
                                    st.success("Удалено!")
                                    st.rerun()
                        st.divider()

# ---------------------------------------------------------------------
# ВКЛАДКА 4: АНАЛИТИКА
# ---------------------------------------------------------------------
if "📈 Аналитика" in tab_dict:
    with tab_dict["📈 Аналитика"]:
        st.subheader("📈 Статистика использования системы")
        
        c_ref, _ = st.columns([1, 4])
        with c_ref:
            if st.button("🔄 Обновить данные аналитики", use_container_width=True):
                st.rerun()

        try:
            scroll_res, _ = qdrant.scroll(
                collection_name=ANALYTICS_COLLECTION,
                limit=1000,
                with_payload=True,
                with_vectors=False
            )

            if scroll_res:
                analytics_data = [pt.payload for pt in scroll_res if pt.payload]
                df_a = pd.DataFrame(analytics_data)

                total_q = len(df_a)
                tg_q = len(df_a[df_a['source'] == 'Telegram']) if 'source' in df_a.columns else 0
                web_q = len(df_a[df_a['source'] == 'Web']) if 'source' in df_a.columns else 0
                
                success_count = len(df_a[df_a['found_in_kb'] == True]) if 'found_in_kb' in df_a.columns else 0
                success_pct = round((success_count / total_q) * 100, 1) if total_q > 0 else 0

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Всего обращений", total_q)
                m2.metric("Из Telegram 📱", tg_q)
                m3.metric("Из Веб-чата 🌐", web_q)
                m4.metric("Найдено в БЗ 🎯", f"{success_pct}%")

                st.divider()
                col_ch1, col_ch2 = st.columns(2)
                
                with col_ch1:
                    st.markdown("### 📱 Распределение по источникам")
                    if 'source' in df_a.columns:
                        st.bar_chart(df_a['source'].value_counts())

                with col_ch2:
                    st.markdown("### 📊 Типы запросов и действий")
                    if 'event_type' in df_a.columns:
                        st.bar_chart(df_a['event_type'].value_counts())

                st.divider()
                st.markdown("### 📜 Подробный журнал операций (Telegram & Web)")

                if 'timestamp' in df_a.columns:
                    df_a = df_a.sort_values(by='timestamp', ascending=False)

                show_cols = [c for c in ['timestamp', 'source', 'username', 'event_type', 'query', 'score', 'status'] if c in df_a.columns]
                
                st.dataframe(
                    df_a[show_cols],
                    column_config={
                        "timestamp": "Время (UTC)",
                        "source": "Источник",
                        "username": "Пользователь",
                        "event_type": "Тип действия",
                        "query": "Запрос / Файл",
                        "score": "Точность (Score)",
                        "status": "Результат"
                    },
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.info("Пока нет зафиксированных данных в аналитике Qdrant. Задайте вопрос в Telegram или Веб-чате!")

        except Exception as e:
            st.warning(f"Не удалось выгрузить данные из базы аналитики: {e}")

        # === РАСШИРЕННЫЕ МЕТРИКИ И ГРАФИКИ ТЕКУЩЕЙ СЕССИИ ===
        if st.session_state.metrics_history:
            st.divider()
            st.markdown("### ⚡ Метрики скорости и токенов текущей веб-сессии (Groq + Qdrant)")
            
            df_m = pd.DataFrame(st.session_state.metrics_history)
            
            total_reqs = len(df_m)
            total_tokens = df_m["Всего токенов"].sum()
            avg_time = round(df_m["Время ответа (сек)"].mean(), 2)
            avg_qdrant = round(df_m["Поиск Qdrant (мс)"].mean(), 0)
            
            # Расчет суточного лимита Groq (Free Tier 100,000 TPD)
            GROQ_DAILY_LIMIT = 100000
            tokens_used_pct = round((total_tokens / GROQ_DAILY_LIMIT) * 100, 2)
            tokens_remaining = GROQ_DAILY_LIMIT - total_tokens

            col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
            col_m1.metric("Запросов в сессии", total_reqs)
            col_m2.metric("Токенов за сессию", f"{total_tokens:,}")
            col_m3.metric("Остаток Groq TPD", f"{tokens_remaining:,}", f"-{tokens_used_pct}% лимита", delta_color="normal")
            col_m4.metric("Средний ответ LLM", f"{avg_time} с")
            col_m5.metric("Средний поиск Qdrant", f"{avg_qdrant:.0f} мс")

            st.caption(f"📊 Расход суточного лимита Groq: **{total_tokens:,}** из **100,000** токенов ({tokens_used_pct}%):")
            st.progress(min(total_tokens / GROQ_DAILY_LIMIT, 1.0))

            st.markdown("---")

            col_s1, col_s2 = st.columns(2)
            
            with col_s1:
                st.markdown("##### 📊 Расход токенов по запросам (Prompt vs Generation)")
                st.bar_chart(
                    df_m.set_index("Запрос №")[["Входные токены", "Выходные токены"]]
                )

            with col_s2:
                st.markdown("##### ⏱️ Динамика задержки ответа (в секундах)")
                st.line_chart(
                    df_m.set_index("Запрос №")[["Время ответа (сек)"]]
                )

            st.markdown("##### 📜 Подробный журнал сессии")
            st.dataframe(
                df_m,
                column_config={
                    "Запрос №": st.column_config.NumberColumn("№", width="small"),
                    "Входные токены": st.column_config.NumberColumn("Входные (Prompt)", format="%d"),
                    "Выходные токены": st.column_config.NumberColumn("Выходные (Gen)", format="%d"),
                    "Всего токенов": st.column_config.NumberColumn("Всего токенов", format="%d"),
                    "Время ответа (сек)": st.column_config.NumberColumn("Время (сек)", format="%.2f s"),
                    "Поиск Qdrant (мс)": st.column_config.NumberColumn("Qdrant (мс)", format="%d ms"),
                    "Проект": "Проект"
                },
                use_container_width=True,
                hide_index=True
            )

# ---------------------------------------------------------------------
# ВКЛАДКА 5: ЖУРНАЛ ЛОГОВ, КАРТА ПОДКЛЮЧЕНИЙ & БЕЗОПАСНОСТЬ
# ---------------------------------------------------------------------
if "📋 Журнал логов & Безопасность" in tab_dict:
    with tab_dict["📋 Журнал логов & Безопасность"]:
        st.subheader("👑 Безопасность и География Подключений")
        
        sub_tab_logs, sub_tab_map, sub_tab_gaps, sub_tab_users = st.tabs([
            "📜 Полный Журнал Логов", 
            "🗺️ Карта Входов (GeoIP)",
            "💡 Пробелы в знаниях & Отзывы", 
            "👥 Управление Аккаунтами"
        ])
        
        logs_data = get_audit_logs()
        df_logs_all = pd.DataFrame(logs_data) if logs_data else pd.DataFrame()

        with sub_tab_logs:
            st.write("История всех действий фиксируется в Qdrant Cloud:")
            if df_logs_all.empty:
                st.info("Журнал аудита пуст.")
            else:
                st.dataframe(
                    df_logs_all[["timestamp", "username", "role", "ip", "country", "city", "action", "details"]], 
                    use_container_width=True
                )

        with sub_tab_map:
            st.markdown("### 🗺️ Интерактивная карта геопозиций входов")
            st.caption("Отображение точек подключения пользователей на основе данных GeoIP:")
            
            if not df_logs_all.empty and "lat" in df_logs_all.columns and "lon" in df_logs_all.columns:
                df_map_data = df_logs_all.dropna(subset=["lat", "lon"]).copy()
                df_map_data["lat"] = pd.to_numeric(df_map_data["lat"], errors="coerce")
                df_map_data["lon"] = pd.to_numeric(df_map_data["lon"], errors="coerce")
                df_map_clean = df_map_data.dropna(subset=["lat", "lon"])
                
                if not df_map_clean.empty:
                    st.map(df_map_clean[["lat", "lon"]], zoom=2)
                    st.divider()
                    st.markdown("### 🌐 Распределение входов по странам и городам")
                    geo_summary = df_map_clean.groupby(["country", "city", "ip"]).size().reset_index(name="Подключений")
                    st.dataframe(geo_summary, use_container_width=True)
                else:
                    st.info("Координаты подключений пока не зафиксированы.")
            else:
                st.info("Нет данных для отображения карты.")

        with sub_tab_gaps:
            st.markdown("### 🔍 1. Вопросы, на которые AI не нашел ответа (Knowledge Gaps)")
            st.caption("Автоматически зафиксированные вопросы, где релевантность базы знаний была < 35%:")
            
            if not df_logs_all.empty and "action" in df_logs_all.columns:
                df_gaps = df_logs_all[df_logs_all["action"] == "KNOWLEDGE_GAP"]
                if df_gaps.empty:
                    st.success("🎉 Вопросов без ответа не зафиксировано.")
                else:
                    st.dataframe(
                        df_gaps[["timestamp", "username", "ip", "details"]], 
                        use_container_width=True
                    )
            else:
                st.info("Данные отсутствуют.")

            st.divider()
            st.markdown("### 👎 2. Замечания и негативные отзывы пользователей")
            if not df_logs_all.empty and "action" in df_logs_all.columns:
                df_neg = df_logs_all[df_logs_all["action"] == "FEEDBACK_NEGATIVE"]
                if df_neg.empty:
                    st.success("🎉 Замечаний от пользователей пока нет.")
                else:
                    st.dataframe(
                        df_neg[["timestamp", "username", "ip", "details"]], 
                        use_container_width=True
                    )
            else:
                st.info("Замечания отсутствуют.")

            st.divider()
            st.markdown("### 👍 3. Положительные отклики")
            if not df_logs_all.empty and "action" in df_logs_all.columns:
                df_pos = df_logs_all[df_logs_all["action"] == "FEEDBACK_POSITIVE"]
                if df_pos.empty:
                    st.info("Положительные оценки пока не поступали.")
                else:
                    st.dataframe(
                        df_pos[["timestamp", "username", "details"]], 
                        use_container_width=True
                    )

        with sub_tab_users:
            st.markdown("### 👥 Список зарегистрированных пользователей")
            
            for login_key, u_info in st.session_state.users_db.items():
                with st.expander(f"👤 **{u_info['name']}** (`{login_key}`) — Роль: `{role_badges.get(u_info['role'], u_info['role'])}`", expanded=True):
                    col_u1, col_u2, col_u3 = st.columns([2, 2, 2])
                    
                    with col_u1:
                        is_blk = u_info.get("is_blocked", False)
                        st.write(f"**Статус:** {'🔴 ЗАБЛОКИРОВАН' if is_blk else '🟢 Активен'}")
                        st.write(f"**Ошибок входа:** `{u_info.get('failed_attempts', 0)} / 3`")
                    
                    with col_u2:
                        st.write(f"**Лимит сессий:** `{u_info.get('max_connections', 1)}`")
                        st.write(f"**Активных сессий:** `{u_info.get('active_sessions', 0)}`")

                    with col_u3:
                        if is_blk:
                            if st.button("🔓 Разблокировать", key=f"unblk_{login_key}"):
                                u_info["is_blocked"] = False
                                u_info["failed_attempts"] = 0
                                log_event("UNBLOCK_USER", f"Собственник разблокировал пользователя '{login_key}'")
                                st.success("Пользователь разблокирован!")
                                st.rerun()
                        else:
                            if login_key != "owner":
                                if st.button("🔒 Заблокировать", key=f"blk_{login_key}", type="primary"):
                                    u_info["is_blocked"] = True
                                    log_event("BLOCK_USER", f"Собственник заблокировал пользователя '{login_key}'")
                                    st.success("Пользователь заблокирован!")
                                    st.rerun()

                        if u_info.get("failed_attempts", 0) > 0:
                            if st.button("🔄 Сбросить счетчик ошибок", key=f"rst_{login_key}"):
                                u_info["failed_attempts"] = 0
                                st.success("Ошибки сброшены!")
                                st.rerun()

                    new_max_conn = st.number_input(
                        "Максимум одновременных подключений:", 
                        min_value=1, 
                        max_value=20, 
                        value=u_info.get("max_connections", 1),
                        key=f"mc_{login_key}"
                    )
                    if new_max_conn != u_info.get("max_connections", 1):
                        u_info["max_connections"] = new_max_conn
                        log_event("UPDATE_CONN_LIMIT", f"Лимит сессий для '{login_key}' изменен на {new_max_conn}")
                        st.success("Лимит обновлен!")
                        st.rerun()

            st.divider()
            st.markdown("### ➕ Добавить нового пользователя")
            with st.form("add_user_form"):
                u_login = st.text_input("Логин:")
                u_name = st.text_input("ФИО / Отображаемое имя:")
                u_pass = st.text_input("Пароль:", type="password")
                u_role = st.selectbox("Роль:", ["user", "admin", "owner"])
                u_max_c = st.number_input("Лимит подключений:", min_value=1, max_value=10, value=1)
                
                if st.form_submit_button("Создать аккаунт", use_container_width=True):
                    login_clean = u_login.strip().lower()
                    if login_clean and u_pass:
                        st.session_state.users_db[login_clean] = {
                            "password": hash_password(u_pass),
                            "role": u_role,
                            "name": u_name if u_name else login_clean,
                            "failed_attempts": 0,
                            "is_blocked": False,
                            "max_connections": u_max_c,
                            "active_sessions": 0
                        }
                        log_event("CREATE_USER", f"Создан аккаунт '{login_clean}' (Роль: {u_role}, Лимит сессий: {u_max_c})")
                        st.success(f"Аккаунт '{login_clean}' успешно создан!")
                        st.rerun()
