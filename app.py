import uuid
import time
import hashlib
import json
import urllib.request
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

# =====================================================================
# 1. НАСТРОЙКИ КЛЮЧЕЙ И СТРАНИЦЫ
# =====================================================================
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
QDRANT_API_KEY = st.secrets["QDRANT_API_KEY"]

QDRANT_URL = "https://18545c10-4b80-4ed2-9304-4ba636a29618.eu-west-1-0.aws.cloud.qdrant.io"
COLLECTION_NAME = "knowledge_base"
LOGS_COLLECTION = "audit_logs"

st.set_page_config(page_title="Enterprise AI Knowledge Base", page_icon="🛡️", layout="wide")

def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

# =====================================================================
# 2. ФУНКЦИИ ОПРЕДЕЛЕНИЯ IP И СТРАНЫ (GeoIP)
# =====================================================================
def get_client_ip() -> str:
    """Извлечение IP-адреса пользователя из заголовков запроса Streamlit"""
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

def get_country_by_ip(ip: str) -> str:
    """Определение страны по IP через lightweight GeoIP API"""
    if ip in ["127.0.0.1", "localhost"] or ip.startswith("192.168.") or ip.startswith("10."):
        return "Локальная сеть / Dev"
    try:
        url = f"http://ip-api.com/json/{ip}?fields=country,status"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=1.5) as response:
            data = json.loads(response.read().decode())
            if data.get("status") == "success":
                return data.get("country", "Неизвестно")
    except Exception:
        pass
    return "Неизвестно"

# =====================================================================
# 3. ИНИЦИАЛИЗАЦИЯ СЕРВИСОВ С ОПТИМИЗАЦИЕЙ ПАМЯТИ
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
# 4. ФУНКЦИЯ ЛОГИРОВАНИЯ И ВСПОМОГАТЕЛЬНЫЕ МЕТОДЫ
# =====================================================================
def log_event(action: str, details: str, ip: str = None, country: str = None):
    """Запись события аудита с фиксированием IP и Страны в Qdrant"""
    try:
        user_info = st.session_state.get("current_user", {})
        username = user_info.get("username", "System")
        role = user_info.get("role", "unknown")
        
        req_ip = ip if ip else user_info.get("ip", get_client_ip())
        req_country = country if country else user_info.get("country", get_country_by_ip(req_ip))
        
        log_point = PointStruct(
            id=uuid.uuid4().hex,
            vector=[0.0] * 384,
            payload={
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "username": username,
                "role": role,
                "action": action,
                "details": details,
                "ip": req_ip,
                "country": req_country
            }
        )
        qdrant.upsert(collection_name=LOGS_COLLECTION, points=[log_point])
    except Exception:
        pass

def get_audit_logs():
    try:
        scroll_res, _ = qdrant.scroll(
            collection_name=LOGS_COLLECTION,
            limit=1000,
            with_payload=True,
            with_vectors=False
        )
        logs = [pt.payload for pt in scroll_res]
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
# 5. ИНИЦИАЛИЗАЦИЯ СЕССИИ И БД ПОЛЬЗОВАТЕЛЕЙ
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

if "sections" not in st.session_state:
    st.session_state.sections = ["Общий раздел", "Продажи и CRM", "Регламенты", "Техническая часть"]

if "projects" not in st.session_state or isinstance(st.session_state.projects, list):
    st.session_state.projects = {
        "Общий проект": ["Общий раздел"],
        "Отдел продаж": ["Продажи и CRM", "Общий раздел"],
        "IT и Разработка": ["Техническая часть", "Регламенты"]
    }

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Здравствуйте! Задайте любой вопрос по подключенной базе знаний."}
    ]

if "metrics_history" not in st.session_state:
    st.session_state.metrics_history = []

# =====================================================================
# 6. ЭКРАН ВХОДА В СИСТЕМУ (С ПРОВЕРКОЙ БЛОКИРОВОК И IP)
# =====================================================================
if not st.session_state.logged_in:
    col_l1, col_l2, col_l3 = st.columns([1, 2, 1])
    with col_l2:
        st.markdown("<h1 style='text-align: center;'>🛡️ Вход в AI Базу Знаний</h1>", unsafe_allow_html=True)
        st.caption("Корпоративная авторизация с контролем безопасности.")
        
        client_ip = get_client_ip()
        client_country = get_country_by_ip(client_ip)
        st.info(f"🌐 Ваш IP: `{client_ip}` | Страна: **{client_country}**")

        with st.form("login_form"):
            user_input = st.text_input("Логин:")
            pass_input = st.text_input("Пароль:", type="password")
            submit_login = st.form_submit_button("Войти в систему", use_container_width=True)

            if submit_login:
                clean_user = user_input.strip().lower()
                user_record = st.session_state.users_db.get(clean_user)

                if not user_record:
                    st.error("Неверный логин или пароль")
                    log_event("LOGIN_FAILED", f"Попытка входа с несуществующим логином '{clean_user}'", client_ip, client_country)
                elif user_record.get("is_blocked", False):
                    st.error("❌ Ваш аккаунт заблокирован! Обратитесь к Собственнику.")
                    log_event("LOGIN_BLOCKED", f"Попытка входа в заблокированный аккаунт '{clean_user}'", client_ip, client_country)
                elif user_record.get("active_sessions", 0) >= user_record.get("max_connections", 1):
                    st.error(f"❌ Превышен лимит одновременных подключений ({user_record['max_connections']})!")
                    log_event("LOGIN_LIMIT_EXCEEDED", f"Превышен лимит сессий для '{clean_user}'", client_ip, client_country)
                elif user_record["password"] != hash_password(pass_input):
                    user_record["failed_attempts"] = user_record.get("failed_attempts", 0) + 1
                    attempts = user_record["failed_attempts"]
                    
                    if attempts >= 3:
                        user_record["is_blocked"] = True
                        st.error("❌ Аккаунт заблокирован из-за 3 неверных попыток ввода пароля!")
                        log_event("AUTO_BLOCK", f"Автоматическая блокировка '{clean_user}' после 3 ошибок", client_ip, client_country)
                    else:
                        st.error(f"Неверный пароль! Осталось попыток: {3 - attempts}")
                        log_event("LOGIN_FAILED", f"Неверный пароль для '{clean_user}' (попытка {attempts}/3)", client_ip, client_country)
                else:
                    # Успешная авторизация
                    user_record["failed_attempts"] = 0
                    user_record["active_sessions"] = user_record.get("active_sessions", 0) + 1
                    
                    st.session_state.logged_in = True
                    st.session_state.current_user = {
                        "username": clean_user,
                        "role": user_record["role"],
                        "name": user_record["name"],
                        "ip": client_ip,
                        "country": client_country
                    }
                    log_event("LOGIN_SUCCESS", f"Успешный вход ({user_record['name']})", client_ip, client_country)
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
# 7. БОКОВАЯ ПАНЕЛЬ С ВЫХОДОМ И НАСТРОЙКАМИ
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
            new_proj_name = st.text_input("Имя проекта:")
            chosen_sections = st.multiselect(
                "Разделы:",
                options=st.session_state.sections,
                default=[st.session_state.sections[0]] if st.session_state.sections else []
            )
            if st.button("Сохранить проект", use_container_width=True):
                if new_proj_name and new_proj_name not in st.session_state.projects:
                    st.session_state.projects[new_proj_name] = chosen_sections
                    log_event("CREATE_PROJECT", f"Создан проект '{new_proj_name}': {chosen_sections}")
                    st.success(f"Проект '{new_proj_name}' создан!")
                    st.rerun()

        with st.expander("⚙️ Изменить разделы проекта"):
            updated_sections = st.multiselect(
                f"Разделы для '{selected_project}':",
                options=st.session_state.sections,
                default=active_sections
            )
            if st.button("Обновить привязку", use_container_width=True):
                st.session_state.projects[selected_project] = updated_sections
                log_event("EDIT_PROJECT", f"Обновлены разделы проекта '{selected_project}': {updated_sections}")
                st.success("Обновлено!")
                st.rerun()

    st.divider()
    
    try:
        if active_sections:
            project_filter = Filter(must=[FieldCondition(key="section", match=MatchAny(any=active_sections))])
            count_res = qdrant.count(collection_name=COLLECTION_NAME, count_filter=project_filter)
            doc_count = count_res.count
        else:
            doc_count = 0
            
        col_stat1, col_stat2 = st.columns(2)
        col_stat1.metric("Чанков", doc_count)
        col_stat2.metric("Разделов", len(active_sections))
    except Exception:
        pass

    st.divider()
    
    st.download_button(
        label="📥 Скачать историю (.md)",
        data=export_chat_history(),
        file_name=f"chat_{selected_project}.md",
        mime="text/markdown",
        use_container_width=True
    )
    
    if st.button("🗑️ Очистить диалог", use_container_width=True):
        st.session_state.messages = [
            {"role": "assistant", "content": f"Диалог очищен. Проект: '{selected_project}'."}
        ]
        st.session_state.metrics_history = []
        st.rerun()

# =====================================================================
# 8. ОСНОВНОЙ ИНТЕРФЕЙС
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
# ВКЛАДКА 1: ЧАТ
# ---------------------------------------------------------------------
with tab_dict["💬 Чат по проекту"]:
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input(f"Задайте вопрос по проекту '{selected_project}'..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        with st.chat_message("assistant"):
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
                            limit=3
                        )
                        search_results = response.points
                    except Exception:
                        response = qdrant.query_points(
                            collection_name=COLLECTION_NAME,
                            query=query_vector,
                            limit=3
                        )
                        search_results = response.points

                t_qdrant = (time.perf_counter() - t_qdrant_start) * 1000

                if not search_results:
                    answer = "В подключенных разделах нет подходящей информации."
                    st.write(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                else:
                    context_chunks = [
                        f"[Раздел: {hit.payload.get('section', 'Общий')} | Файл: {hit.payload.get('source_file', 'Документ')}]\n{hit.payload.get('text', '')}"
                        for hit in search_results
                    ]
                    context = "\n\n---\n\n".join(context_chunks)

                    llm_prompt = f"""Ты — вежливый виртуальный ассистент базы знаний проекта "{selected_project}".
Ответь на вопрос пользователя, используя ТОЛЬКО предоставленную ниже информацию.

--- ИНФОРМАЦИЯ ИЗ БАЗЫ ЗНАНИЙ ---
{context}

--- ВОПРОС ПОЛЬЗОВАТЕЛЯ ---
{prompt}

--- ОТВЕТ ---"""

                    t_llm_start = time.perf_counter()
                    res = groq_client.chat.completions.create(
                        model="llama-3.3-70b-versatile",
                        messages=[{"role": "user", "content": llm_prompt}],
                        temperature=0.2
                    )
                    t_llm = time.perf_counter() - t_llm_start
                    t_total = time.perf_counter() - t_start

                    answer = res.choices[0].message.content
                    st.write(answer)

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

                    with st.expander("📊 Метрики ответа и релевантность"):
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("Общее время", f"{t_total:.2f} сек")
                        col2.metric("Поиск Qdrant", f"{t_qdrant:.0f} мс")
                        col3.metric("Генерация LLM", f"{t_llm:.2f} сек")
                        col4.metric("Токены", res.usage.total_tokens)

                        st.markdown("---")
                        for idx, hit in enumerate(search_results, 1):
                            score_pct = round(hit.score * 100, 1)
                            src_file = hit.payload.get('source_file', 'Документ')
                            src_section = hit.payload.get('section', 'Общий')
                            st.write(f"**{idx}. [{src_section}] {src_file}** — `{score_pct}%`")

                    st.session_state.messages.append({"role": "assistant", "content": answer})

# ---------------------------------------------------------------------
# ВКЛАДКА 2: ЗАГРУЗКА ДОКУМЕНТОВ
# ---------------------------------------------------------------------
if "📁 Загрузка документов" in tab_dict:
    with tab_dict["📁 Загрузка документов"]:
        st.subheader("📁 Пополнение Базы Знаний")
        col_up1, col_up2 = st.columns([2, 1])
        
        with col_up1:
            target_section = st.selectbox("Целевой раздел:", st.session_state.sections)
        with col_up2:
            new_sec_input = st.text_input("➕ Новый раздел:")
            if st.button("Добавить раздел", use_container_width=True):
                if new_sec_input and new_sec_input not in st.session_state.sections:
                    st.session_state.sections.append(new_sec_input)
                    log_event("CREATE_SECTION", f"Создан раздел '{new_sec_input}'")
                    st.success(f"Раздел '{new_sec_input}' создан!")
                    st.rerun()

        st.divider()
        uploaded_files = st.file_uploader("Перетащите `.md` файлы:", type=["md"], accept_multiple_files=True)

        if uploaded_files and st.button(f"🚀 Загрузить файлы в '{target_section}'", use_container_width=True):
            headers_to_split_on = [("#", "Header 1"), ("##", "Header 2"), ("###", "Header 3")]
            markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on, strip_headers=False)

            all_points = []
            with st.spinner("Векторизация и отправка в облако..."):
                for file in uploaded_files:
                    file_content = file.read().decode("utf-8")
                    chunks = markdown_splitter.split_text(file_content)
                    texts = [c.page_content for c in chunks] if chunks else [file_content]
                    metadatas = [c.metadata for c in chunks] if chunks else [{}]

                    embeddings = list(embedding_model.embed(texts))

                    for idx, emb in enumerate(embeddings):
                        all_points.append(
                            PointStruct(
                                id=uuid.uuid4().hex,
                                vector=emb.tolist(),
                                payload={
                                    "text": texts[idx],
                                    "source_file": file.name,
                                    "section": target_section,
                                    **metadatas[idx]
                                }
                            )
                        )

                qdrant.upsert(collection_name=COLLECTION_NAME, points=all_points)
                log_event("UPLOAD_FILES", f"Загружено {len(uploaded_files)} файлов в раздел '{target_section}'")
                st.success("Документы векторизованы!")
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
        st.subheader("📈 Статистика использования")
        if not st.session_state.metrics_history:
            st.info("Нет данных за текущую сессию.")
        else:
            df_m = pd.DataFrame(st.session_state.metrics_history)
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Запросов", len(df_m))
            m2.metric("Токенов всего", f"{df_m['Всего токенов'].sum():,}")
            m3.metric("Средний ответ", f"{df_m['Время ответа (сек)'].mean():.2f} с")
            m4.metric("Поиск Qdrant", f"{df_m['Поиск Qdrant (мс)'].mean():.0f} мс")

            st.divider()
            st.markdown("### 📊 Расход токенов")
            st.bar_chart(df_m.set_index("Запрос №")[["Входные токены", "Выходные токены"]])
            st.markdown("### ⏱️ Динамика задержки")
            st.line_chart(df_m.set_index("Запрос №")[["Время ответа (сек)"]])

# ---------------------------------------------------------------------
# ВКЛАДКА 5: ЖУРНАЛ ЛОГОВ & БЕЗОПАСНОСТЬ (Только Owner)
# ---------------------------------------------------------------------
if "📋 Журнал логов & Безопасность" in tab_dict:
    with tab_dict["📋 Журнал логов & Безопасность"]:
        st.subheader("👑 Безопасность и Аудит (Панель Собственника)")
        
        sub_tab_logs, sub_tab_users = st.tabs(["📜 Полный Журнал Логов (GeoIP)", "👥 Управление Учётными Записями и Блокировками"])
        
        # 1. ЖУРНАЛ АУДИТА С IP И СТРАНОЙ
        with sub_tab_logs:
            st.write("Логи фиксируются в базе данных Qdrant Cloud:")
            logs_data = get_audit_logs()
            if not logs_data:
                st.info("Журнал аудита пуст.")
            else:
                df_logs = pd.DataFrame(logs_data)
                # Вывод таблицы с новыми полями IP и Country
                st.dataframe(
                    df_logs[["timestamp", "username", "role", "ip", "country", "action", "details"]], 
                    use_container_width=True
                )

        # 2. УПРАВЛЕНИЕ ПОЛЬЗОВАТЕЛЯМИ И БЛОКИРОВКАМИ
        with sub_tab_users:
            st.markdown("### 👥 Список зарегистрированных пользователей")
            
            for login_key, u_info in st.session_state.users_db.items():
                with st.expander(f"👤 **{u_info['name']}** (`{login_key}`) — Роль: `{role_badges.get(u_info['role'], u_info['role'])}`", expanded=True):
                    col_u1, col_u2, col_u3 = st.columns([2, 2, 2])
                    
                    with col_u1:
                        # Статус блокировки
                        is_blk = u_info.get("is_blocked", False)
                        st.write(f"**Статус:** {'🔴 ЗАБЛОКИРОВАН' if is_blk else '🟢 Активен'}")
                        st.write(f"**Ошибок входа:** `{u_info.get('failed_attempts', 0)} / 3`")
                    
                    with col_u2:
                        # Лимит и активные сессии
                        st.write(f"**Лимит сессий:** `{u_info.get('max_connections', 1)}`")
                        st.write(f"**Активных сессий:** `{u_info.get('active_sessions', 0)}`")

                    with col_u3:
                        # Кнопка блокировки / разблокировки
                        if is_blk:
                            if st.button("🔓 Разблокировать", key=f"unblk_{login_key}"):
                                u_info["is_blocked"] = False
                                u_info["failed_attempts"] = 0
                                log_event("UNBLOCK_USER", f"Собственник разблокировал пользователя '{login_key}'")
                                st.success("Пользователь разблокирован!")
                                st.rerun()
                        else:
                            if login_key != "owner": # Нельзя заблокировать самого собственника
                                if st.button("🔒 Заблокировать", key=f"blk_{login_key}", type="primary"):
                                    u_info["is_blocked"] = True
                                    log_event("BLOCK_USER", f"Собственник заблокировал пользователя '{login_key}'")
                                    st.success("Пользователь заблокирован!")
                                    st.rerun()

                        # Сброс ошибок ввода
                        if u_info.get("failed_attempts", 0) > 0:
                            if st.button("🔄 Сбросить счетчик ошибок", key=f"rst_{login_key}"):
                                u_info["failed_attempts"] = 0
                                st.success("Ошибки сброшены!")
                                st.rerun()

                    # Настройка лимита одновременных подключений
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
