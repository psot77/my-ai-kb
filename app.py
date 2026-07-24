import uuid
import time
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

st.set_page_config(page_title="Модульная База Знаний AI", page_icon="📚", layout="wide")

# =====================================================================
# 2. ИНИЦИАЛИЗАЦИЯ СЕРВИСОВ С ОПТИМИЗАЦИЕЙ ПАМЯТИ (RAM)
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
# 3. ИНИЦИАЛИЗАЦИЯ СЕССИИ И ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# =====================================================================
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
        {"role": "assistant", "content": "Здравствуйте! Выберите проект в меню слева и задайте вопрос."}
    ]

if "metrics_history" not in st.session_state:
    st.session_state.metrics_history = []

def export_chat_history():
    text = f"# 📝 История диалога (Проект: {st.session_state.get('selected_project', 'Общий')})\n\n"
    for msg in st.session_state.messages:
        role = "👤 **Пользователь**" if msg["role"] == "user" else "🤖 **Ассистент**"
        text += f"{role}:\n{msg['content']}\n\n---\n\n"
    return text

# Функция получения всех файлов из Qdrant с группировкой по разделам
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

# =====================================================================
# 4. БОКОВАЯ ПАНЕЛЬ
# =====================================================================
with st.sidebar:
    st.header("📂 Проекты и Настройки")
    
    project_names = list(st.session_state.projects.keys())
    selected_project = st.selectbox("Активный проект:", project_names)
    st.session_state.selected_project = selected_project
    
    active_sections = st.session_state.projects.get(selected_project, [])
    st.caption(f"Подключенные разделы: **{', '.join(active_sections) if active_sections else 'Нет'}**")

    with st.expander("➕ Создать новый проект"):
        new_proj_name = st.text_input("Имя проекта:")
        chosen_sections = st.multiselect(
            "Выберите разделы знаний:",
            options=st.session_state.sections,
            default=[st.session_state.sections[0]] if st.session_state.sections else []
        )
        if st.button("Сохранить проект", use_container_width=True):
            if new_proj_name and new_proj_name not in st.session_state.projects:
                st.session_state.projects[new_proj_name] = chosen_sections
                st.success(f"Проект '{new_proj_name}' успешно создан!")
                st.rerun()

    with st.expander("⚙️ Изменить разделы проекта"):
        updated_sections = st.multiselect(
            f"Разделы для '{selected_project}':",
            options=st.session_state.sections,
            default=active_sections
        )
        if st.button("Обновить привязку", use_container_width=True):
            st.session_state.projects[selected_project] = updated_sections
            st.success("Состав разделов обновлен!")
            st.rerun()

    st.divider()
    
    st.header("📊 Статистика хранилища")
    try:
        if active_sections:
            project_filter = Filter(must=[FieldCondition(key="section", match=MatchAny(any=active_sections))])
            count_res = qdrant.count(collection_name=COLLECTION_NAME, count_filter=project_filter)
            doc_count = count_res.count
        else:
            doc_count = 0
            
        col_stat1, col_stat2 = st.columns(2)
        col_stat1.metric("Чанков в проекте", doc_count)
        col_stat2.metric("Разделов", len(active_sections))
    except Exception:
        st.caption("Данные обновляются...")

    st.divider()
    
    st.download_button(
        label="📥 Скачать историю (.md)",
        data=export_chat_history(),
        file_name=f"chat_{selected_project}.md",
        mime="text/markdown",
        use_container_width=True
    )
    
    if st.button("🗑️ Очистить диалог и метрики", use_container_width=True):
        st.session_state.messages = [
            {"role": "assistant", "content": f"Диалог очищен. Вы работаете в проекте '{selected_project}'."}
        ]
        st.session_state.metrics_history = []
        st.rerun()

# =====================================================================
# 5. ОСНОВНОЙ ИНТЕРФЕЙС
# =====================================================================
st.title(f"🤖 Ассистент — [{selected_project}]")

tab_chat, tab_upload, tab_manage, tab_analytics = st.tabs([
    "💬 Чат по проекту", 
    "📁 Загрузка документов", 
    "🗂️ Управление файлами", 
    "📈 Аналитика и Графики"
])

# ---------------------------------------------------------------------
# ВКЛАДКА 1: ЧАТ
# ---------------------------------------------------------------------
with tab_chat:
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input(f"Задайте вопрос по проекту '{selected_project}'..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Поиск информации в подключенных разделах..."):
                t_start = time.perf_counter()

                t_embed_start = time.perf_counter()
                query_vector = list(embedding_model.embed([prompt]))[0].tolist()
                t_embed = (time.perf_counter() - t_embed_start) * 1000

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
                    sections_str = ", ".join([f"'{s}'" for s in active_sections]) if active_sections else "нет подключенных разделов"
                    answer = f"В разделах ({sections_str}) пока не найдено подходящей информации. Загрузите `.md` файлы во вкладке **'Загрузка документов'**."
                    st.write(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                else:
                    context_chunks = [
                        f"[Раздел: {hit.payload.get('section', 'Общий')} | Файл: {hit.payload.get('source_file', 'Документ')}]\n{hit.payload.get('text', '')}"
                        for hit in search_results
                    ]
                    context = "\n\n---\n\n".join(context_chunks)

                    llm_prompt = f"""Ты — вежливый виртуальный ассистент базы знаний проекта "{selected_project}".
Ответь на вопрос пользователя, используя ТОЛЬКО предоставленную ниже информацию из подключенных разделов.

--- ИНФОРМАЦИЯ ИЗ БАЗЫ ЗНАНИЙ (Разделы: {', '.join(active_sections)}) ---
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

                    st.session_state.metrics_history.append({
                        "Запрос №": len(st.session_state.metrics_history) + 1,
                        "Входные токены": res.usage.prompt_tokens,
                        "Выходные токены": res.usage.completion_tokens,
                        "Всего токенов": res.usage.total_tokens,
                        "Время ответа (сек)": round(t_total, 2),
                        "Поиск Qdrant (мс)": round(t_qdrant, 0),
                        "Проект": selected_project
                    })

                    with st.expander("📊 Метрики ответа и релевантность источников"):
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("Общее время", f"{t_total:.2f} сек")
                        col2.metric("Поиск Qdrant", f"{t_qdrant:.0f} мс")
                        col3.metric("Генерация LLM", f"{t_llm:.2f} сек")
                        col4.metric("Токены (Всего)", res.usage.total_tokens)

                        st.markdown("---")
                        st.markdown("**Найденные фрагменты:**")
                        for idx, hit in enumerate(search_results, 1):
                            score_pct = round(hit.score * 100, 1)
                            src_file = hit.payload.get('source_file', 'Документ')
                            src_section = hit.payload.get('section', 'Общий')
                            st.write(f"**{idx}. [{src_section}] {src_file}** — Релевантность: `{score_pct}%`")

                    st.session_state.messages.append({"role": "assistant", "content": answer})

# ---------------------------------------------------------------------
# ВКЛАДКА 2: ЗАГРУЗКА .MD ФАЙЛОВ
# ---------------------------------------------------------------------
with tab_upload:
    st.subheader("📁 Пополнение Базы Знаний по Разделам")
    
    col_up1, col_up2 = st.columns([2, 1])
    
    with col_up1:
        target_section = st.selectbox("Выберите раздел, куда загрузить файлы:", st.session_state.sections)
    
    with col_up2:
        new_sec_input = st.text_input("➕ Или создайте новый раздел:")
        if st.button("Добавить раздел", use_container_width=True):
            if new_sec_input and new_sec_input not in st.session_state.sections:
                st.session_state.sections.append(new_sec_input)
                st.success(f"Раздел '{new_sec_input}' создан!")
                st.rerun()

    st.divider()

    uploaded_files = st.file_uploader(
        f"Перетащите `.md` файлы для добавления в раздел **'{target_section}'**:", 
        type=["md"], 
        accept_multiple_files=True
    )

    if uploaded_files and st.button(f"🚀 Загрузить файлы в раздел '{target_section}'", use_container_width=True):
        headers_to_split_on = [("#", "Header 1"), ("##", "Header 2"), ("###", "Header 3")]
        markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on, strip_headers=False)

        all_points = []
        
        with st.spinner(f"Векторизация и сохранение в раздел '{target_section}'..."):
            for file in uploaded_files:
                file_content = file.read().decode("utf-8")
                chunks = markdown_splitter.split_text(file_content)
                
                if not chunks:
                    texts = [file_content]
                    metadatas = [{}]
                else:
                    texts = [c.page_content for c in chunks]
                    metadatas = [c.metadata for c in chunks]

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
            st.success(f"🎉 Успешно загружено документов: {len(uploaded_files)} (всего {len(all_points)} фрагментов) в раздел '{target_section}'!")
            st.rerun()

# ---------------------------------------------------------------------
# ВКЛАДКА 3: УПРАВЛЕНИЕ ФАЙЛАМИ И РАЗДЕЛАМИ (НОВАЯ ФУНКЦИЯ)
# ---------------------------------------------------------------------
with tab_manage:
    st.subheader("🗂️ Обзор, перемещение и удаление файлов базы знаний")
    st.caption("Здесь вы можете перераспределять файлы между разделами или удалять устаревшие инструкции.")

    files_by_sec = get_db_files_summary()

    if not files_by_sec:
        st.info("В базе данных Qdrant пока нет загруженных файлов.")
    else:
        # Синхронизируем списки разделов из базы
        for sec in files_by_sec.keys():
            if sec not in st.session_state.sections:
                st.session_state.sections.append(sec)

        for sec_name, files_dict in files_by_sec.items():
            with st.expander(f"📁 Раздел: **{sec_name}** ({len(files_dict)} файлов)", expanded=True):
                for fname, chunk_cnt in files_dict.items():
                    st.markdown(f"📄 **{fname}** — `{chunk_cnt} фрагментов`")
                    
                    c_move, c_del = st.columns([3, 1])
                    
                    with c_move:
                        # Форма перемещения
                        other_sections = [s for s in st.session_state.sections if s != sec_name]
                        if other_sections:
                            dest_sec = st.selectbox(
                                "Переместить в раздел:", 
                                options=other_sections, 
                                key=f"sel_{sec_name}_{fname}"
                            )
                            if st.button("🚚 Переместить", key=f"btn_m_{sec_name}_{fname}"):
                                with st.spinner("Перемещение..."):
                                    # Находим ID всех чанков файла
                                    pts, _ = qdrant.scroll(
                                        collection_name=COLLECTION_NAME,
                                        scroll_filter=Filter(
                                            must=[
                                                FieldCondition(key="source_file", match=MatchValue(value=fname)),
                                                FieldCondition(key="section", match=MatchValue(value=sec_name))
                                            ]
                                        ),
                                        limit=10000,
                                        with_payload=False,
                                        with_vectors=False
                                    )
                                    p_ids = [p.id for p in pts]
                                    if p_ids:
                                        qdrant.set_payload(
                                            collection_name=COLLECTION_NAME,
                                            payload={"section": dest_sec},
                                            points=p_ids
                                        )
                                        st.success(f"Файл '{fname}' перемещен в раздел '{dest_sec}'!")
                                        st.rerun()

                    with c_del:
                        st.write("") # Отступ
                        st.write("")
                        if st.button("🗑️ Удалить файл", key=f"btn_d_{sec_name}_{fname}", type="primary"):
                            with st.spinner("Удаление..."):
                                pts, _ = qdrant.scroll(
                                    collection_name=COLLECTION_NAME,
                                    scroll_filter=Filter(
                                        must=[
                                            FieldCondition(key="source_file", match=MatchValue(value=fname)),
                                            FieldCondition(key="section", match=MatchValue(value=sec_name))
                                        ]
                                    ),
                                    limit=10000,
                                    with_payload=False,
                                    with_vectors=False
                                )
                                p_ids = [p.id for p in pts]
                                if p_ids:
                                    qdrant.delete(
                                        collection_name=COLLECTION_NAME,
                                        points_selector=p_ids
                                    )
                                    st.success(f"Файл '{fname}' полностью удален из базы!")
                                    st.rerun()
                    st.divider()

# ---------------------------------------------------------------------
# ВКЛАДКА 4: АНАЛИТИКА И ГРАФИКИ
# ---------------------------------------------------------------------
with tab_analytics:
    st.subheader("📈 Аналитика производительности и использования LLM")
    
    if not st.session_state.metrics_history:
        st.info("Задайте несколько вопросов в чате, чтобы здесь появились графики расхода токенов и задержек.")
    else:
        df_metrics = pd.DataFrame(st.session_state.metrics_history)

        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        m_col1.metric("Всего запросов", len(df_metrics))
        m_col2.metric("Сумма токенов", f"{df_metrics['Всего токенов'].sum():,}")
        m_col3.metric("Среднее время ответа", f"{df_metrics['Время ответа (сек)'].mean():.2f} сек")
        m_col4.metric("Средний поиск Qdrant", f"{df_metrics['Поиск Qdrant (мс)'].mean():.0f} мс")

        st.divider()

        st.markdown("### 📊 Расход токенов (Prompt vs Completion)")
        tokens_chart_data = df_metrics.set_index("Запрос №")[["Входные токены", "Выходные токены"]]
        st.bar_chart(tokens_chart_data)

        st.markdown("### ⏱️ Динамика времени ответа (секунды)")
        latency_chart_data = df_metrics.set_index("Запрос №")[["Время ответа (сек)"]]
        st.line_chart(latency_chart_data)

        with st.expander("📄 Полная таблица метрик сессии"):
            st.dataframe(df_metrics, use_container_width=True)
