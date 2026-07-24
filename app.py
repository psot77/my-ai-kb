import uuid
import time
import pandas as pd
import streamlit as st
from fastembed import TextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from langchain_text_splitters import MarkdownHeaderTextSplitter
from groq import Groq

# =====================================================================
# 1. НАСТРОЙКИ КЛЮЧЕЙ И СТРАНИЦЫ
# =====================================================================
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
QDRANT_API_KEY = st.secrets["QDRANT_API_KEY"]

QDRANT_URL = "https://18545c10-4b80-4ed2-9304-4ba636a29618.eu-west-1-0.aws.cloud.qdrant.io"
COLLECTION_NAME = "knowledge_base"

st.set_page_config(page_title="Мульти-проектная База Знаний AI", page_icon="📚", layout="wide")

# =====================================================================
# 2. ИНИЦИАЛИЗАЦИЯ СЕРВИСОВ И БАЗЫ
# =====================================================================
@st.cache_resource
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
        
    groq_client = Groq(api_key=GROQ_API_KEY)
    embed_model = TextEmbedding(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    return qdrant, groq_client, embed_model

qdrant, groq_client, embedding_model = init_services()

# =====================================================================
# 3. ИНИЦИАЛИЗАЦИЯ СЕССИИ (Проекты, Сообщения, Метрики)
# =====================================================================
if "projects" not in st.session_state:
    st.session_state.projects = ["Общий", "Медицина", "IT Проекты", "Отдел продаж"]

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Здравствуйте! Выберите проект в меню слева и задайте вопрос."}
    ]

# Хранилище метрик для графиков
if "metrics_history" not in st.session_state:
    st.session_state.metrics_history = []

def export_chat_history():
    text = f"# 📝 История диалога (Проект: {st.session_state.get('selected_project', 'Общий')})\n\n"
    for msg in st.session_state.messages:
        role = "👤 **Пользователь**" if msg["role"] == "user" else "🤖 **Ассистент**"
        text += f"{role}:\n{msg['content']}\n\n---\n\n"
    return text

# =====================================================================
# 4. БОКОВАЯ ПАНЕЛЬ (ПРОЕКТЫ И МЕТРИКИ ХРАНИЛИЩА)
# =====================================================================
with st.sidebar:
    st.header("📂 Управление проектами")
    
    selected_project = st.selectbox("Активный проект / Тема:", st.session_state.projects)
    st.session_state.selected_project = selected_project
    
    new_proj = st.text_input("➕ Создать новый проект:")
    if st.button("Добавить проект", use_container_width=True):
        if new_proj and new_proj not in st.session_state.projects:
            st.session_state.projects.append(new_proj)
            st.success(f"Проект '{new_proj}' создан!")
            st.rerun()

    st.divider()
    
    st.header("📊 Статистика хранилища")
    try:
        project_filter = Filter(must=[FieldCondition(key="project", match=MatchValue(value=selected_project))])
        count_res = qdrant.count(collection_name=COLLECTION_NAME, count_filter=project_filter)
        
        col_stat1, col_stat2 = st.columns(2)
        col_stat1.metric("Чанков в проекте", count_res.count)
        col_stat2.metric("Модель LLM", "Llama 3.3")
    except Exception:
        st.caption("Не удалось загрузить данные о хранилище")

    st.divider()
    st.header("⚙️ Опции чата")
    
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
# 5. ОСНОВНОЙ ИНТЕРФЕЙС (ВКЛАДКИ)
# =====================================================================
st.title(f"🤖 Ассистент Базы Знаний — [{selected_project}]")

tab_chat, tab_upload, tab_analytics = st.tabs([
    "💬 Чат по проекту", 
    "📁 Загрузка документов (.md)", 
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
            with st.spinner("Замер производительности и поиск в базе..."):
                t_start = time.perf_counter()

                # 1. Векторизация
                t_embed_start = time.perf_counter()
                query_vector = list(embedding_model.embed([prompt]))[0].tolist()
                t_embed = (time.perf_counter() - t_embed_start) * 1000

                # 2. Поиск Qdrant
                t_qdrant_start = time.perf_counter()
                project_filter = Filter(must=[FieldCondition(key="project", match=MatchValue(value=selected_project))])
                response = qdrant.query_points(
                    collection_name=COLLECTION_NAME,
                    query=query_vector,
                    query_filter=project_filter,
                    limit=3
                )
                t_qdrant = (time.perf_counter() - t_qdrant_start) * 1000
                search_results = response.points

                if not search_results:
                    answer = f"В проекте **'{selected_project}'** не найдено подходящей информации."
                    st.write(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                else:
                    context_chunks = [
                        f"[Источник: {hit.payload.get('source_file', 'Документ')}]\n{hit.payload.get('text', '')}"
                        for hit in search_results
                    ]
                    context = "\n\n---\n\n".join(context_chunks)

                    llm_prompt = f"""Ты — вежливый виртуальный ассистент базы знаний проекта "{selected_project}".
Ответь на вопрос пользователя, используя ТОЛЬКО предоставленную ниже информацию.

--- ИНФОРМАЦИЯ ИЗ БАЗЫ ЗНАНИЙ (Проект: {selected_project}) ---
{context}

--- ВОПРОС ПОЛЬЗОВАТЕЛЯ ---
{prompt}

--- ОТВЕТ ---"""

                    # 3. Запрос к LLM (Groq)
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

                    # ФИКСАЦИЯ МЕТРИК В ИСТОРИЮ
                    st.session_state.metrics_history.append({
                        "Запрос №": len(st.session_state.metrics_history) + 1,
                        "Входные токены": res.usage.prompt_tokens,
                        "Выходные токены": res.usage.completion_tokens,
                        "Всего токенов": res.usage.total_tokens,
                        "Время ответа (сек)": round(t_total, 2),
                        "Поиск Qdrant (мс)": round(t_qdrant, 0),
                        "Проект": selected_project
                    })

                    # Вывод метрик под ответом
                    with st.expander("📊 Метрики ответа и релевантность источников"):
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("Общее время", f"{t_total:.2f} сек")
                        col2.metric("Поиск Qdrant", f"{t_qdrant:.0f} мс")
                        col3.metric("Генерация LLM", f"{t_llm:.2f} сек")
                        col4.metric("Токены (Всего)", res.usage.total_tokens)

                        st.markdown("---")
                        st.caption(f"**Детализация:** Prompt: {res.usage.prompt_tokens} | Completion: {res.usage.completion_tokens}")
                        
                        st.markdown("**Найденные фрагменты (Top Match):**")
                        for idx, hit in enumerate(search_results, 1):
                            score_pct = round(hit.score * 100, 1)
                            src_file = hit.payload.get('source_file', 'Документ')
                            st.write(f"**{idx}. {src_file}** — Релевантность: `{score_pct}%`")

                    st.session_state.messages.append({"role": "assistant", "content": answer})

# ---------------------------------------------------------------------
# ВКЛАДКА 2: ЗАГРУЗКА .MD ФАЙЛОВ
# ---------------------------------------------------------------------
with tab_upload:
    st.subheader(f"Загрузка новых инструкций в проект: **{selected_project}**")

    uploaded_files = st.file_uploader("Перетащите .md файлы сюда", type=["md"], accept_multiple_files=True)

    if uploaded_files and st.button("🚀 Векторизовать и сохранить в проект", use_container_width=True):
        headers_to_split_on = [("#", "Header 1"), ("##", "Header 2"), ("###", "Header 3")]
        markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on, strip_headers=False)

        all_points = []
        
        with st.spinner("Обработка и отправка векторов в облако Qdrant..."):
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
                                "project": selected_project,
                                **metadatas[idx]
                            }
                        )
                    )

            qdrant.upsert(collection_name=COLLECTION_NAME, points=all_points)
            st.success(f"🎉 Успешно загружено документов: {len(uploaded_files)} (всего {len(all_points)} фрагментов) в проект '{selected_project}'!")
            st.rerun()

# ---------------------------------------------------------------------
# ВКЛАДКА 3: АНАЛИТИКА И ГРАФИКИ
# ---------------------------------------------------------------------
with tab_analytics:
    st.subheader("📈 Аналитика производительности и использования LLM")
    
    if not st.session_state.metrics_history:
        st.info("Задайте несколько вопросов в чате, чтобы здесь появились графики расхода токенов и задержек.")
    else:
        df_metrics = pd.DataFrame(st.session_state.metrics_history)

        # Сводные карточки
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        m_col1.metric("Всего запросов", len(df_metrics))
        m_col2.metric("Сумма токенов", f"{df_metrics['Всего токенов'].sum():,}")
        m_col3.metric("Среднее время ответа", f"{df_metrics['Время ответа (сек)'].mean():.2f} сек")
        m_col4.metric("Средний поиск Qdrant", f"{df_metrics['Поиск Qdrant (мс)'].mean():.0f} мс")

        st.divider()

        # График 1: Расход токенов по запросам
        st.markdown("### 📊 Расход токенов (Prompt vs Completion)")
        tokens_chart_data = df_metrics.set_index("Запрос №")[["Входные токены", "Выходные токены"]]
        st.bar_chart(tokens_chart_data)

        # График 2: Скорость ответа (Latency)
        st.markdown("### ⏱️ Динамика времени ответа (секунды)")
        latency_chart_data = df_metrics.set_index("Запрос №")[["Время ответа (сек)"]]
        st.line_chart(latency_chart_data)

        # Таблица сырых данных
        with st.expander("📄 Полная таблица метрик сессии"):
            st.dataframe(df_metrics, use_container_width=True)
