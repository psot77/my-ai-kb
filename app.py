import uuid
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
    
    # Проверяем/создаем коллекцию в Qdrant
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
# 3. ИНИЦИАЛИЗАЦИЯ СЕССИИ (Проекты и Чат)
# =====================================================================
if "projects" not in st.session_state:
    st.session_state.projects = ["Общий", "Медицина", "IT Проекты", "Отдел продаж"]

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Здравствуйте! Выберите проект в меню слева и задайте вопрос."}
    ]

def export_chat_history():
    text = f"# 📝 История диалога (Проект: {st.session_state.get('selected_project', 'Общий')})\n\n"
    for msg in st.session_state.messages:
        role = "👤 **Пользователь**" if msg["role"] == "user" else "🤖 **Ассистент**"
        text += f"{role}:\n{msg['content']}\n\n---\n\n"
    return text

# =====================================================================
# 4. БОКОВАЯ ПАНЕЛЬ (ВЫБОР И СОЗДАНИЕ ПРОЕКТОВ)
# =====================================================================
with st.sidebar:
    st.header("📂 Управление проектами")
    
    # Выбор проекта
    selected_project = st.selectbox("Активный проект / Тема:", st.session_state.projects)
    st.session_state.selected_project = selected_project
    
    # Создание нового проекта
    new_proj = st.text_input("➕ Создать новый проект:")
    if st.button("Добавить проект", use_container_width=True):
        if new_proj and new_proj not in st.session_state.projects:
            st.session_state.projects.append(new_proj)
            st.success(f"Проект '{new_proj}' создан!")
            st.rerun()

    st.divider()
    st.header("⚙️ Опции чата")
    
    st.download_button(
        label="📥 Скачать историю (.md)",
        data=export_chat_history(),
        file_name=f"chat_{selected_project}.md",
        mime="text/markdown",
        use_container_width=True
    )
    
    if st.button("🗑️ Очистить диалог", use_container_width=True):
        st.session_state.messages = [
            {"role": "assistant", "content": f"Диалог очищен. Вы работаете в проекте '{selected_project}'."}
        ]
        st.rerun()

# =====================================================================
# 5. ОСНОВНОЙ ИНТЕРФЕЙС (ВКЛАДКИ)
# =====================================================================
st.title(f"🤖 Ассистент Базы Знаний — [{selected_project}]")

tab_chat, tab_upload = st.tabs(["💬 Чат по проекту", "📁 Загрузка документов (.md)"])

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
            with st.spinner("Поиск в базе знаний проекта..."):
                query_vector = list(embedding_model.embed([prompt]))[0].tolist()

                # ФИЛЬТР: Ищем документы СТРОГО по выбранному проекту!
                project_filter = Filter(
                    must=[
                        FieldCondition(
                            key="project",
                            match=MatchValue(value=selected_project)
                        )
                    ]
                )

                response = qdrant.query_points(
                    collection_name=COLLECTION_NAME,
                    query=query_vector,
                    query_filter=project_filter,
                    limit=3
                )
                search_results = response.points

                if not search_results:
                    answer = f"В проекте **'{selected_project}'** не найдено подходящей информации. Попробуйте загрузить соответствующие .md файлы во вкладке 'Загрузка документов'."
                else:
                    context_chunks = [
                        f"[Источник: {hit.payload.get('source_file', 'Документ')}]\n{hit.payload.get('text', '')}"
                        for hit in search_results
                    ]
                    context = "\n\n---\n\n".join(context_chunks)

                    llm_prompt = f"""Ты — вежливый виртуальный ассистент базы знаний проекта "{selected_project}".
Ответь на вопрос пользователя, используя ТОЛЬКО предоставленную ниже информацию.
Если ответа нет в информации, честно ответь: "В базе знаний этого проекта нет информации по данному вопросу".

--- ИНФОРМАЦИЯ ИЗ БАЗЫ ЗНАНИЙ (Проект: {selected_project}) ---
{context}

--- ВОПРОС ПОЛЬЗОВАТЕЛЯ ---
{prompt}

--- ОТВЕТ ---"""

                    res = groq_client.chat.completions.create(
                        model="llama-3.3-70b-versatile",
                        messages=[{"role": "user", "content": llm_prompt}],
                        temperature=0.2
                    )
                    answer = res.choices[0].message.content

                st.write(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})

# ---------------------------------------------------------------------
# ВКЛАДКА 2: ЗАГРУЗКА .MD ФАЙЛОВ
# ---------------------------------------------------------------------
with tab_upload:
    st.subheader(f"Загрузка новых инструкций в проект: **{selected_project}**")
    st.write("Выберите один или несколько `.md` файлов. Они автоматически векторизуются и привяжутся к текущему проекту.")

    uploaded_files = st.file_uploader(
        "Перетащите .md файлы сюда", 
        type=["md"], 
        accept_multiple_files=True
    )

    if uploaded_files and st.button("🚀 Векторизовать и сохранить в проект", use_container_width=True):
        headers_to_split_on = [("#", "Header 1"), ("##", "Header 2"), ("###", "Header 3")]
        markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on, strip_headers=False)

        all_points = []
        
        with st.spinner("Обработка и отправка векторов в облако Qdrant..."):
            for file in uploaded_files:
                file_content = file.read().decode("utf-8")
                chunks = markdown_splitter.split_text(file_content)
                
                # Если в markdown нет заголовков, сохраняем целиком
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
                            id=uuid.uuid4().hex,  # Уникальный ID чанка
                            vector=emb.tolist(),
                            payload={
                                "text": texts[idx],
                                "source_file": file.name,
                                "project": selected_project,  # ПРИВЯЗКА К ПРОЕКТУ
                                **metadatas[idx]
                            }
                        )
                    )

            # Сохраняем все чанки в Qdrant Cloud
            qdrant.upsert(collection_name=COLLECTION_NAME, points=all_points)
            
            st.success(f"🎉 Успешно загружено документов: {len(uploaded_files)} (всего {len(all_points)} фрагментов) в проект '{selected_project}'!")
