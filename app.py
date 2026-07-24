import streamlit as st
from fastembed import TextEmbedding
from qdrant_client import QdrantClient
from groq import Groq

# =====================================================================
# 1. НАСТРОЙКИ КЛЮЧЕЙ И СТРАНИЦЫ
# =====================================================================
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
QDRANT_API_KEY = st.secrets["QDRANT_API_KEY"]

QDRANT_URL = "https://18545c10-4b80-4ed2-9304-4ba636a29618.eu-west-1-0.aws.cloud.qdrant.io"
COLLECTION_NAME = "knowledge_base"

st.set_page_config(page_title="База Знаний AI", page_icon="🤖", layout="wide")

# =====================================================================
# 2. ИНИЦИАЛИЗАЦИЯ СЕРВИСОВ (Кеширование)
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
    groq_client = Groq(api_key=GROQ_API_KEY)
    embed_model = TextEmbedding(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    return qdrant, groq_client, embed_model

qdrant, groq_client, embedding_model = init_services()

# =====================================================================
# 3. ИСТОРИЯ ЧАТА И ФУНКЦИЯ ВЫГРУЗКИ
# =====================================================================
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Здравствуйте! Я готов ответить на любой вопрос по вашей базе знаний."}
    ]

def export_chat_history():
    text = "# 📝 История диалога с AI-ассистентом\n\n"
    for msg in st.session_state.messages:
        role = "👤 **Пользователь**" if msg["role"] == "user" else "🤖 **Ассистент**"
        text += f"{role}:\n{msg['content']}\n\n---\n\n"
    return text

# =====================================================================
# 4. БОКОВАЯ ПАНЕЛЬ (ВЫГРУЗКА И УПРАВЛЕНИЕ)
# =====================================================================
with st.sidebar:
    st.header("⚙️ Управление чатом")
    st.write("История сохраняется автоматически во время сессии.")
    
    # Кнопка скачивания истории
    st.download_button(
        label="📥 Скачать историю (.md)",
        data=export_chat_history(),
        file_name="chat_history.md",
        mime="text/markdown",
        use_container_width=True
    )
    
    # Кнопка сброса
    if st.button("🗑️ Очистить диалог", use_container_width=True):
        st.session_state.messages = [
            {"role": "assistant", "content": "Здравствуйте! Я готов ответить на любой вопрос по вашей базе знаний."}
        ]
        st.rerun()

# =====================================================================
# 5. ОСНОВНОЙ ИНТЕРФЕЙС
# =====================================================================
st.title("🤖 Виртуальный Ассистент Базы Знаний")
st.caption("Задавайте любые вопросы по загруженным .md инструкциям.")

# Отрисовка всех сообщений
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# Обработка ввода
if prompt := st.chat_input("Задайте вопрос по базе знаний..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Ищу ответ в базе знаний..."):
            query_vector = list(embedding_model.embed([prompt]))[0].tolist()

            response = qdrant.query_points(
                collection_name=COLLECTION_NAME,
                query=query_vector,
                limit=3
            )
            search_results = response.points

            if not search_results:
                answer = "К сожалению, в базе знаний нет подходящей информации."
            else:
                context_chunks = [
                    f"[Источник: {hit.payload.get('source_file', 'Документ')}]\n{hit.payload.get('text', '')}"
                    for hit in search_results
                ]
                context = "\n\n---\n\n".join(context_chunks)

                llm_prompt = f"""Ты — вежливый виртуальный ассистент базы знаний.
Ответь на вопрос пользователя, используя ТОЛЬКО предоставленную ниже информацию.
Если в информации нет ответа, честно ответь: "В базе знаний нет информации по этому вопросу".

--- ИНФОРМАЦИЯ ИЗ БАЗЫ ЗНАНИЙ ---
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
