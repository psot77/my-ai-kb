import streamlit as st
from fastembed import TextEmbedding
from qdrant_client import QdrantClient
from groq import Groq

# =====================================================================
# 1. НАСТРОЙКИ КЛЮЧЕЙ
# =====================================================================
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
QDRANT_API_KEY = st.secrets["QDRANT_API_KEY"]
QDRANT_URL = "https://18545c10-4b80-4ed2-9304-4ba636a29618.eu-west-1-0.aws.cloud.qdrant.io"
COLLECTION_NAME = "knowledge_base"

# Настройка заголовка вкладки браузера
st.set_page_config(page_title="База Знаний AI", page_icon="🤖", layout="centered")

# =====================================================================
# 2. ИНИЦИАЛИЗАЦИЯ СЕРВИСОВ (Загружаются 1 раз при старте)
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
# 3. ИНТЕРФЕЙС И ИСТОРИЯ ЧАТА
# =====================================================================
st.title("🤖 Виртуальный Ассистент Базы Знаний")
st.caption("Задавайте любые вопросы по загруженным .md инструкциям.")

# Инициализация истории сообщений в сессии браузера
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Здравствуйте! Я готов ответить на любой вопрос по вашей базе знаний."}
    ]

# Отрисовка всех прошлых сообщений на экране
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# =====================================================================
# 4. ОБРАБОТКА ВВОДА ПОЛЬЗОВАТЕЛЯ
# =====================================================================
if prompt := st.chat_input("Задайте вопрос по базе знаний..."):
    # 1. Показываем вопрос пользователя в чате
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # 2. Генерируем ответ ассистента
    with st.chat_message("assistant"):
        with st.spinner("Ищу ответ в базе знаний..."):
            # А. Превращаем вопрос в вектор
            query_vector = list(embedding_model.embed([prompt]))[0].tolist()

            # Б. Поиск 3 подходящих чанков в Qdrant Cloud
            response = qdrant.query_points(
                collection_name=COLLECTION_NAME,
                query=query_vector,
                limit=3
            )
            search_results = response.points

            if not search_results:
                answer = "К сожалению, в базе знаний нет подходящей информации."
            else:
                context_chunks = []
                for hit in search_results:
                    source = hit.payload.get("source_file", "Документ")
                    text = hit.payload.get("text", "")
                    context_chunks.append(f"[Источник: {source}]\n{text}")

                context = "\n\n---\n\n".join(context_chunks)

                # В. Формируем промпт
                llm_prompt = f"""Ты — вежливый виртуальный ассистент базы знаний.
Ответь на вопрос пользователя, используя ТОЛЬКО предоставленную ниже информацию.
Если в информации нет ответа, честно ответь: "В базе знаний нет информации по этому вопросу".

--- ИНФОРМАЦИЯ ИЗ БАЗЫ ЗНАНИЙ ---
{context}

--- ВОПРОС ПОЛЬЗОВАТЕЛЯ ---
{prompt}

--- ОТВЕТ ---"""

                # Г. Запрос в Groq (Llama 3.3 70B)
                res = groq_client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[{"role": "user", "content": llm_prompt}],
                    temperature=0.2
                )
                answer = res.choices[0].message.content

            # Показываем ответ и сохраняем его в историю
            st.write(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
