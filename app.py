import streamlit as st
import os
import csv
from datetime import datetime

# ==========================================
# 1. 核心引用
# ==========================================
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# ==========================================
# 2. 配置区域
# ==========================================
os.environ["OPENAI_API_KEY"] = "sk-i0HXYWyGQZ6v5VKdoM0alDBvTpPD8GxVHja1ex6rR0lfP29G"
os.environ["OPENAI_API_BASE"] = "https://api.openai-proxy.org/v1" 

INDEX_PATH = "faiss_index_local"
LOCAL_MODEL_NAME = "shibing624/text2vec-base-chinese"
# ✅ 日志文件名称
LOG_FILE = "chat_history_log.csv"

# ==========================================
# 3. 核心功能
# ==========================================

@st.cache_resource
def load_embedding_model():
    """加载本地模型"""
    print(f"🔄 正在加载本地模型: {LOCAL_MODEL_NAME} ...")
    return HuggingFaceEmbeddings(
        model_name=LOCAL_MODEL_NAME,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

def process_documents():
    """构建知识库"""
    if not os.path.exists("./data"): return False, "❌ 无 data 文件夹"
    
    loader = DirectoryLoader('./data', glob="**/*.txt", loader_cls=TextLoader, loader_kwargs={'encoding': 'utf-8'})
    try: docs = loader.load()
    except Exception as e: return False, f"❌ 读取失败: {e}"
    if not docs: return False, "⚠️ data 为空"

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
    splits = text_splitter.split_documents(docs)

    try:
        embeddings = load_embedding_model()
        vectorstore = FAISS.from_documents(splits, embeddings)
        vectorstore.save_local(INDEX_PATH)
        return True, f"✅ 知识库构建成功！"
    except Exception as e:
        return False, f"❌ 构建失败: {e}"

def get_chain():
    if not os.path.exists(INDEX_PATH): return None
    
    embeddings = load_embedding_model()
    try:
        vectorstore = FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    except: return None

    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.2)
    
    # 人设提示词
    template = """
    你是一位拥有10年经验的儿科过敏专科营养师，名叫“敏宝守护者”。
    请务必严格遵守以下【回答原则】：
    1. 语气温柔、坚定，多用“咱们宝宝”等亲切词汇。
    2. 严格基于【参考资料】回答。
    3. 如果没有答案，请诚恳地说不知道，并建议咨询医生。

    【参考资料】：
    {context}

    家长的问题：{question}
    """
    prompt = ChatPromptTemplate.from_template(template)
    
    def format_docs(docs): return "\n\n".join(d.page_content for d in docs)

    chain = ({"context": retriever | format_docs, "question": RunnablePassthrough()} 
             | prompt | llm | StrOutputParser())
    return chain

# ✅ 新增功能：保存聊天记录到 CSV 文件
def save_log(user_question, ai_answer):
    # 如果文件不存在，先创建并写入表头
    file_exists = os.path.isfile(LOG_FILE)
    
    with open(LOG_FILE, mode='a', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['时间', '用户问题', 'AI回答']) # 表头
        
        # 写入当前时间、问题、回答
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        writer.writerow([current_time, user_question, ai_answer])

# ==========================================
# 4. 界面逻辑
# ==========================================
st.title("🛡️ 敏宝守护者")

with st.sidebar:
    if st.button("🔄 重建知识库"):
        with st.spinner("处理中..."):
            s, m = process_documents()
            if s: st.success(m)
            else: st.error(m)
    
    # ✅ 在侧边栏增加一个下载按钮，方便管理员查看记录
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "rb") as f:
            st.download_button(
                label="📥 下载所有聊天记录",
                data=f,
                file_name="chat_history.csv",
                mime="text/csv"
            )

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if input := st.chat_input("请输入问题..."):
    st.session_state.messages.append({"role": "user", "content": input})
    st.chat_message("user").write(input)
    
    chain = get_chain()
    if chain:
        with st.chat_message("assistant"):
            response_container = st.empty() # 创建占位符
            response = chain.invoke(input)
            response_container.write(response)
            
            st.session_state.messages.append({"role": "assistant", "content": response})
            
            # ✅ 关键一步：记录到后台文件
            save_log(input, response)
            print(f"📝 已记录: {input}") # 在黑色终端也打印一下
    else:
        st.warning("⚠️ 请先重建知识库")

