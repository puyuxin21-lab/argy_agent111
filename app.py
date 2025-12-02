import streamlit as st
import os

# ==========================================
# 1. 核心引用 (Windows 稳定版)
# ==========================================
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
# ✅ 使用本地模型 (你刚才已经下载好了)
from langchain_huggingface import HuggingFaceEmbeddings
# ✅ 关键替换：使用 FAISS (Windows上绝对不闪退，且支持本地保存)
from langchain_community.vectorstores import FAISS

from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
# LLM 依然用云端，保证回答质量
from langchain_openai import ChatOpenAI

# ==========================================
# 2. 配置区域
# ==========================================
os.environ["OPENAI_API_KEY"] = "sk-i0HXYWyGQZ6v5VKdoM0alDBvTpPD8GxVHja1ex6rR0lfP29G"
os.environ["OPENAI_API_BASE"] = "https://api.openai-proxy.org/v1"

# ✅ FAISS 索引保存路径 (实现永久记忆)
INDEX_PATH = "faiss_index_local"

# ✅ 本地模型名称 (和你刚才下载的一致，不会重复下载)
LOCAL_MODEL_NAME = "shibing624/text2vec-base-chinese"


# ==========================================
# 3. 核心功能
# ==========================================

@st.cache_resource
def load_embedding_model():
    """加载本地模型"""
    print(f"🔄 正在加载本地模型: {LOCAL_MODEL_NAME} ...")
    # 强制指定 device='cpu'，避开显卡报错
    return HuggingFaceEmbeddings(
        model_name=LOCAL_MODEL_NAME,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )


def process_documents():
    """构建知识库"""
    if not os.path.exists("./data"): return False, "❌ 无 data 文件夹"

    loader = DirectoryLoader('./data', glob="**/*.txt", loader_cls=TextLoader, loader_kwargs={'encoding': 'utf-8'})
    try:
        docs = loader.load()
    except Exception as e:
        return False, f"❌ 读取失败: {e}"
    if not docs: return False, "⚠️ data 为空"

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=80)
    splits = text_splitter.split_documents(docs)

    try:
        # 1. 加载本地模型 (秒开)
        embeddings = load_embedding_model()

        # 2. 使用 FAISS 构建向量库 (绝对不闪退)
        vectorstore = FAISS.from_documents(splits, embeddings)

        # 3. 保存到硬盘 (实现记忆)
        vectorstore.save_local(INDEX_PATH)
        return True, f"✅ 成功收录 {len(splits)} 条知识 (本地模型+FAISS)"
    except Exception as e:
        return False, f"❌ 构建失败: {e}"


def get_chain():
    if not os.path.exists(INDEX_PATH): return None

    embeddings = load_embedding_model()

    try:
        # ✅ 加载本地 FAISS 索引
        vectorstore = FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    except:
        return None

    # 检索器
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

    template = """
        你是一位拥有10年经验的儿科过敏专科营养师，名叫“敏宝守护者”。
        你的服务对象是因宝宝牛奶蛋白过敏（CMPA）而感到焦虑、无助的家长。

        请务必严格遵守以下【回答原则】：
        1.  **共情安抚（第一优先级）**：
            - 开场请先安抚家长的情绪，例如：“宝妈/宝爸别急，过敏是宝宝常见的成长小挑战...”
            - 语气要温柔、坚定，多用“咱们宝宝”、“小肚肚”等亲切词汇。

        2.  **基于事实**：
            - 必须严格基于下方的【参考资料】回答。
            - 如果资料里有数据（如转奶天数、冲泡温度），请精确列出。

        3.  **通俗易懂（比喻法）**：
            - 遇到专业术语要解释。例如：
              * 把“深度水解奶粉”比喻为“切得很碎的面条，好消化但还有一点点口感”。
              * 把“氨基酸奶粉”比喻为“彻底磨成粉的食物，完全没有致敏性”。

        4.  **诚实与安全**：
            - 如果【参考资料】里没有答案，请诚恳地说：“抱歉，我的知识库里暂时没查到这点，为了宝宝安全，建议直接咨询医生。” **绝对不要瞎编！**
            - 回答结尾必须加上：“💡 温馨提示：以上建议仅供参考，具体诊疗方案请以医生面诊为准。”

        【参考资料】：
        {context}

        家长的问题：{question}
        """
    prompt = ChatPromptTemplate.from_template(template)

    def format_docs(docs):
        return "\n\n".join(d.page_content for d in docs)

    chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt | llm | StrOutputParser()
    )
    return chain


# ==========================================
# 4. 界面逻辑
# ==========================================
st.title("🛡️ 敏宝守护者 (Windows 最终版)")
st.caption("架构：本地 Embedding (CPU) + FAISS 持久化 + 零成本")

with st.sidebar:
    if st.button("🔄 重建知识库"):
        with st.spinner("正在处理..."):
            s, m = process_documents()
            if s:
                st.success(m)
            else:
                st.error(m)

# ✅ 修复点：初始化聊天记录 (防止 AttributeError)
if "messages" not in st.session_state:
    st.session_state.messages = []

# ✅ 修复点：显示历史聊天记录
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if input := st.chat_input("请输入问题..."):
    st.session_state.messages.append({"role": "user", "content": input})
    st.chat_message("user").write(input)

    chain = get_chain()
    if chain:
        with st.chat_message("assistant"):
            st.write(chain.invoke(input))
    else:
        st.warning("⚠️ 请先重建知识库")