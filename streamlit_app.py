
import streamlit as st
from langchain.prompts.prompt import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import time

from utils.rag_utils import get_milvus_retriever, hugging_face_embeddings

st.title("Chia-Tai Fertilizer 👨🏼‍🌾 & Drone 🚁 Chat Bot")

# llm = ChatOpenAI(model_name="gpt-3.5-turbo", api_key=st.secrets["OPENAI_API_KEY"], max_tokens=512)
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    temperature=0,
    max_tokens=1024,
    timeout=None,
    max_retries=2,
    google_api_key=st.secrets["GOOGLE_API_KEY"]
)
embeddings = hugging_face_embeddings(st.secrets["HUGGINGFACE_API_KEY"])
vectorstore = get_milvus_retriever(uri=st.secrets["URI"], embeddings=embeddings, collection_name=st.secrets["MILVUS_COLLECTION_NAME"])
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 2})


def format_docs(docs):
    documents = "\n\n".join(str(doc.page_content) for doc in docs)
    return documents

def modify_output(input):
    # Iterate over each word in the input string
    for text in input.split():
        # Yield the word with an added space
        yield text + " "
        # Introduce a small delay between each word
        time.sleep(0.05)


if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-3.5-turbo"

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):

        prompt_template = """
        คุณคือแชทบอทหญิงของบริษัท Chia-Tai โดยมีหน้าที่ให้ความรู้เกี่ยวกับปุ๋ยและโดรนของบริษัท โดยสามารถตอบตามองค์ความรู้ที่มีอยู่เท่านั้น หากคำถามอยู่นอกเหนือองค์ความรู้ที่มีให้ตอบกลับว่า "ดิฉันไม่สามารถให้คำตอบกับคำถามนี้ได้ ต้องขออภัยเป็นอย่างสูงค่ะ"
        
            องค์ความรู้ที่เรามีอยู่: 
                {context}

            คำถาม:{input}
            คำตอบ:
        """
        

        template = PromptTemplate.from_template(prompt_template)

        rag_chain = (
            {"context": retriever | format_docs, "input": RunnablePassthrough()}
            | template
            | llm
            | StrOutputParser()
        )

        # response = rag_chain.invoke(prompt)
        response = rag_chain.invoke(
            prompt
        )
        # response = result["response"]
        st.write_stream(modify_output(response))

    st.session_state.messages.append({"role": "assistant", "content": response})
    print(response)