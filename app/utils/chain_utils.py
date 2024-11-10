import os
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.prompts.prompt import PromptTemplate

from app.utils.rag_utils import hugging_face_embeddings, get_milvus_retriever

HUGGINGFACE_API_KEY = os.environ.get('HUGGINGFACE_API_KEY')
MILVUS_COLLECTION_NAME = os.environ.get('MILVUS_COLLECTION_NAME')
MILVUS_HOST = os.environ.get('MILVUS_HOST')
MILVUS_PORT = os.environ.get('MILVUS_PORT')
URI = os.environ.get('URI')

def format_docs(docs):
    documents = "\n\n".join(str(doc.page_content) for doc in docs)
    return documents

def get_response_from_llm(llm, input):

    embeddings = hugging_face_embeddings(HUGGINGFACE_API_KEY)
    vectorstore = get_milvus_retriever(uri=URI, embeddings=embeddings, collection_name=MILVUS_COLLECTION_NAME)
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 2})

    prompt_template = """
        คุณคือแชทบอทหญิงของบริษัท Chia-Tai โดยมีหน้าที่ให้ความรู้เกี่ยวกับปุ๋ยและโดรนของบริษัท โดยสามารถตอบคำถามจากองค์ความรู้ที่มีอยู่เท่านั้น 
        หากคำถามอยู่นอกเหนือองค์ความรู้ที่มีให้ตอบกลับว่า "ดิฉันไม่สามารถให้คำตอบกับคำถามนี้ได้ ต้องขออภัยเป็นอย่างสูงค่ะ"
        
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
        input
    )

    return response