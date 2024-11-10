from langchain_milvus.vectorstores import Milvus
from langchain_community.embeddings import HuggingFaceHubEmbeddings

def hugging_face_embeddings(api_key, hf_model="intfloat/multilingual-e5-base"):
    """
    Create customer embeddings using huggingface
    """
    embeddings = HuggingFaceHubEmbeddings(
            huggingfacehub_api_token=api_key, model=hf_model
        )
    
    return embeddings

def get_milvus_retriever(uri, embeddings, collection_name="HR_4Plus"):
    """
    Get retriever from the following vector db (Milvus)
    """

    # The vectorstore to use to index the summaries
    vectorstore = Milvus(
        embeddings,
        connection_args={"uri": uri},
        collection_name=collection_name,
    )

    return vectorstore