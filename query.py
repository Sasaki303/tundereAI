import os
from langchain_community.chat_models import ChatOllama
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever
from get_vector_db import get_vector_db

# 環境変数から LLM モデルを取得（デフォルトは 'mistral'）
LLM_MODEL = os.getenv('LLM_MODEL', 'tundere-ai:Q4_K_M')

# プロンプトテンプレートを取得する関数
def get_prompt():
    QUERY_PROMPT = PromptTemplate(
        input_variables=["question"],
        template="""You are an AI language model assistant. Your task is to generate 
        five different versions of the given user question to retrieve relevant 
        documents from a vector database. By generating multiple perspectives on 
        the user question, your goal is to help the user overcome some of the 
        limitations of the distance-based similarity search. 
        Provide these alternative questions separated by newlines. 
        Original question: {question}"""
    )

    template = """Answer the question based ONLY on the following context: {context} 
    Question: {question}"""
    
    prompt = ChatPromptTemplate.from_template(template)
    
    return QUERY_PROMPT, prompt

# クエリ処理を実行する関数
def query(input):
    if not input:
        return None

    # Ollama の LLM モデルを初期化
    llm = ChatOllama(model=LLM_MODEL)

    # ベクトルデータベースを取得
    db = get_vector_db()

    # プロンプトテンプレートを取得
    QUERY_PROMPT, prompt = get_prompt()

    # MultiQueryRetriever を設定
    retriever = MultiQueryRetriever.from_llm(
        retriever=db.as_retriever(),
        llm=llm,
        prompt=QUERY_PROMPT
    )

    # チェーンを設定（Retriever → プロンプト → LLM → 出力パーサー）
    chain = (
        {"context": retriever, "question": RunnablePassthrough()} 
        | prompt 
        | llm 
        | StrOutputParser()
    )

    # クエリを実行し、応答を取得
    response = chain.invoke(input)

    return response
