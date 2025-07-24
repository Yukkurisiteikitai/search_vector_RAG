import os
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain.memory import VectorStoreRetrieverMemory
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate

# --- 1. 必要なコンポーネントの準備 ---

# LM StudioのEmbeddingモデルを使う場合:
# OpenAIEmbeddingsの代わりに、LM Studioのサーバーを指す設定を行う
# 例: embeddings = OpenAIEmbeddings(base_url="http://localhost:1234/v1", api_key="not-needed")
# ※モデルがOpenAI互換のAPIを提供している必要があります。
embeddings = OpenAIEmbeddings()

# ベクトルストアの準備（Chromaを使用、データは 'conversation_db' フォルダに永続化）
vectorstore = Chroma("conversation_history", embeddings, persist_directory="./conversation_db")

# レトリーバー（検索エンジン）の作成
# search_kwargs={"k": 2} は、関連する過去の会話を2つまで見つける設定
retriever = vectorstore.as_retriever(search_kwargs=dict(k=2))

# VectorStoreRetrieverMemoryのインスタンスを作成
# memory_key="history" は、プロンプト内で過去の履歴を参照する際のキー名
memory = VectorStoreRetrieverMemory(retriever=retriever, memory_key="history")


# --- 2. LLMとプロンプト、そしてチェーンの作成 ---

# LM Studioのチャットモデルを使う場合:
# llm = OpenAI(base_url="http://localhost:1234/v1", api_key="not-needed", temperature=0)
llm = OpenAI(temperature=0)

# プロンプトテンプレート：{history} と {input} が埋め込まれる
template = """
You are an AI chatbot having a conversation with a human.
Given the following extracted parts of a long conversation and a question, create a final answer.

{history}

Human: {input}
AI:"""

prompt = PromptTemplate(
    input_variables=["history", "input"], template=template
)

# 会話チェーンの作成
chain = ConversationChain(
    llm=llm,
    prompt=prompt,
    memory=memory,
    verbose=True # 実行時に何が起きているか詳細に出力
)


# --- 3. 会話の実行 ---

# 1回目の会話
chain.invoke({"input": "こんにちは！私の名前は健太です。LangChainについて学びたいです。"})

# 2回目の会話（LangChainについて）
chain.invoke({"input": "LangChainで特に重要な概念は何ですか？"})

# 3回目の会話（全く関係ない話題）
chain.invoke({"input": "今日のランチは何がいいかな？"})

# 4回目の会話（過去の話題に戻る）
# ここで "私の名前" と "重要な概念" という過去の情報を参照できるかが見どころ
# VectorStoreRetrieverMemoryが、1回目と2回目の会話を検索してプロンプトに含めてくれるはず
chain.invoke({"input": "先ほど話した重要な概念について、私の名前を呼んで、もう一度説明してくれますか？"})

# Chromaに保存したデータを永続化
vectorstore.persist()