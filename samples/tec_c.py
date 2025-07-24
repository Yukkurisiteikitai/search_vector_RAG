import os
from langchain_community.vectorstores import Chroma
# LangChainでHugging FaceのEmbeddingモデルを扱うためのクラス
from langchain_community.embeddings import HuggingFaceEmbeddings
# LM Studioのローカルサーバーと通信するためのクラス
from langchain_openai import OpenAI 
from langchain.memory import VectorStoreRetrieverMemory
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate

# --- 1. 準備：Embeddingモデルとベクトルストア ---

print("Embeddingモデルをロードしています... (初回は時間がかかる場合があります)")
# ご指定のモデルをLangChainのラッパー経由で読み込む
# これで、LangChainの様々な機能と互換性が生まれる
model_name = "text-embedding-all-minilm-l6-v2"
model_kwargs = {'device': 'cpu'} # CPUで実行する場合
encode_kwargs = {'normalize_embeddings': False}
embeddings = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)
print("Embeddingモデルのロードが完了しました。")


# ベクトルストアの準備 (ChromaDBを使用)
# データを 'conversation_db' フォルダに永続化する設定
vectorstore = Chroma("conversation_history_st", embedding_function=embeddings, persist_directory="./conversation_db_st")

# レトリーバー（検索エンジン）の作成
# search_kwargs={"k": 2} は、関連する過去の会話を2つまで見つける設定
retriever = vectorstore.as_retriever(search_kwargs=dict(k=2))

# VectorStoreRetrieverMemoryのインスタンスを作成
# memory_key="history" は、プロンプト内で過去の履歴を参照する際のキー名
memory = VectorStoreRetrieverMemory(retriever=retriever, memory_key="history")


# --- 2. LLMとプロンプト、そしてチェーンの作成 ---

# LM Studioのサーバーに接続 (ポート番号はLM Studioの設定に合わせてください)
# api_keyは何でも良いので "not-needed" などを入れておく
llm = OpenAI(base_url="http://localhost:1234/v1", api_key="not-needed", temperature=0.7)

# プロンプトテンプレート：{history} と {input} が埋め込まれる
template = """
あなたはフレンドリーなAIアシスタントです。
人間との会話の過去の関連部分と、新しい質問が与えられます。
過去の会話の流れを踏まえて、自然な形で回答を生成してください。

過去の関連する会話:
{history}

新しい質問: {input}
AI:"""

prompt = PromptTemplate(
    input_variables=["history", "input"], template=template
)

# 会話チェーンの作成
chain = ConversationChain(
    llm=llm,
    prompt=prompt,
    memory=memory,
    verbose=True # 実行時にプロンプトの内容などを詳細に出力
)


# --- 3. 会話の実行 ---

print("\n--- 会話を開始します ---")

# 1回目の会話
chain.invoke({"input": "こんにちは！私の名前は佐藤です。最近、家庭菜園に興味があります。"})

# 2回目の会話（家庭菜園について）
chain.invoke({"input": "初心者におすすめの野菜は何ですか？ミニトマトとか良さそうですね。"})

# 3回目の会話（全く関係ない話題）
chain.invoke({"input": "ところで、昨日のサッカーの試合見ましたか？"})

# 4回目の会話（過去の話題に戻る）
# ここで "私の名前" と "おすすめの野菜" という過去の情報を参照できるかが見どころ
# VectorStoreRetrieverMemoryが、1回目と2回目の会話を検索してプロンプトに含めてくれるはず
print("\n--- 過去の情報を参照する質問をします ---")
chain.invoke({"input": "先ほど話したおすすめの野菜について、私の名前を呼んで、もう一度詳しく教えてくれませんか？"})

# Chromaに保存したデータを永続化（プログラム終了時に保存）
vectorstore.persist()
print("\n会話履歴をDBに永続化しました。")