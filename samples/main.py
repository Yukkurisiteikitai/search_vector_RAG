import chromadb
from sentence_transformers import SentenceTransformer

# 1. 準備：モデルとVectorDBクライアントの用意
# オープンソースのモデルを読み込む (方法B)
model = SentenceTransformer('all-MiniLM-L6-v2') 
client = chromadb.Client() # ChromaDBのクライアントを作成

# "my_collection"という名前のDB（コレクション）を作成または取得
collection = client.get_or_create_collection(name="my_collection")

# 2. データのベクトル化とDBへの保存
documents = [
    "猫はかわいい動物です。",
    "犬は人間の最高の友達です。",
    "今日は良い天気ですね。",
    "AI技術は急速に進歩しています。"
]
# 各ドキュメントをベクトルに変換
embeddings = model.encode(documents).tolist() 

# ドキュメントとIDを一緒にDBへ保存
collection.add(
    embeddings=embeddings,
    documents=documents,
    ids=[f"id_{i}" for i in range(len(documents))]
)

print("--- データベースにデータを保存しました ---")

# 3. 検索クエリで類似データを「読み込む」
query_text = "私の好きなペットは何ですか？"

# 検索クエリも同じモデルでベクトル化
query_embedding = model.encode(query_text).tolist()

# DBに問い合わせて、類似度の高いドキュメントを2つ検索
results = collection.query(
    query_embeddings=[query_embedding],
    n_results=2
)

print("\n--- 検索結果（読み込まれたデータ） ---")

print(client)
print(results['documents'])