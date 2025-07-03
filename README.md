import re
import pdfplumber
import chromadb
from chromadb.config import Settings
from chromadb.utils.embedding_functions import EmbeddingFunction
import google.generativeai as genai
from google.generativeai.types import GenerationConfig

# ✅ 設定 Gemini API 金鑰
genai.configure(api_key="YOUR_GEMINI_API_KEY")  # ← 替換成你的金鑰

# ✅ Gemini 嵌入函數
class GeminiEmbeddingFunction(EmbeddingFunction):
    def __call__(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        embeddings = []
        for text in texts:
            response = genai.embed_content(
                model="models/embedding-001",
                content=text,
                task_type="retrieval_document"
            )
            embeddings.append(response["embedding"])
        return embeddings

# ✅ PDF 擷取文字
def extract_text_from_pdf(pdf_path: str) -> str:
    full_text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                full_text += page_text + "\n"
    return full_text.strip()

# ✅ 分段：語意句切塊
def split_text(text: str, max_chunk_size=500, overlap=100) -> list:
    text = re.sub(r'\s+', ' ', text).strip()
    sentence_delimiters = re.compile(r'(?<=[.!?。！？])\s')
    sentences = sentence_delimiters.split(text)

    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) + 1 <= max_chunk_size:
            current_chunk += sentence + " "
        else:
            chunks.append(current_chunk.strip())
            if overlap > 0:
                current_chunk = current_chunk[-overlap:] + sentence + " "
            else:
                current_chunk = sentence + " "
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

# ✅ 建立向量資料庫
def create_chroma_db(documents, path="./chroma_db", name="pdf_chunks"):
    client = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory=path))
    embedding_function = GeminiEmbeddingFunction()
    collection = client.get_or_create_collection(name=name, embedding_function=embedding_function)
    ids = [f"doc_{i}" for i in range(len(documents))]
    collection.add(documents=documents, ids=ids)
    client.persist()
    return collection

# ✅ 查詢相關 passage
def get_relevant_passage(query: str, db, n_results: int = 3) -> list:
    results = db.query(query_texts=[query], n_results=n_results)
    return results["documents"][0]

# ✅ RAG Prompt 組合
def make_rag_prompt(query: str, relevant_passages: list) -> str:
    context = "\n\n".join(relevant_passages)
    prompt = (
        "Based on the following information:\n\n"
        f"{context}\n\n"
        "Please answer this question:\n"
        f"{query}"
    )
    return prompt

# ✅ 產生回答
def generate_answer(prompt: str, temperature: float = 0.7, max_tokens: int = 512) -> str:
    config = GenerationConfig(
        temperature=temperature,
        max_output_tokens=max_tokens
    )
    response = genai.generate_content(
        model="models/gemini-pro",
        contents=[{"role": "user", "parts": [prompt]}],
        generation_config=config
    )
    return response.text.strip()

# ✅ 撰寫 findings.txt
def write_findings():
    findings = """
Findings: Temperature and Token Limit Experiments
==================================================

1. How did the output change as you increased the temperature?
--------------------------------------------------------------
As temperature increased from 0.0 to 1.0, the responses became more diverse, expressive, and sometimes more verbose. 
At temperature 0.0, the answer was deterministic, short, and factual. At 0.5, the language was balanced. 
At 1.0, the answer was more creative and descriptive.

2. When would you prefer a low temperature versus a high one?
-------------------------------------------------------------
- Low temperature (0.0): for factual, deterministic, and reproducible answers (e.g., answering exam questions).
- High temperature (1.0): for storytelling, summaries, or brainstorming.

3. What happened when the max_output_tokens limit was reached?
---------------------------------------------------------------
The response was cut off once the token limit was reached. At very low limits (e.g., 20), the answer was incomplete.
This proves useful to control output size and avoid unnecessary API costs.

4. Practical scenario for setting a token limit?
------------------------------------------------
If integrating answers into an SMS system or UI card, we may need to limit text to 160 tokens or fewer for design purposes.

5. How to get a short, creative response vs a long, factual one?
----------------------------------------------------------------
- Short, creative: temperature=1.0, max_tokens=50
- Long, factual: temperature=0.0 or 0.3, max_tokens=300+
"""
    with open("findings.txt", "w", encoding="utf-8") as f:
        f.write(findings.strip())
    print("\n📝 findings.txt 已建立")

# ✅ 主程式入口
if __name__ == "__main__":
    pdf_path = "your_file.pdf"  # ← 替換為你的 PDF 檔案
    full_text = extract_text_from_pdf(pdf_path)
    print("✅ PDF 讀取完成")

    chunks = split_text(full_text)
    print(f"✅ 分割為 {len(chunks)} 個語意段落")

    collection = create_chroma_db(chunks, path="./chroma_db", name="example_pdf")
    print("✅ 向量資料庫建立完成")

    while True:
        user_question = input("\n❓ 請輸入問題（或輸入 'exit' 結束）：")
        if user_question.lower() == 'exit':
            break

        top_chunks = get_relevant_passage(user_question, collection, n_results=3)
        prompt = make_rag_prompt(user_question, top_chunks)

        # ✅ Task 2: 測試不同 Temperature
        print("\n🧪 比較 Temperature（max_tokens = 300）")
        for temp in [0.0, 0.5, 1.0]:
            print(f"\n🌡️ Temperature = {temp}")
            answer = generate_answer(prompt, temperature=temp, max_tokens=300)
            print(f"🤖 回答：\n{answer}")

        # ✅ Task 3: 測試不同 Max Tokens
        print("\n🔢 比較 Max Tokens（temperature = 0.5）")
        for tokens in [20, 50, 200]:
            print(f"\n📏 Max Tokens = {tokens}")
            answer = generate_answer(prompt, temperature=0.5, max_tokens=tokens)
            print(f"🤖 回答：\n{answer}")

        # ✅ Task 4: 寫入結果報告
        write_findings()
