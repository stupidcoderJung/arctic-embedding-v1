import torch
import lancedb
import gc
import os
from transformers import AutoModel, AutoTokenizer

def get_embeddings(text_list):
    """
    MacBook Air M1 (8GB) 최적화 임베딩 생성 함수.
    작업 후 즉시 메모리를 해제합니다.
    """
    model_name = "Snowflake/snowflake-arctic-embed-tiny"
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    # 모델 및 토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()
    
    # 임베딩 생성
    with torch.no_grad():
        inputs = tokenizer(text_list, padding=True, truncation=True, return_tensors="pt").to(device)
        outputs = model(**inputs)
        # Arctic Embed: CLS 토큰 사용
        embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
    
    # 메모리 즉시 반환 로직
    del model
    del tokenizer
    del inputs
    del outputs
    
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    
    gc.collect()
    return embeddings

def save_to_lancedb(data, table_name="memory_embeddings", db_path="/Users/young/.openclaw/workspace/memory-lancedb"):
    """
    OpenClaw LanceDB 연동 예시
    """
    db = lancedb.connect(db_path)
    if table_name in db.table_names():
        table = db.open_table(table_name)
        table.add(data)
    else:
        table = db.create_table(table_name, data=data)
    print(f"Table '{table_name}' updated.")

if __name__ == "__main__":
    test_texts = ["명령을 수행합니다.", "메모리 최적화 완료."]
    embeddings = get_embeddings(test_texts)
    
    # LanceDB 저장용 데이터 구조화
    data_to_save = [
        {"vector": emb.tolist(), "text": text} 
        for emb, text in zip(embeddings, test_texts)
    ]
    
    # save_to_lancedb(data_to_save)
    print(f"Embedding Dimensions: {embeddings.shape}")
    print("Memory Cleared.")
