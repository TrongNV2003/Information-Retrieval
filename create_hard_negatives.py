import json
from tqdm import tqdm
from typing import List
from rank_bm25 import BM25Okapi
from transformers import AutoTokenizer


def load_json_data(file_path: str) -> List[dict]:
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def prepare_corpus_for_bm25(corpus: List[dict], tokenizer: AutoTokenizer, max_length: int = 512) -> tuple:
    """
    Tokenize các tài liệu và tạo danh sách các document.
    
    Args:
        corpus (List[dict]): Danh sách các tài liệu từ corpus.json.
        tokenizer: Tokenizer để xử lý văn bản.
        max_length (int): Độ dài tối đa của văn bản sau khi tokenize.
    
    Returns:
        tuple: (tokenized_corpus, doc_ids, corpus_texts)
    """
    tokenized_corpus = []
    doc_ids = []
    corpus_texts = []
    
    for doc in tqdm(corpus, desc="Tokenizing corpus", unit="doc"):
        text = f"{doc['title']}\n{doc['text']}"
        tokens = tokenizer(text, truncation=True, max_length=max_length, return_tensors='pt')['input_ids'][0]
        tokenized_text = tokenizer.decode(tokens, skip_special_tokens=True).split()
        tokenized_corpus.append(tokenized_text)
        doc_ids.append((doc['law_id'], doc['article_id']))
        corpus_texts.append(text)
    
    return tokenized_corpus, doc_ids, corpus_texts

def generate_hard_negatives(
    train_data: List[dict],
    corpus: List[dict],
    num_negatives: int = 3,
    max_length: int = 512
) -> List[dict]:
    """
    Tạo hard negatives cho mỗi mẫu huấn luyện sử dụng BM25.
    
    Args:
        train_data (List[dict]): Dữ liệu huấn luyện từ split_data.json.
        corpus (List[dict]): Corpus chứa tất cả các tài liệu.
        num_negatives (int): Số lượng hard negatives cần tạo cho mỗi mẫu.
        max_length (int): Độ dài tối đa của văn bản.
    
    Returns:
        List[dict]: Danh sách các mẫu huấn luyện với hard negatives.
    """
    tokenizer = AutoTokenizer.from_pretrained('hiieu/halong_embedding')
    
    tokenized_corpus, doc_ids, corpus_texts = prepare_corpus_for_bm25(corpus, tokenizer, max_length)
    bm25 = BM25Okapi(tokenized_corpus)
    
    new_train_data = []
    
    for sample in tqdm(train_data, desc="Generating hard negatives", unit="sample"):
        question = sample['question']
        positive_article = sample['relevant_articles'][0]
        positive_id = (positive_article['law_id'], positive_article['article_id'])
        
        query_tokens = tokenizer(question, truncation=True, max_length=max_length, return_tensors='pt')['input_ids'][0]
        query_tokens = tokenizer.decode(query_tokens, skip_special_tokens=True).split()
        
        scores = bm25.get_scores(query_tokens)
        
        # Lấy top-k tài liệu có điểm cao nhất
        top_k_indices = scores.argsort()[-num_negatives-1:][::-1]
        
        hard_negatives = []
        for idx in top_k_indices:
            if doc_ids[idx] != positive_id:
                hard_negatives.append({
                    "law_id": corpus[idx]['law_id'],
                    "article_id": corpus[idx]['article_id'],
                    "title": corpus[idx]['title'],
                    "text": corpus[idx]['text']
                })
            if len(hard_negatives) >= num_negatives:
                break
        
        new_sample = {
            "question": sample['question'],
            "relevant_articles": sample['relevant_articles'],
            "num_articles": sample['num_articles'],
            "relevant_titles": sample['relevant_titles'],
            "hard_negatives": hard_negatives
        }
        new_train_data.append(new_sample)
    
    return new_train_data

def save_json_data(data: List[dict], output_file: str):
    """
    Lưu dữ liệu vào file JSON.
    
    Args:
        data (List[dict]): Dữ liệu cần lưu.
        output_file (str): Đường dẫn đến file JSON đầu ra.
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    print(f"Đã lưu {len(data)} mẫu vào {output_file}")

def main():
    train_file = 'dataset/ZaloTextRetrieval/dataset_norm/train.json'
    corpus_file = 'dataset/legal_corpus_docs.json'
    output_file = 'dataset/ZaloTextRetrieval/dataset_negatives/train_negatives.json'

    train_data = load_json_data(train_file)
    corpus = load_json_data(corpus_file)
    
    new_train_data = generate_hard_negatives(train_data, corpus, num_negatives=3)
    
    print(f"Tổng số mẫu: {len(new_train_data)}")
    for i, sample in enumerate(new_train_data[:2]):
        print(f"Mẫu {i+1}:")
        print(f"Question: {sample['question']}")
        print(f"Positive Article: {sample['relevant_articles']}")
        print(f"Hard Negatives: {sample['hard_negatives']}")
        print("-" * 50)
    
    save_json_data(new_train_data, output_file)

if __name__ == "__main__":
    main()