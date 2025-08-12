from enum import Enum
import re
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
import os
import torch
from chromadb.utils import embedding_functions
import tiktoken
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer
from typing import Callable
from tqdm import tqdm


def find_query_despite_whitespace(document, query):

    # Normalize spaces and newlines in the query
    normalized_query = re.sub(r'\s+', ' ', query).strip()
    
    # Create a regex pattern from the normalized query to match any whitespace characters between words
    pattern = r'\s*'.join(re.escape(word) for word in normalized_query.split())
    
    # Compile the regex to ignore case and search for it in the document
    regex = re.compile(pattern, re.IGNORECASE)
    match = regex.search(document)
    
    if match:
        return document[match.start(): match.end()], match.start(), match.end()
    else:
        return None

def rigorous_document_search(document: str, target: str):
    """
    This function performs a rigorous search of a target string within a document. 
    It handles issues related to whitespace, changes in grammar, and other minor text alterations.
    The function first checks for an exact match of the target in the document. 
    If no exact match is found, it performs a raw search that accounts for variations in whitespace.
    If the raw search also fails, it splits the document into sentences and uses fuzzy matching 
    to find the sentence that best matches the target.
    
    Args:
        document (str): The document in which to search for the target.
        target (str): The string to search for within the document.

    Returns:
        tuple: A tuple containing the best match found in the document, its start index, and its end index.
        If no match is found, returns None.
    """
    if target.endswith('.'):
        target = target[:-1]
    
    if target in document:
        start_index = document.find(target)
        end_index = start_index + len(target)
        return target, start_index, end_index
    else:
        raw_search = find_query_despite_whitespace(document, target)
        if raw_search is not None:
            return raw_search

    # Split the text into sentences
    sentences = re.split(r'[.!?]\s*|\n', document)

    # Find the sentence that matches the query best
    best_match = process.extractOne(target, sentences, scorer=fuzz.token_sort_ratio)

    if best_match[1] < 98:
        return None
    
    reference = best_match[0]

    start_index = document.find(reference)
    end_index = start_index + len(reference)

    return reference, start_index, end_index

def get_bge_m3_embedding_function():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"正在載入 BGE-M3 模型到裝置: {device}")
    
    # 使用 SentenceTransformer 類別直接載入模型
    # 這可以讓您更好地控制模型，並確保它被明確地載入到 GPU
    model = SentenceTransformer("BAAI/bge-m3", device=device)
    
    # 創建一個自訂的嵌入函式，以便在處理時可以加入進度條
    class CustomEmbeddingFunction(embedding_functions.EmbeddingFunction):
        def __call__(self, texts: embedding_functions.Documents) -> embedding_functions.Embeddings:
            # 根據您的 GPU VRAM 大小，嘗試不同的 batch_size
            # 如果您的 4070 Ti SUPER 顯存充足 (16GB)，可以嘗試更大的值
            EMBEDDING_BATCH_SIZE = 16384

            # 使用一個進度條來追蹤 embedding 的進度
            # 將輸入的 texts 分成批次
            text_batches = [texts[i:i + EMBEDDING_BATCH_SIZE] for i in range(0, len(texts), EMBEDDING_BATCH_SIZE)]

            all_embeddings = []
            for batch in tqdm(text_batches, desc="生成嵌入向量中"):
                embeddings = model.encode(batch, convert_to_numpy=False)
                all_embeddings.extend(embeddings)

            return [emb.tolist() for emb in all_embeddings]
            
    return CustomEmbeddingFunction()

# Count the number of tokens in each page_content
def bge_m3_token_count(text: str) -> int:
    """
    使用 BAAI/bge-m3 模型的 tokenizer 來計算 token 數量。
    """
    tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-m3")
    tokens = tokenizer.encode(text)
    return len(tokens)

class Language(str, Enum):
    """Enum of the programming languages."""

    CPP = "cpp"
    GO = "go"
    JAVA = "java"
    KOTLIN = "kotlin"
    JS = "js"
    TS = "ts"
    PHP = "php"
    PROTO = "proto"
    PYTHON = "python"
    RST = "rst"
    RUBY = "ruby"
    RUST = "rust"
    SCALA = "scala"
    SWIFT = "swift"
    MARKDOWN = "markdown"
    LATEX = "latex"
    HTML = "html"
    SOL = "sol"
    CSHARP = "csharp"
    COBOL = "cobol"
    C = "c"
    LUA = "lua"
    PERL = "perl"