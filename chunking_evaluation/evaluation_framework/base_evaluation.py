from typing import Callable, List, Tuple
import os
import pandas as pd
import json
import chromadb
import numpy as np
from importlib import resources
from chromadb.utils import embedding_functions
from pathlib import Path
from tqdm import tqdm
from bs4 import BeautifulSoup
import re
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

def sum_of_ranges(ranges):
    return sum(end - start for start, end in ranges)

def union_ranges(ranges):
    sorted_ranges = sorted(ranges, key=lambda x: x[0])
    
    if not sorted_ranges:
        return []
        
    merged_ranges = [sorted_ranges[0]]
    
    for current_start, current_end in sorted_ranges[1:]:
        last_start, last_end = merged_ranges[-1]
        
        if current_start <= last_end:
            merged_ranges[-1] = (last_start, max(last_end, current_end))
        else:
            merged_ranges.append((current_start, current_end))
    
    return merged_ranges

def intersect_two_ranges(range1, range2):
    start1, end1 = range1
    start2, end2 = range2
    
    intersect_start = max(start1, start2)
    intersect_end = min(end1, end2)
    
    if intersect_start <= intersect_end:
        return (intersect_start, intersect_end)
    else:
        return None
    
def difference(ranges, target):
    result = []
    target_start, target_end = target

    for start, end in ranges:
        if end < target_start or start > target_end:
            result.append((start, end))
        elif start < target_start and end > target_end:
            result.append((start, target_start))
            result.append((target_end, end))
        elif start < target_start:
            result.append((start, target_start))
        elif end > target_end:
            result.append((target_end, end))

    return result

def find_target_in_document(document, target):
    start_index = document.find(target)
    if start_index == -1:
        return None
    end_index = start_index + len(target)
    return start_index, end_index

class BaseEvaluation:
    def __init__(self, questions_csv_path: str, chroma_db_path=None, corpora_id_paths=None):
        self.corpora_id_paths = corpora_id_paths
        self.questions_csv_path = questions_csv_path
        self.corpus_list = []
        self._load_questions_df()

        if chroma_db_path is not None:
            self.chroma_client = chromadb.PersistentClient(path=chroma_db_path)
        else:
            self.chroma_client = chromadb.Client()

        self.is_general = False
        
        self.bge_m3_embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="BAAI/bge-m3"
        )

    def _load_questions_df(self):
        if os.path.exists(self.questions_csv_path):
            self.questions_df = pd.read_csv(self.questions_csv_path)
        else:
            print(f"Warning: CSV file not found at {self.questions_csv_path}")
            self.questions_df = pd.DataFrame(columns=['question', 'references', 'corpus_id'])
        
        if 'corpus_id' in self.questions_df.columns and len(self.questions_df) > 0:
            self.corpus_list = self.questions_df['corpus_id'].unique().tolist()
        else:
            self.corpus_list = []

    def _read_and_process_corpus(self, corpus_id: str) -> str:
        """
        讀取原始檔案並進行預處理，特別是處理 HTML 檔案。
        """
        file_path = self.corpora_id_paths.get(os.path.basename(corpus_id))
        if not file_path:
            raise FileNotFoundError(f"Corpus ID '{corpus_id}' 的路徑未找到。")
            
        file_name = os.path.basename(str(file_path))

        with open(file_path, 'r', encoding='utf-8') as f:
            raw_content = f.read()

        if file_name.endswith('.html'):
            print(f"➡️ 處理 HTML 檔案: {file_name}。使用 BeautifulSoup 提取純文字...")
            soup = BeautifulSoup(raw_content, 'html.parser')
            text_content = soup.get_text()
        else:
            text_content = raw_content
        
        print(f"提取後的純文字大小（字元數）: {len(text_content)}")
        return text_content
    
    def _get_chunks_from_documents(self, splitter) -> tuple[list[str], list[dict]]:
        """
        【已修正】
        此方法會返回兩個列表：
        1. 區塊文字內容 (list[str])
        2. 區塊中繼資料，包含 `start_index` 和 `end_index` (list[dict])
        
        修正後，我們使用 `splitter.split_documents` 方法，並在之後手動將索引資訊加入元資料。
        """
        all_documents = []
        all_metadatas = []
        corpus_ids_to_process = self.questions_df['corpus_id'].unique()
        
        for i, corpus_id in enumerate(tqdm(corpus_ids_to_process, desc="✅ 正在處理語料庫檔案並切割")):
            try:
                file_path = self.corpora_id_paths[os.path.basename(corpus_id)]
                file_name = os.path.basename(file_path)
                
                text_content = self._read_and_process_corpus(corpus_id)
                
                # 使用 `splitter.create_documents` 來創建 Document 物件
                doc = Document(page_content=text_content, metadata={"corpus_id": corpus_id})
                chunks = splitter.split_documents([doc])

                # 手動計算並添加索引
                current_position = 0
                for chunk in chunks:
                    chunk_text = chunk.page_content
                    start_index = text_content.find(chunk_text, current_position)
                    
                    if start_index != -1:
                        end_index = start_index + len(chunk_text)
                        
                        all_documents.append(chunk_text)
                        all_metadatas.append({
                            "corpus_id": corpus_id,
                            "start_index": start_index,
                            "end_index": end_index
                        })
                        current_position = end_index + 1 # 從下一個位置開始搜尋
                    else:
                        print(f"⚠️ 警告: 找不到精確匹配的區塊。此區塊將被跳過。")
                        
                print(f"✔️ 檔案 '{file_name}' 已切割成 {len(chunks)} 個區塊，並生成 {len(all_metadatas)} 個中繼資料。")

            except FileNotFoundError as e:
                print(e)
                continue
            except KeyError:
                print(f"Error: Path for corpus_id '{corpus_id}' not found in corpora_id_paths.")
                continue

        return all_documents, all_metadatas

    def _full_precision_score(self, chunk_metadatas):
        ioc_scores = []
        recall_scores = []
        highlighted_chunks_count = []
        for index, row in self.questions_df.iterrows():
            references = row['references']
            corpus_id = row['corpus_id']
            
            if not isinstance(references, list) or not all(isinstance(r, dict) for r in references):
                print(f"Warning: Invalid format for references in row {index}. Skipping or assigning zero scores.")
                ioc_scores.append(0)
                recall_scores.append(0)
                highlighted_chunks_count.append(0)
                continue
                
            ioc_score = 0
            numerator_sets = []
            denominator_chunks_sets = []
            unused_highlights = [(x['start_index'], x['end_index']) for x in references]
            highlighted_chunk_count = 0
            for metadata in chunk_metadatas:
                chunk_start, chunk_end, chunk_corpus_id = metadata['start_index'], metadata['end_index'], metadata['corpus_id']
                if chunk_corpus_id != corpus_id:
                    continue
                contains_highlight = False
                for ref_obj in references:
                    ref_start, ref_end = int(ref_obj['start_index']), int(ref_obj['end_index'])
                    intersection = intersect_two_ranges((chunk_start, chunk_end), (ref_start, ref_end))
                    if intersection is not None:
                        contains_highlight = True
                        unused_highlights = difference(unused_highlights, intersection)
                        numerator_sets = union_ranges([intersection] + numerator_sets)
                        denominator_chunks_sets = union_ranges([(chunk_start, chunk_end)] + denominator_chunks_sets)
                if contains_highlight:
                    highlighted_chunk_count += 1
            highlighted_chunks_count.append(highlighted_chunk_count)
            denominator_sets = union_ranges(denominator_chunks_sets + unused_highlights)
            if numerator_sets:
                ioc_score = sum_of_ranges(numerator_sets) / sum_of_ranges(denominator_sets)
            ioc_scores.append(ioc_score)
            recall_score = 1 - (sum_of_ranges(unused_highlights) / sum_of_ranges([(x['start_index'], x['end_index']) for x in references]))
            recall_scores.append(recall_score)
        return ioc_scores, highlighted_chunks_count

    def _scores_from_dataset_and_retrievals(self, question_metadatas, highlighted_chunks_count):
        iou_scores = []
        recall_scores = []
        precision_scores = []
        for (index, row), highlighted_chunk_count, metadatas in zip(self.questions_df.iterrows(), highlighted_chunks_count, question_metadatas):
            references = row['references']
            corpus_id = row['corpus_id']

            numerator_sets = []
            denominator_chunks_sets = []
            unused_highlights = [(x['start_index'], x['end_index']) for x in references]

            for metadata in metadatas[:highlighted_chunk_count]:
                chunk_start, chunk_end, chunk_corpus_id = metadata['start_index'], metadata['end_index'], metadata['corpus_id']

                if chunk_corpus_id != corpus_id:
                    continue
                
                for ref_obj in references:
                    ref_start, ref_end = int(ref_obj['start_index']), int(ref_obj['end_index'])
                    
                    intersection = intersect_two_ranges((chunk_start, chunk_end), (ref_start, ref_end))
                    
                    if intersection is not None:
                        unused_highlights = difference(unused_highlights, intersection)
                        numerator_sets = union_ranges([intersection] + numerator_sets)
                        denominator_chunks_sets = union_ranges([(chunk_start, chunk_end)] + denominator_chunks_sets)
            
            if numerator_sets:
                numerator_value = sum_of_ranges(numerator_sets)
            else:
                numerator_value = 0

            recall_denominator = sum_of_ranges([(x['start_index'], x['end_index']) for x in references])
            precision_denominator = sum_of_ranges([(x['start_index'], x['end_index']) for x in metadatas[:highlighted_chunk_count]])
            iou_denominator = precision_denominator + sum_of_ranges(unused_highlights)

            recall_score = numerator_value / recall_denominator if recall_denominator > 0 else 0
            recall_scores.append(recall_score)

            precision_score = numerator_value / precision_denominator if precision_denominator > 0 else 0
            precision_scores.append(precision_score)

            iou_score = numerator_value / iou_denominator if iou_denominator > 0 else 0
            iou_scores.append(iou_score)

        return iou_scores, recall_scores, precision_scores

    def _create_collection_from_chunker(self, chunker, embedding_function, chroma_db_path:str = None, collection_name:str = None):
        if chroma_db_path is not None and collection_name is not None:
            try:
                chunk_client = chromadb.PersistentClient(path=chroma_db_path)
                try:
                    chunk_client.delete_collection(collection_name)
                    print(f"已刪除舊的 collection: {collection_name}")
                except Exception:
                    pass
                
                collection = chunk_client.create_collection(collection_name, embedding_function=embedding_function, metadata={"hnsw:search_ef":50})
                print("Created collection: ", collection_name)
            except Exception as e:
                print(f"Failed to create collection: {e}")
                return None, None
        else:
            collection_name = "auto_chunk"
            try:
                self.chroma_client.delete_collection(collection_name)
                print(f"Existing collection '{collection_name}' deleted.")
            except chromadb.errors.NotFoundError:
                print(f"Collection '{collection_name}' does not exist, proceeding to create.")
            except Exception as e:
                print(f"Warning: Could not delete collection {collection_name} due to an unexpected error: {e}")
                pass

            collection = self.chroma_client.create_collection(
                collection_name,
                embedding_function=embedding_function,
                metadata={"hnsw:search_ef":50}
            )
            print(f"New collection '{collection_name}' created.")

        docs, metas = self._get_chunks_from_documents(chunker)
        
        if not docs:
            print("❌ 錯誤：沒有找到可處理的區塊。")
            return None, None
        
        BATCH_SIZE = 500
        for i in tqdm(range(0, len(docs), BATCH_SIZE), desc=f"✅ 處理中: {len(docs)} 個區塊"):
            batch_docs = docs[i:i+BATCH_SIZE]
            batch_metas = metas[i:i+BATCH_SIZE]
            batch_ids = [str(j) for j in range(i, i+len(batch_docs))]
            collection.add(
                documents=batch_docs,
                metadatas=batch_metas,
                ids=batch_ids
            )

        return collection, metas

    def _convert_question_references_to_json(self):
        def safe_json_loads(row):
            if pd.isna(row):
                return None
            try:
                if isinstance(row, str):
                    row = row.replace('""', '"')
                return json.loads(row, strict=False)
            except json.JSONDecodeError as e:
                print(f"JSONDecodeError: {e} for string: {row}")
                return None
            except:
                return None

        self.questions_df['references'] = self.questions_df['references'].apply(safe_json_loads)

    def run(self, chunker, embedding_function: Callable = None, retrieve: int = 5, db_to_save_chunks: str = None):
        self._load_questions_df()
        self._convert_question_references_to_json() 

        if embedding_function is None:
            print(f"use get_bge_m3_embedding_function()")
            embedding_function = self.bge_m3_embedding_function

        collection = None
        all_chunks_with_metas = None
        
        if db_to_save_chunks is not None:
            chunk_size = chunker._chunk_size if hasattr(chunker, '_chunk_size') else "0"
            chunk_overlap = chunker._chunk_overlap if hasattr(chunker, '_chunk_overlap') else "0"
            
            embedding_function_name = "BGE_M3_EmbeddingFunction"
            collection_name = embedding_function_name + '_' + chunker.__class__.__name__ + '_' + str(int(chunk_size)) + '_' + str(int(chunk_overlap))
            try:
                chunk_client = chromadb.PersistentClient(path=db_to_save_chunks)
                existing_collections = chunk_client.list_collections()
                collection_names = [col.name for col in existing_collections]

                if collection_name in collection_names:
                    print(f"✅ Collection '{collection_name}' 已存在。正在使用現有集合。")
                    collection = chunk_client.get_collection(collection_name, embedding_function=embedding_function)
                    
                    all_chunks_with_metas = collection.get(include=["metadatas"])['metadatas']
                else:
                    collection, all_chunks_with_metas = self._create_collection_from_chunker(chunker, embedding_function, chroma_db_path=db_to_save_chunks, collection_name=collection_name)
            except Exception as e:
                print(f"Error accessing collection: {e}")
                collection, all_chunks_with_metas = self._create_collection_from_chunker(chunker, embedding_function, chroma_db_path=db_to_save_chunks, collection_name=collection_name)

        if collection is None or all_chunks_with_metas is None:
            collection, all_chunks_with_metas = self._create_collection_from_chunker(chunker, embedding_function)
        
        if all_chunks_with_metas is None:
            print("❌ 錯誤：無法獲取區塊中繼資料，評估將無法進行。")
            return {}
        
        if self.questions_df.empty:
            print("❌ 致命錯誤：`questions_df` 是空的。沒有要添加到集合中的查詢。")
            return {}

        try:
            self.chroma_client.delete_collection("auto_questions")
            print("Existing 'auto_questions' collection deleted.")
        except Exception as e:
            print(f"Collection 'auto_questions' did not exist or could not be deleted: {e}")
            pass

        question_collection = self.chroma_client.create_collection(
            "auto_questions",
            embedding_function=embedding_function,
            metadata={"hnsw:search_ef":50}
        )
        print("New 'auto_questions' collection created.")
        
        question_collection.add(
            documents=self.questions_df['question'].tolist(),
            metadatas=[{"corpus_id": x} for x in self.questions_df['corpus_id'].tolist()],
            ids=[str(i) for i in self.questions_df.index]
        )
        print(f"{len(self.questions_df)} questions added to 'auto_questions' collection.")
        
        question_db = question_collection.get(include=['embeddings'])
        question_db['ids'] = [int(id) for id in question_db['ids']]
        _, sorted_embeddings = zip(*sorted(zip(question_db['ids'], question_db['embeddings'])))
        self.questions_df = self.questions_df.sort_index()

        brute_iou_scores, highlighted_chunks_count = self._full_precision_score(all_chunks_with_metas)

        if retrieve == -1:
            maximum_n = min(20, max(highlighted_chunks_count))
        else:
            highlighted_chunks_count = [retrieve] * len(highlighted_chunks_count)
            maximum_n = retrieve

        retrievals = collection.query(query_embeddings=list(sorted_embeddings), n_results=maximum_n)

        iou_scores, recall_scores, precision_scores = self._scores_from_dataset_and_retrievals(retrievals['metadatas'], highlighted_chunks_count)

        corpora_scores = {}
        for index, row in self.questions_df.iterrows():
            if row['corpus_id'] not in corpora_scores:
                corpora_scores[row['corpus_id']] = {
                    "precision_omega_scores": [],
                    "iou_scores": [],
                    "recall_scores": [],
                    "precision_scores": []
                }
            
            corpora_scores[row['corpus_id']]['precision_omega_scores'].append(brute_iou_scores[index])
            corpora_scores[row['corpus_id']]['iou_scores'].append(iou_scores[index])
            corpora_scores[row['corpus_id']]['recall_scores'].append(recall_scores[index])
            corpora_scores[row['corpus_id']]['precision_scores'].append(precision_scores[index])

        brute_iou_mean = np.mean(brute_iou_scores)
        brute_iou_std = np.std(brute_iou_scores)

        recall_mean = np.mean(recall_scores)
        recall_std = np.std(recall_scores)

        iou_mean = np.mean(iou_scores)
        iou_std = np.std(iou_scores)

        precision_mean = np.mean(precision_scores)
        precision_std = np.std(precision_scores)

        return {
            "corpora_scores": corpora_scores,
            "iou_mean": iou_mean,
            "iou_std": iou_std,
            "recall_mean": recall_mean,
            "recall_std": recall_std,
            "precision_omega_mean": brute_iou_mean,
            "precision_omega_std": brute_iou_std,
            "precision_mean": precision_mean,
            "precision_std": precision_std
        }