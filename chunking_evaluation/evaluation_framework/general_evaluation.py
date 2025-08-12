import os
import pandas as pd
from .base_evaluation import BaseEvaluation
from langchain.text_splitter import RecursiveCharacterTextSplitter

class GeneralEvaluation(BaseEvaluation):
    def __init__(self):
        """
        初始化 GeneralEvaluation 類別。
        它會自動使用固定的路徑來讀取問題 CSV 和語料庫。
        """
        # 設置固定路徑
        current_dir = os.path.dirname(os.path.abspath(__file__))
        evaluation_data_path = os.path.join(current_dir, 'general_evaluation_data')
        
        questions_csv_path = os.path.join(evaluation_data_path, 'generated_queries_and_excerpts.csv')
        corpora_dir = os.path.join(evaluation_data_path, 'corpora')
        
        # 建立 corpora_id_paths 字典
        corpora_id_paths = {os.path.basename(f): os.path.join(corpora_dir, f) for f in os.listdir(corpora_dir)}

        # 呼叫父類別的建構函式，傳入固定的路徑
        super().__init__(questions_csv_path=questions_csv_path, corpora_id_paths=corpora_id_paths)