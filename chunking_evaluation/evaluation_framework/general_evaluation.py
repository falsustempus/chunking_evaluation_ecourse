from .base_evaluation import BaseEvaluation
from importlib import resources
from pathlib import Path
import os

from .base_evaluation import BaseEvaluation
from importlib import resources
from pathlib import Path
import os
import pandas as pd

class GeneralEvaluation(BaseEvaluation):
    def __init__(self, chroma_db_path=None):
        # 取得專案根目錄（即你執行 Jupyter Notebook 的目錄）
        project_root = Path(os.getcwd())

        # 取得 general_benchmark_path 的絕對路徑
        with resources.as_file(resources.files('chunking_evaluation.evaluation_framework') / 'general_evaluation_data') as general_benchmark_path:
            self.general_benchmark_path = general_benchmark_path
            
            # questions_df_path 應該是絕對路徑
            questions_df_path = self.general_benchmark_path / 'generated_queries_and_excerpts.csv'

            corpora_folder_path = self.general_benchmark_path / 'corpora'
            corpora_filenames = [f for f in corpora_folder_path.iterdir() if f.is_file()]
            
            corpora_id_paths = {}
            for file_path in corpora_filenames:
                # 建立相對於專案根目錄的相對路徑作為字典的鍵
                relative_path_key = str(file_path.relative_to(project_root)).replace('\\', '/')
                corpora_id_paths[relative_path_key] = str(file_path)

            super().__init__(questions_csv_path=questions_df_path, chroma_db_path=chroma_db_path, corpora_id_paths=corpora_id_paths)

            self.is_general = True