from .base_evaluation import BaseEvaluation
from importlib import resources
from pathlib import Path
import os

class GeneralEvaluation(BaseEvaluation):
    def __init__(self, chroma_db_path=None):
        with resources.as_file(resources.files('chunking_evaluation.evaluation_framework') / 'general_evaluation_data') as general_benchmark_path:
            self.general_benchmark_path = general_benchmark_path
            
            # questions_df_path 應該是絕對路徑
            questions_df_path = str(self.general_benchmark_path / 'generated_queries_and_excerpts.csv')

            corpora_folder_path = self.general_benchmark_path / 'corpora'
            corpora_filenames = [f for f in corpora_folder_path.iterdir() if f.is_file()]
            
            corpora_id_paths = {}
            for file_path in corpora_filenames:
                # 取得相對於 general_benchmark_path 的相對路徑
                relative_path = os.path.relpath(file_path, start=self.general_benchmark_path.parent)
                
                # 標準化路徑，確保使用正斜線作為鍵
                standardized_path = str(relative_path).replace('\\', '/')
                corpora_id_paths[standardized_path] = str(file_path)

            super().__init__(questions_csv_path=questions_df_path, chroma_db_path=chroma_db_path, corpora_id_paths=corpora_id_paths)

            self.is_general = True