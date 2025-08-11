from typing import List, Dict, Any
import os
import json
import random
import re

from bs4 import BeautifulSoup
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
from accelerate import infer_auto_device_map, init_empty_weights

from chromadb import Client, Settings
from chromadb.utils import embedding_functions

from chunking_evaluation.utils import rigorous_document_search, get_bge_m3_embedding_function
from .base_evaluation import BaseEvaluation

import pandas as pd
import numpy as np
from importlib import resources

class SyntheticEvaluation(BaseEvaluation):
    def __init__(self, corpora_paths: List[str], queries_csv_path: str, chroma_db_path: str = None):
        super().__init__(questions_csv_path=queries_csv_path, chroma_db_path=chroma_db_path)
        self.corpora_paths = corpora_paths
        self.questions_csv_path = queries_csv_path

        llama_model_id = "meta-llama/Llama-3.1-8B-Instruct"

        # 定義量化配置
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            llm_int8_enable_fp32_cpu_offload=True
        )

        try:
            # 載入分詞器
            self.llama_tokenizer = AutoTokenizer.from_pretrained(llama_model_id)
            if self.llama_tokenizer.pad_token is None:
                self.llama_tokenizer.pad_token = self.llama_tokenizer.eos_token

            # 使用 accelerate 預先計算裝置分佈，並給出進度條
            print("正在預估模型分佈...")
            with init_empty_weights():
                dummy_model = AutoModelForCausalLM.from_pretrained(
                    llama_model_id,
                    torch_dtype=torch.bfloat16,
                    trust_remote_code=True
                )
            
            # 根據您的系統調整這些值。這是一個示例配置。
            # 如果您有單張 24GB VRAM 的 GPU，您可以將其設定為 {0: "22GiB"}。
            max_memory = {0: "12GiB", "cpu": "8GiB"}
            print(f"根據 max_memory={max_memory} 預估裝置分佈中...")
            device_map = infer_auto_device_map(
                dummy_model,
                max_memory=max_memory,
                no_split_module_classes=["LlamaDecoderLayer"]
            )
            print(f"預估的裝置分佈: {device_map}")
            
            # 載入模型，並使用上面計算的裝置分佈
            print(f"正在載入 {llama_model_id} 模型...")
            self.llama_model = AutoModelForCausalLM.from_pretrained(
                llama_model_id,
                device_map=device_map,
                quantization_config=quantization_config,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            
            print(f"{llama_model_id} 模型已成功載入到裝置: {self.llama_model.hf_device_map}")
        
        except Exception as e:
            print("=" * 50)
            print("❌ 載入模型時發生致命錯誤 ❌")
            print("這通常是因為記憶體不足。請嘗試以下方法：")
            print("1. 載入更小的模型，例如 `meta-llama/Llama-3.1-8B-Instruct`。")
            print("2. 確保您有足夠的 GPU VRAM 和系統 RAM。")
            print(f"原始錯誤訊息: {e}")
            print("=" * 50)
            # 在這裡選擇是否要讓程式停止，或是載入一個備用模型
            raise e
        
        self.synth_questions_df = None

        with resources.as_file(resources.files('chunking_evaluation.evaluation_framework') / 'prompts') as prompt_path:
            with open(os.path.join(prompt_path, 'question_maker_system.txt'), 'r') as f:
                self.question_maker_system_prompt = f.read()

            with open(os.path.join(prompt_path, 'question_maker_approx_system.txt'), 'r') as f:
                self.question_maker_approx_system_prompt = f.read()
            
            with open(os.path.join(prompt_path, 'question_maker_user.txt'), 'r') as f:
                self.question_maker_user_prompt = f.read()

            with open(os.path.join(prompt_path, 'question_maker_approx_user.txt'), 'r') as f:
                self.question_maker_approx_user_prompt = f.read()

    def _safe_json_loads(self, json_str):
        """
        安全的 JSON 解析函數，處理可能的雙重序列化問題
        """
        if pd.isna(json_str) or json_str == '':
            return []
        
        try:
            # 第一次解析
            result = json.loads(json_str)
            
            # 如果結果是字串，說明有雙重序列化，需要再解析一次
            if isinstance(result, str):
                print("檢測到雙重序列化，進行第二次解析...")
                result = json.loads(result)
            
            # 確保結果是 list
            if not isinstance(result, list):
                print(f"警告：references 不是 list 格式，而是 {type(result)}，嘗試轉換...")
                if isinstance(result, dict):
                    result = [result]
                else:
                    raise ValueError(f"無法處理的 references 格式：{type(result)}")
            
            return result
            
        except json.JSONDecodeError as e:
            print(f"JSON 解析錯誤：{e}")
            print(f"原始字串：{json_str[:200]}...")
            return []
        except Exception as e:
            print(f"其他解析錯誤：{e}")
            return []

    def _save_questions_df(self):
        """
        修正的儲存函數，避免雙重序列化
        """
        # 創建一個副本用於儲存
        df_to_save = self.synth_questions_df.copy()
        
        # 確保 references 欄位正確序列化
        def safe_serialize_references(refs):
            if isinstance(refs, str):
                # 如果已經是字串，先嘗試解析再重新序列化
                try:
                    parsed_refs = self._safe_json_loads(refs)
                    return json.dumps(parsed_refs, ensure_ascii=False)
                except:
                    return refs
            elif isinstance(refs, (list, dict)):
                # 如果是物件，直接序列化
                return json.dumps(refs, ensure_ascii=False)
            else:
                # 其他情況，轉為空 list
                return json.dumps([], ensure_ascii=False)
        
        df_to_save['references'] = df_to_save['references'].apply(safe_serialize_references)
        df_to_save.to_csv(self.questions_csv_path, index=False)
        
    def _clean_html(self, html_content: str) -> str:
        """使用 BeautifulSoup 清理 HTML 內容，提取純文本。"""
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            for script_or_style in soup(['script', 'style']):
                script_or_style.decompose()
            clean_text = soup.get_text()
            clean_text = ' '.join(clean_text.split())
            return clean_text
        except Exception as e:
            print(f"HTML 清理失敗: {e}")
            return html_content

    def _clean_text(self, text_content: str) -> str:
        """簡單清理純文本內容，移除多餘空白。"""
        return re.sub(r'\s+', ' ', text_content).strip()

    def _get_cleaned_document_content(self, file_path: str) -> str:
        """根據檔案副檔名，讀取並清理文件內容。"""
        file_extension = os.path.splitext(file_path)[1].lower()
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                raw_content = f.read()
        except UnicodeDecodeError:
            print(f"嘗試以 latin-1 讀取檔案: {file_path}")
            with open(file_path, 'r', encoding='latin-1') as f:
                raw_content = f.read()
        except Exception as e:
            print(f"讀取檔案失敗 {file_path}: {e}")
            return ""

        if file_extension in ['.html', '.htm']:
            print(f" - 正在清理 HTML 檔案: {file_path}")
            return self._clean_html(raw_content)
        elif file_extension in ['.txt', '.md', '.markdown']:
            print(f" - 正在清理 TEXT/MARKDOWN 檔案: {file_path}")
            return self._clean_text(raw_content)
        else:
            print(f" - 未知檔案類型 ({file_extension})，進行通用清理: {file_path}")
            return self._clean_text(raw_content)

    def _tag_text(self, text):
        chunk_length = 100
        chunks = []
        tag_indexes = [0]
        start = 0
        while start < len(text):
            end = start + chunk_length
            chunk = text[start:end]
            if end < len(text):
                space_index = chunk.rfind(' ')
                if space_index != -1:
                    end = start + space_index + 1
                    chunk = text[start:end]
            chunks.append(chunk)
            tag_indexes.append(end)
            start = end
        tagged_text = ""
        for i, chunk in enumerate(chunks):
            tagged_text += f"<start_chunk_{i}>" + chunk + f"<end_chunk_{i}>"
        return tagged_text, tag_indexes
    
    def _run_llama_inference(self, system_prompt: str, user_prompt: str, max_tokens: int) -> str:
        """
        改進的 Llama 推理函式，使用更可靠的回應提取邏輯
        """
        
        # Llama 3.1 的 chat 格式
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        # 應用 chat template
        prompt = self.llama_tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        # 加強的 JSON 格式提示
        json_instruction = "\n\nIMPORTANT: You must respond with ONLY a valid JSON object. Do not include any explanations, comments, or additional text outside the JSON structure."
        full_prompt = prompt + json_instruction
        
        # 記錄原始 prompt 的長度，用於後續精確切割
        self._original_prompt = full_prompt
        
        inputs = self.llama_tokenizer(
            full_prompt, 
            return_tensors="pt", 
            padding=True,
            truncation=True,
            max_length=4096
        ).to(self.llama_model.device)
        
        with torch.no_grad():
            outputs = self.llama_model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=0.1,
                top_p=0.9,
                top_k=50,
                repetition_penalty=1.05,
                pad_token_id=self.llama_tokenizer.eos_token_id,
                eos_token_id=self.llama_tokenizer.eos_token_id,
                use_cache=True
            )
        
        # 解碼完整輸出
        full_response = self.llama_tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 使用改進的回應提取邏輯
        model_response = self._extract_model_response_from_full_output(full_response, full_prompt)
        
        return model_response

    def _extract_model_response_from_full_output(self, full_response: str, original_prompt: str) -> str:
        """
        改進的模型回應提取邏輯，更可靠地分離模型的實際回應
        """
        # 策略1: 精確前綴匹配
        if full_response.startswith(original_prompt):
            model_response = full_response[len(original_prompt):].strip()
            if model_response:
                print("✅ 成功使用精確前綴匹配提取回應")
                return model_response

        # 策略2: 尋找 assistant 標記（適用於 Llama chat format）
        assistant_markers = [
            "<|start_header_id|>assistant<|end_header_id|>\n\n",
            "<|start_header_id|>assistant<|end_header_id|>\n",
            "<|start_header_id|>assistant<|end_header_id|>",
            "assistant<|end_header_id|>\n\n",
            "assistant<|end_header_id|>\n",
            "assistant<|end_header_id|>",
        ]
        
        for marker in assistant_markers:
            if marker in full_response:
                # 找到最後一個 assistant 標記（避免提取範例）
                last_pos = full_response.rfind(marker)
                if last_pos != -1:
                    response_start = last_pos + len(marker)
                    model_response = full_response[response_start:].strip()
                    if model_response and not self._contains_example_content(model_response):
                        print(f"✅ 成功使用 assistant 標記提取回應: {marker}")
                        return model_response

        # 策略3: 尋找 JSON 指令後的第一個真實 JSON（避開範例）
        json_instruction_variants = [
            "IMPORTANT: You must respond with ONLY a valid JSON object",
            "You must respond with ONLY a valid JSON object",
            "respond with ONLY a valid JSON object"
        ]
        
        for instruction in json_instruction_variants:
            instruction_pos = full_response.find(instruction)
            if instruction_pos != -1:
                # 從指令後開始搜索
                search_start = instruction_pos + len(instruction)
                potential_jsons = self._find_all_json_objects(full_response[search_start:])
                
                # 過濾掉包含範例內容的 JSON
                for json_obj in potential_jsons:
                    if not self._contains_example_content(json_obj):
                        print("✅ 成功使用 JSON 指令後定位提取回應")
                        return json_obj

        # 策略4: 尋找最後一個完整的 JSON 對象（排除範例）
        all_jsons = self._find_all_json_objects(full_response)
        for json_obj in reversed(all_jsons):  # 從後往前找
            if not self._contains_example_content(json_obj):
                print("✅ 使用最後 JSON 對象提取回應")
                return json_obj

        # 策略5: 如果以上都失敗，嘗試找到不包含範例的任何 JSON
        for json_obj in all_jsons:
            if not self._contains_example_content(json_obj):
                print("⚠️ 使用任意非範例 JSON 對象提取回應")
                return json_obj

        # 最後手段：返回原始回應
        print("❌ 所有提取策略失敗，返回原始回應")
        return full_response

    def _contains_example_content(self, text: str) -> bool:
        """
        檢測文本是否包含範例內容，針對課程大綱優化
        """
        # 更新為課程大綱相關的範例關鍵詞
        example_indicators = [
            "資料結構課程",  # 來自新範例
            "張明華教授",    # 來自新範例
            "CS201",         # 來自新範例
            "Experiment A",  # 保留舊範例檢測
            "Experiment B", 
            "Experiment C",
            "Experiment D",
            "這篇論文進行了哪些實驗",  # 舊範例
            "temperature control test",
            "pH sensitivity test",
            "enzyme activity assay",
            "light exposure trial",
            "reaction rate",
            "UV light",
            "degradation of reactants"
        ]
        
        text_lower = text.lower()
        for indicator in example_indicators:
            if indicator.lower() in text_lower:
                return True
        return False

    def _find_all_json_objects(self, text: str) -> List[str]:
        """
        找到文本中所有完整的 JSON 對象
        """
        json_objects = []
        i = 0
        
        while i < len(text):
            # 尋找 JSON 開始
            start_pos = text.find('{', i)
            if start_pos == -1:
                break
            
            # 尋找匹配的結束括號
            brace_count = 0
            end_pos = -1
            
            for j in range(start_pos, len(text)):
                if text[j] == '{':
                    brace_count += 1
                elif text[j] == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        end_pos = j + 1
                        break
            
            if end_pos != -1:
                json_candidate = text[start_pos:end_pos]
                json_objects.append(json_candidate)
                i = end_pos
            else:
                i = start_pos + 1
        
        return json_objects

    def _extract_json_from_response(self, response_text: str) -> dict:
        """
        改進的 JSON 解析器，支持多種 JSON 提取策略
        """
        import re
        import json
        
        print(f"📥 開始解析回應，長度: {len(response_text)} 字符")
        
        # 首先檢查是否包含範例內容
        if self._contains_example_content(response_text):
            print("⚠️ 警告：檢測到回應包含範例內容，這可能表示回應提取有問題")
        
        # 1. 嘗試找到所有 JSON 對象
        json_candidates = self._find_all_json_objects(response_text)
        print(f"📋 找到 {len(json_candidates)} 個 JSON 候選對象")
        
        # 2. 按優先級嘗試解析每個候選對象
        for i, json_str in enumerate(json_candidates):
            print(f"🧪 嘗試解析候選對象 {i+1}: 長度 {len(json_str)} 字符")
            
            # 跳過明顯包含範例的 JSON
            if self._contains_example_content(json_str):
                print(f"⏭️ 跳過候選對象 {i+1}: 包含範例內容")
                continue
            
            try:
                result = self._parse_and_validate_json(json_str)
                if result:
                    print(f"✅ 成功解析候選對象 {i+1}")
                    return result
            except Exception as e:
                print(f"❌ 候選對象 {i+1} 解析失敗: {e}")
                continue
        
        # 3. 如果直接解析失敗，嘗試 JSON 修復
        print("🔧 嘗試 JSON 修復策略...")
        for i, json_str in enumerate(json_candidates):
            if self._contains_example_content(json_str):
                continue
                
            try:
                fixed_json = self._fix_common_json_issues(json_str)
                result = self._parse_and_validate_json(fixed_json)
                if result:
                    print(f"✅ 修復後成功解析候選對象 {i+1}")
                    return result
            except Exception as e:
                continue
        
        # 4. 最後手段：嘗試從回應中提取可能的 JSON 片段
        print("🚨 使用最後手段：提取 JSON 片段")
        try:
            return self._extract_json_fragments(response_text)
        except Exception as e:
            raise ValueError(f"所有 JSON 解析策略均失敗。原始錯誤: {e}")

    def _parse_and_validate_json(self, json_str: str) -> dict:
        """
        解析和驗證 JSON 對象
        """
        import json
        
        result = json.loads(json_str)
        
        # 驗證必需字段
        required_fields = ['question', 'references']
        for field in required_fields:
            if field not in result:
                raise ValueError(f"缺少必需字段: {field}")
        
        # 驗證 question 不為空且不是範例
        question = result['question']
        if not question or len(question.strip()) < 3:
            raise ValueError("問題為空或過短")
        
        if self._contains_example_content(question):
            raise ValueError("問題包含範例內容")
        
        # 驗證 references 格式
        references = result['references']
        if not isinstance(references, list):
            raise ValueError("references 必須是列表")
        
        if len(references) == 0:
            raise ValueError("references 不能為空")
        
        if len(references) > 5:
            raise ValueError("references 數量不能超過 5")
        
        # 驗證每個 reference
        for i, ref in enumerate(references):
            if not isinstance(ref, dict):
                raise ValueError(f"references[{i}] 必須是字典")
            
            required_ref_fields = ['content', 'start_chunk', 'end_chunk']
            for ref_field in required_ref_fields:
                if ref_field not in ref:
                    raise ValueError(f"references[{i}] 缺少字段: {ref_field}")
            
            # 檢查 content 不為空且不是範例
            if not ref['content'] or len(ref['content'].strip()) < 3:
                raise ValueError(f"references[{i}] content 為空或過短")
            
            if self._contains_example_content(ref['content']):
                raise ValueError(f"references[{i}] content 包含範例內容")
        
        print(f"✅ JSON 驗證通過 - 問題: {question[:50]}...")
        print(f"   參考資料數量: {len(references)}")
        
        return result

    def _fix_common_json_issues(self, json_str: str) -> str:
        """
        修復常見的 JSON 格式問題
        """
        import re
        
        # 移除尾隨逗號
        fixed = re.sub(r',\s*}', '}', json_str)
        fixed = re.sub(r',\s*]', ']', fixed)
        
        # 修復中文標點
        fixed = fixed.replace('，', ',').replace('：', ':')
        
        # 移除可能的多餘字符
        fixed = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', fixed)
        
        return fixed

    def _extract_json_fragments(self, text: str) -> dict:
        """
        最後手段：從文本中提取 JSON 片段並嘗試重構
        """
        import re
        
        # 嘗試提取問題
        question_patterns = [
            r'"question":\s*"([^"]+)"',
            r"'question':\s*'([^']+)'",
            r'question[:\s]+"([^"]+)"'
        ]
        
        question = None
        for pattern in question_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match and not self._contains_example_content(match.group(1)):
                question = match.group(1)
                break
        
        if not question:
            raise ValueError("無法提取問題")
        
        # 簡單構造一個基本的回應
        return {
            "question": question,
            "references": [{"content": "無法解析參考資料", "start_chunk": 0, "end_chunk": 0}]
        }

    def debug_prompt_cutting(self, corpus_id, use_approx=True):
        """
        調試 prompt 切割效果的函式
        """
        print("="*80)
        print("🔍 Prompt 切割調試")
        print("="*80)
        
        # 準備數據
        corpus = self._get_cleaned_document_content(corpus_id)
        if len(corpus) > 4000:
            document = corpus[:4000]
        else:
            document = corpus
        
        if use_approx:
            tagged_text, _ = self._tag_text(document)
            user_prompt = self.question_maker_approx_user_prompt.replace("{document}", tagged_text).replace("{prev_questions_str}", "")
            system_prompt = self.question_maker_approx_system_prompt
        else:
            user_prompt = self.question_maker_user_prompt.replace("{document}", document).replace("{prev_questions_str}", "")
            system_prompt = self.question_maker_system_prompt
        
        print("執行推理並切割...")
        response = self._run_llama_inference(system_prompt, user_prompt, 600)
        
        print("\n" + "="*80)
        print("🎯 切割結果檢查")
        print("="*80)
        
        try:
            result = self._extract_json_from_response(response)
            print("✅ 最終解析成功!")
            print(f"問題: {result.get('question', 'NO QUESTION')}")
            print(f"參考資料數量: {len(result.get('references', []))}")
            
            for i, ref in enumerate(result.get('references', [])):
                print(f"  參考 {i+1}: {ref.get('content', '')[:50]}...")
            
            return result
            
        except Exception as e:
            print(f"❌ 最終解析失敗: {e}")
            return None

    def _extract_question_and_approx_references(self, corpus, document_length=4000, prev_questions=[]):
        if len(corpus) > document_length:
            start_index = random.randint(0, len(corpus) - document_length)
            document = corpus[start_index : start_index + document_length]
        else:
            start_index = 0
            document = corpus
        
        if prev_questions is not None:
            if len(prev_questions) > 20:
                questions_sample = random.sample(prev_questions, 20)
                prev_questions_str = '\n'.join(questions_sample)
            else:
                prev_questions_str = '\n'.join(prev_questions)
        else:
            prev_questions_str = ""

        tagged_text, tag_indexes = self._tag_text(document)

        user_prompt_filled = self.question_maker_approx_user_prompt.replace("{document}", tagged_text).replace("{prev_questions_str}", prev_questions_str)
        response_text = self._run_llama_inference(self.question_maker_approx_system_prompt, user_prompt_filled, 600)
        
        json_response = self._extract_json_from_response(response_text)
        
        try:
            text_references = json_response['references']
        except KeyError:
            raise ValueError("The response does not contain a 'references' field.")
        try:
            question = json_response['question']
        except KeyError:
            raise ValueError("The response does not contain a 'question' field.")

        references = []
        max_chunk_index = len(tag_indexes) - 2  # 可用的最大chunk索引 (因為我們要訪問 index+1)
        
        print(f"📊 Chunk索引調試資訊: tag_indexes長度={len(tag_indexes)}, 最大可用chunk索引={max_chunk_index}")
        
        for i, reference in enumerate(text_references):
            reference_keys = list(reference.keys())

            if len(reference_keys) != 3:
                raise ValueError(f"Each reference must have exactly 3 keys: 'content', 'start_chunk', and 'end_chunk'. Got keys: {reference_keys}")

            if 'start_chunk' not in reference_keys or 'end_chunk' not in reference_keys:
                raise ValueError("Each reference must contain 'start_chunk' and 'end_chunk' keys.")

            # 添加索引邊界檢查
            start_chunk = reference['start_chunk']
            end_chunk = reference['end_chunk']
            
            print(f"  參考 {i+1}: start_chunk={start_chunk}, end_chunk={end_chunk}")
            
            # 驗證 start_chunk 索引
            if start_chunk < 0 or start_chunk >= len(tag_indexes):
                print(f"⚠️  警告: start_chunk {start_chunk} 超出範圍 [0, {len(tag_indexes)-1}]，調整為有效值")
                start_chunk = max(0, min(start_chunk, len(tag_indexes)-1))
            
            # 驗證 end_chunk 索引 (需要考慮 +1 的訪問)
            if end_chunk < 0 or end_chunk >= max_chunk_index:
                print(f"⚠️  警告: end_chunk {end_chunk} 超出安全範圍 [0, {max_chunk_index}]，調整為有效值")
                end_chunk = max(0, min(end_chunk, max_chunk_index))
            
            # 確保 start_chunk <= end_chunk
            if start_chunk > end_chunk:
                print(f"⚠️  警告: start_chunk ({start_chunk}) > end_chunk ({end_chunk})，交換數值")
                start_chunk, end_chunk = end_chunk, start_chunk

            # 使用修正後的索引進行計算
            try:
                if 'end_chunk' not in reference_keys:
                    reference_keys.remove('content')
                    reference_keys.remove('start_chunk')
                    end_chunk_key = reference_keys[0]
                    end_index = start_index + tag_indexes[reference[end_chunk_key]+1]
                else:
                    # 安全地訪問 tag_indexes
                    if end_chunk + 1 < len(tag_indexes):
                        end_index = start_index + tag_indexes[end_chunk + 1]
                    else:
                        # 如果 end_chunk+1 超出範圍，使用最後一個可用的索引
                        print(f"⚠️  end_chunk+1 ({end_chunk+1}) 超出 tag_indexes 範圍，使用最後可用位置")
                        end_index = start_index + tag_indexes[-1]

                start_index_ref = start_index + tag_indexes[start_chunk]
                
                # 額外的邊界檢查：確保不超出原始corpus範圍
                end_index = min(end_index, len(corpus))
                start_index_ref = min(start_index_ref, len(corpus))
                
                # 確保 start_index_ref <= end_index
                if start_index_ref > end_index:
                    print(f"⚠️  警告: start_index_ref ({start_index_ref}) > end_index ({end_index})，調整為相同值")
                    end_index = start_index_ref
                
                print(f"    最終索引: start_index_ref={start_index_ref}, end_index={end_index}")
                
                extracted_text = corpus[start_index_ref:end_index]
                if len(extracted_text.strip()) == 0:
                    print(f"⚠️  警告: 提取的文本為空，跳過此參考")
                    continue
                    
                references.append((extracted_text, start_index_ref, end_index))
                
            except IndexError as e:
                print(f"❌ 索引錯誤 (參考 {i+1}): {e}")
                print(f"   start_chunk={start_chunk}, end_chunk={end_chunk}")
                print(f"   tag_indexes長度={len(tag_indexes)}")
                print(f"   嘗試訪問的索引: {end_chunk+1}")
                # 跳過這個有問題的參考，而不是讓整個程序崩潰
                continue
            except Exception as e:
                print(f"❌ 其他錯誤 (參考 {i+1}): {e}")
                continue
        
        # 檢查是否至少有一個有效的參考
        if len(references) == 0:
            raise ValueError("No valid references could be extracted from the model response.")
        
        print(f"✅ 成功提取 {len(references)} 個參考資料")
        return question, references

    def _extract_question_and_references(self, corpus, document_length=4000, prev_questions=[]):
        if len(corpus) > document_length:
            start_index = random.randint(0, len(corpus) - document_length)
            document = corpus[start_index : start_index + document_length]
        else:
            document = corpus
        
        if prev_questions is not None:
            if len(prev_questions) > 20:
                questions_sample = random.sample(prev_questions, 20)
                prev_questions_str = '\n'.join(questions_sample)
            else:
                prev_questions_str = '\n'.join(prev_questions)
        else:
            prev_questions_str = ""

        user_prompt_filled = self.question_maker_user_prompt.replace("{document}", document).replace("{prev_questions_str}", prev_questions_str)
        response_text = self._run_llama_inference(self.question_maker_system_prompt, user_prompt_filled, 600)

        json_response = self._extract_json_from_response(response_text)
        
        try:
            text_references = json_response['references']
        except KeyError:
            raise ValueError("The response does not contain a 'references' field.")
        try:
            question = json_response['question']
        except KeyError:
            raise ValueError("The response does not contain a 'question' field.")

        references = []
        for reference in text_references:
            if not isinstance(reference, str):
                raise ValueError(f"Expected reference to be of type str, but got {type(reference).__name__}")
            target = rigorous_document_search(corpus, reference)
            if target is not None:
                reference, start_index, end_index = target
                references.append((reference, start_index, end_index))
            else:
                raise ValueError(f"No match found in the document for the given reference.\nReference: {reference}")
        
        return question, references

    def _generate_corpus_questions(self, corpus_id, approx=False, n=5):
        corpus = self._get_cleaned_document_content(corpus_id)
        
        if not corpus:
            print(f"文件 {corpus_id} 內容為空或無法處理，跳過。")
            return

        i = 0
        while i < n:
            retry_count = 0
            max_retries = 3
            success = False
            
            while retry_count < max_retries and not success:
                try:
                    print(f"Trying Query {i} for {corpus_id} (attempt {retry_count + 1})")
                    questions_list = self.synth_questions_df[self.synth_questions_df['corpus_id'] == corpus_id]['question'].tolist()
                    
                    if approx:
                        question, references = self._extract_question_and_approx_references(corpus, 4000, questions_list)
                    else:
                        question, references = self._extract_question_and_references(corpus, 4000, questions_list)
                    
                    if len(references) > 5:
                        raise ValueError("The number of references exceeds 5.")
                    
                    # 驗證問題有效性
                    if not question or len(question.strip()) < 3:
                        raise ValueError("Generated question is too short or empty")
                    
                    # 額外驗證：確保問題不包含範例內容
                    if self._contains_example_content(question):
                        raise ValueError("Generated question contains example content")
                    
                    print(f"SUCCESS: Generated question: {question}")
                    
                    # 修正：直接使用 list of dict，不進行額外的 JSON 序列化
                    references_list = [{'content': ref[0], 'start_index': ref[1], 'end_index': ref[2]} for ref in references]
                    new_question = {
                        'question': question,
                        'references': references_list,  # 直接儲存為 list，稍後統一序列化
                        'corpus_id': corpus_id
                    }

                    new_df = pd.DataFrame([new_question])
                    self.synth_questions_df = pd.concat([self.synth_questions_df, new_df], ignore_index=True)
                    self._save_questions_df()
                    
                    success = True
                    print(f"Question {i} saved successfully")

                except (ValueError, json.JSONDecodeError) as e:
                    print(f"Error occurred (attempt {retry_count + 1}): {e}")
                    retry_count += 1
                    
                    if retry_count >= max_retries:
                        print(f"Max retries reached for query {i}, skipping...")
                        break
            
            if success:
                i += 1
            else:
                print(f"Failed to generate question {i}, moving to next")
                i += 1

    def _get_synth_questions_df(self):
        """
        修正的載入函數，處理可能的雙重序列化問題
        """
        if os.path.exists(self.questions_csv_path):
            synth_questions_df = pd.read_csv(self.questions_csv_path)
            
            # 修正 references 欄位的解析問題
            if 'references' in synth_questions_df.columns:
                def fix_references(refs_str):
                    try:
                        # 使用安全的 JSON 解析
                        parsed_refs = self._safe_json_loads(refs_str)
                        return parsed_refs
                    except:
                        print(f"無法解析 references: {refs_str[:100]}...")
                        return []
                
                synth_questions_df['references'] = synth_questions_df['references'].apply(fix_references)
        else:
            synth_questions_df = pd.DataFrame(columns=['question', 'references', 'corpus_id'])
        return synth_questions_df

    def generate_queries_and_excerpts(self, approximate_excerpts=False, num_rounds = -1, queries_per_corpus = 5):
        self.synth_questions_df = self._get_synth_questions_df()

        rounds = 0
        while num_rounds == -1 or rounds < num_rounds:
            for corpus_id in self.corpora_paths:
                self._generate_corpus_questions(corpus_id, approx=approximate_excerpts, n=queries_per_corpus)
            rounds += 1

    def _get_sim(self, target, references):
        texts = [target] + references
        embedding_function = get_bge_m3_embedding_function()
        
        # 修改這裡：使用正確的方法調用
        # 選項1：如果 HuggingFaceBgeEmbeddings 有 embed_documents 方法
        try:
            embeddings = embedding_function.embed_documents(texts)
        except AttributeError:
            # 選項2：如果只有 embed_query 方法，逐個處理
            embeddings = [embedding_function.embed_query(text) for text in texts]
        
        nparray1 = embeddings[0]

        full_sim = []
        for i in range(1, len(embeddings)):
            nparray2 = embeddings[i]
            cosine_similarity = np.dot(nparray1, nparray2) / (np.linalg.norm(nparray1) * np.linalg.norm(nparray2))
            full_sim.append(cosine_similarity)

        return full_sim

    def _corpus_filter_poor_highlights(self, corpus_id, synth_questions_df, threshold):
        corpus_questions_df = synth_questions_df[synth_questions_df['corpus_id'] == corpus_id].copy()

        def edit_row(row):
            question = row['question']
            
            # 修正：使用安全的 JSON 解析
            try:
                references_data = row['references']
                if isinstance(references_data, str):
                    # 如果是字串，解析它
                    references_list = self._safe_json_loads(references_data)
                elif isinstance(references_data, list):
                    # 如果已經是 list，直接使用
                    references_list = references_data
                else:
                    print(f"未知的 references 格式: {type(references_data)}")
                    references_list = []
                
                references = [ref['content'] for ref in references_list if isinstance(ref, dict) and 'content' in ref]
                
                if references:
                    similarity_scores = self._get_sim(question, references)
                    if similarity_scores:
                        worst_ref_score = min(similarity_scores)
                        row['worst_ref_score'] = worst_ref_score
                    else:
                        row['worst_ref_score'] = -1.0
                else:
                    row['worst_ref_score'] = -1.0
                    
            except Exception as e:
                print(f"處理 row 時發生錯誤: {e}")
                row['worst_ref_score'] = -1.0
            
            return row

        corpus_questions_df = corpus_questions_df.apply(edit_row, axis=1)

        count_before = len(corpus_questions_df)

        corpus_questions_df = corpus_questions_df[corpus_questions_df['worst_ref_score'] >= threshold]
        corpus_questions_df = corpus_questions_df.drop(columns=['worst_ref_score'])

        count_after = len(corpus_questions_df)

        print(f"Corpus: {corpus_id} - Removed {count_before - count_after} .")

        # 確保 references 正確序列化
        def ensure_json_string(refs):
            if isinstance(refs, list):
                return json.dumps(refs, ensure_ascii=False)
            elif isinstance(refs, str):
                # 驗證是否為有效 JSON
                try:
                    json.loads(refs)
                    return refs
                except:
                    return json.dumps([], ensure_ascii=False)
            else:
                return json.dumps([], ensure_ascii=False)
        
        corpus_questions_df['references'] = corpus_questions_df['references'].apply(ensure_json_string)

        full_questions_df = pd.read_csv(self.questions_csv_path)
        full_questions_df = full_questions_df[full_questions_df['corpus_id'] != corpus_id]

        full_questions_df = pd.concat([full_questions_df, corpus_questions_df], ignore_index=True)
        for col in ['fixed', 'worst_ref_score', 'diff_score']:
            if col in full_questions_df.columns:
                full_questions_df = full_questions_df.drop(columns=col)

        full_questions_df.to_csv(self.questions_csv_path, index=False)

    def filter_poor_excerpts(self, threshold=0.36, corpora_subset=[]):
        if os.path.exists(self.questions_csv_path):
            synth_questions_df = pd.read_csv(self.questions_csv_path)
            if len(synth_questions_df) > 0:
                corpus_list = synth_questions_df['corpus_id'].unique().tolist()
                if corpora_subset:
                    corpus_list = [c for c in corpus_list if c in corpora_subset]
                for corpus_id in corpus_list:
                    self._corpus_filter_poor_highlights(corpus_id, synth_questions_df, threshold)

    def _corpus_filter_duplicates(self, corpus_id, synth_questions_df, threshold):
        corpus_questions_df = synth_questions_df[synth_questions_df['corpus_id'] == corpus_id].copy()
        count_before = len(corpus_questions_df)
        
        # 先去除完全相同的問題
        corpus_questions_df.drop_duplicates(subset='question', keep='first', inplace=True)
        
        questions = corpus_questions_df['question'].tolist()
        
        # 如果只有一個或零個問題，直接返回
        if len(questions) <= 1:
            print(f"Corpus: {corpus_id} - Only {len(questions)} question(s), no duplicates to filter.")
            # 保存處理後的結果 - 修正序列化問題
            def ensure_json_string(refs):
                if isinstance(refs, list):
                    return json.dumps(refs, ensure_ascii=False)
                elif isinstance(refs, str):
                    try:
                        json.loads(refs)
                        return refs
                    except:
                        return json.dumps([], ensure_ascii=False)
                else:
                    return json.dumps([], ensure_ascii=False)
            
            corpus_questions_df['references'] = corpus_questions_df['references'].apply(ensure_json_string)
            full_questions_df = pd.read_csv(self.questions_csv_path)
            full_questions_df = full_questions_df[full_questions_df['corpus_id'] != corpus_id]
            full_questions_df = pd.concat([full_questions_df, corpus_questions_df], ignore_index=True)
            for col in ['fixed', 'worst_ref_score', 'diff_score']:
                if col in full_questions_df.columns:
                    full_questions_df = full_questions_df.drop(columns=col)
            full_questions_df.to_csv(self.questions_csv_path, index=False)
            return
        
        # 使用正確的 embedding 函式
        try:
            # 方法1: 嘗試使用 BGE-M3 embedding function
            embedding_function = get_bge_m3_embedding_function()
            embeddings_list = []
            print(f"正在計算 {len(questions)} 個問題的 embeddings...")
            for i, question in enumerate(questions):
                if i % 10 == 0:  # 每10個打印一次進度
                    print(f"  處理進度: {i+1}/{len(questions)}")
                try:
                    # 嘗試 embed_documents 方法
                    embedding = embedding_function.embed_documents([question])[0]
                except AttributeError:
                    # 如果沒有 embed_documents，使用 embed_query
                    embedding = embedding_function.embed_query(question)
                embeddings_list.append(embedding)
            
            embeddings_matrix = np.array(embeddings_list)
            
        except Exception as e:
            print(f"BGE-M3 embedding 失敗: {e}")
            print("嘗試使用備用方案...")
            
            # 方法2: 備用方案 - 使用簡單的文本相似度
            try:
                from sklearn.feature_extraction.text import TfidfVectorizer
                from sklearn.metrics.pairwise import cosine_similarity
                
                print("使用 TF-IDF + Cosine Similarity 作為備用方案")
                vectorizer = TfidfVectorizer(max_features=1000, stop_words=None)
                tfidf_matrix = vectorizer.fit_transform(questions)
                dot_product_matrix = cosine_similarity(tfidf_matrix)
                
            except ImportError:
                print("sklearn 未安裝，使用基於字符的簡單相似度")
                # 方法3: 最後手段 - 基於字符的相似度
                def simple_similarity(s1, s2):
                    """計算兩個字符串的簡單相似度"""
                    s1_set = set(s1.lower().replace(' ', ''))
                    s2_set = set(s2.lower().replace(' ', ''))
                    if len(s1_set) == 0 and len(s2_set) == 0:
                        return 1.0
                    if len(s1_set) == 0 or len(s2_set) == 0:
                        return 0.0
                    intersection = len(s1_set.intersection(s2_set))
                    union = len(s1_set.union(s2_set))
                    return intersection / union
                
                # 構建相似度矩陣
                n = len(questions)
                dot_product_matrix = np.zeros((n, n))
                print("計算字符相似度矩陣...")
                for i in range(n):
                    for j in range(i, n):
                        if i == j:
                            dot_product_matrix[i][j] = 1.0
                        else:
                            sim = simple_similarity(questions[i], questions[j])
                            dot_product_matrix[i][j] = sim
                            dot_product_matrix[j][i] = sim
                
                # 跳過後續的 embedding 處理
                similarity_pairs = [(i, j, dot_product_matrix[i][j]) for i in range(len(dot_product_matrix)) for j in range(i+1, len(dot_product_matrix))]
                similarity_pairs.sort(key=lambda x: x[2], reverse=True)
                similarity_scores = np.array([x[2] for x in similarity_pairs])
                most_similars = (dot_product_matrix - np.eye(dot_product_matrix.shape[0])).max(axis=1)
                
                # 執行過濾
                def filter_vectors(sim_matrix, threshold):
                    n = sim_matrix.shape[0]
                    remaining = np.ones(n, dtype=bool)
                    for i in range(n):
                        if remaining[i] == 1:
                            for j in range(i+1, n):
                                if remaining[j] == 1 and sim_matrix[i, j] > threshold:
                                    remaining[j] = 0
                    return remaining
                
                rows_to_keep = filter_vectors(dot_product_matrix, threshold)
                corpus_questions_df = corpus_questions_df.iloc[rows_to_keep]
                count_after = len(corpus_questions_df)
                print(f"Corpus: {corpus_id} - Removed {count_before - count_after} questions due to similarity.")
                
                # 保存結果 - 修正序列化問題
                def ensure_json_string(refs):
                    if isinstance(refs, list):
                        return json.dumps(refs, ensure_ascii=False)
                    elif isinstance(refs, str):
                        try:
                            json.loads(refs)
                            return refs
                        except:
                            return json.dumps([], ensure_ascii=False)
                    else:
                        return json.dumps([], ensure_ascii=False)
                
                corpus_questions_df['references'] = corpus_questions_df['references'].apply(ensure_json_string)
                full_questions_df = pd.read_csv(self.questions_csv_path)
                full_questions_df = full_questions_df[full_questions_df['corpus_id'] != corpus_id]
                full_questions_df = pd.concat([full_questions_df, corpus_questions_df], ignore_index=True)
                for col in ['fixed', 'worst_ref_score', 'diff_score']:
                    if col in full_questions_df.columns:
                        full_questions_df = full_questions_df.drop(columns=col)
                full_questions_df.to_csv(self.questions_csv_path, index=False)
                return
        
        # 如果 BGE-M3 成功，繼續正常的 embedding 處理
        print("計算 cosine similarity 矩陣...")
        # 計算點積矩陣 (假設 embeddings 已經標準化)
        dot_product_matrix = np.dot(embeddings_matrix, embeddings_matrix.T)
        
        # 如果 embeddings 沒有標準化，使用 cosine similarity
        # from sklearn.metrics.pairwise import cosine_similarity
        # dot_product_matrix = cosine_similarity(embeddings_matrix)
        
        similarity_pairs = [(i, j, dot_product_matrix[i][j]) for i in range(len(dot_product_matrix)) for j in range(i+1, len(dot_product_matrix))]
        similarity_pairs.sort(key=lambda x: x[2], reverse=True)
        similarity_scores = np.array([x[2] for x in similarity_pairs])
        most_similars = (dot_product_matrix - np.eye(dot_product_matrix.shape[0])).max(axis=1)
        
        def filter_vectors(sim_matrix, threshold):
            n = sim_matrix.shape[0]
            remaining = np.ones(n, dtype=bool)
            for i in range(n):
                if remaining[i] == 1:
                    for j in range(i+1, n):
                        if remaining[j] == 1 and sim_matrix[i, j] > threshold:
                            remaining[j] = 0
            return remaining
        
        rows_to_keep = filter_vectors(dot_product_matrix, threshold)
        corpus_questions_df = corpus_questions_df.iloc[rows_to_keep]
        count_after = len(corpus_questions_df)
        
        print(f"Corpus: {corpus_id} - Removed {count_before - count_after} questions due to similarity.")
        
        # 保存結果 - 修正序列化問題
        def ensure_json_string(refs):
            if isinstance(refs, list):
                return json.dumps(refs, ensure_ascii=False)
            elif isinstance(refs, str):
                try:
                    json.loads(refs)
                    return refs
                except:
                    return json.dumps([], ensure_ascii=False)
            else:
                return json.dumps([], ensure_ascii=False)
        
        corpus_questions_df['references'] = corpus_questions_df['references'].apply(ensure_json_string)
        full_questions_df = pd.read_csv(self.questions_csv_path)
        full_questions_df = full_questions_df[full_questions_df['corpus_id'] != corpus_id]
        full_questions_df = pd.concat([full_questions_df, corpus_questions_df], ignore_index=True)
        for col in ['fixed', 'worst_ref_score', 'diff_score']:
            if col in full_questions_df.columns:
                full_questions_df = full_questions_df.drop(columns=col)
        full_questions_df.to_csv(self.questions_csv_path, index=False)

    def filter_duplicates(self, threshold=0.78, corpora_subset=[]):
        if os.path.exists(self.questions_csv_path):
            synth_questions_df = pd.read_csv(self.questions_csv_path)
            if len(synth_questions_df) > 0:
                corpus_list = synth_questions_df['corpus_id'].unique().tolist()
                if corpora_subset:
                    corpus_list = [c for c in corpus_list if c in corpora_subset]
                for corpus_id in corpus_list:
                    self._corpus_filter_duplicates(corpus_id, synth_questions_df, threshold)

    def question_ref_filter(self):
        self.synth_questions_df = self._get_synth_questions_df()

    def debug_full_output(self, corpus_id, use_approx=True, save_to_file=False):
        """
        增強版調試函式：檢查模型的完整輸入和輸出，包含改進的解析邏輯
        
        Args:
            corpus_id: 文檔路徑
            use_approx: 是否使用 approximate 模式
            save_to_file: 是否將結果保存到文件
        """
        print("="*100)
        print("🔍 完整輸入輸出調試 (增強版)")
        print("="*100)
        
        debug_results = {
            'corpus_id': corpus_id,
            'use_approx': use_approx,
            'steps': {}
        }
        
        # 1. 檢查文檔清理後的內容
        print("📄 步驟 1: 檢查文檔清理結果")
        print("-" * 50)
        corpus = self._get_cleaned_document_content(corpus_id)
        if not corpus:
            print("❌ ERROR: 無法讀取或清理文檔內容")
            return None
        
        debug_results['steps']['document_cleaning'] = {
            'success': True,
            'total_length': len(corpus),
            'preview': corpus[:500]
        }
        
        print(f"文檔總長度: {len(corpus)} 字符")
        print("前 500 字符:")
        print(repr(corpus[:500]))
        print("\n後 500 字符:")
        print(repr(corpus[-500:]))
        print()
        
        # 2. 檢查文檔片段選擇
        print("📋 步驟 2: 檢查選取的文檔片段")
        print("-" * 50)
        if len(corpus) > 4000:
            start_index = 0  # 固定選擇開頭，便於調試
            document = corpus[start_index:start_index + 4000]
            print(f"選取片段: 位置 {start_index} 到 {start_index + 4000}")
        else:
            document = corpus
            print("文檔長度小於 4000，使用完整內容")
        
        debug_results['steps']['document_selection'] = {
            'selected_length': len(document),
            'selection_start': 0,
            'selection_end': len(document)
        }
        
        print(f"選取片段長度: {len(document)} 字符")
        print("選取片段前 300 字符:")
        print(repr(document[:300]))
        print()
        
        # 3. 檢查 prompt 組裝
        print("🛠️ 步驟 3: 檢查 Prompt 組裝")
        print("-" * 50)
        
        if use_approx:
            print("使用 APPROXIMATE 模式")
            tagged_text, tag_indexes = self._tag_text(document)
            print(f"標記後文本長度: {len(tagged_text)} 字符")
            print("標記後文本前 500 字符:")
            print(repr(tagged_text[:500]))
            
            user_prompt = self.question_maker_approx_user_prompt.replace("{document}", tagged_text).replace("{prev_questions_str}", "")
            system_prompt = self.question_maker_approx_system_prompt
        else:
            print("使用 EXACT 模式")
            user_prompt = self.question_maker_user_prompt.replace("{document}", document).replace("{prev_questions_str}", "")
            system_prompt = self.question_maker_system_prompt
        
        debug_results['steps']['prompt_assembly'] = {
            'system_prompt_length': len(system_prompt),
            'user_prompt_length': len(user_prompt),
            'mode': 'APPROXIMATE' if use_approx else 'EXACT'
        }
        
        print(f"System prompt 長度: {len(system_prompt)} 字符")
        print("System prompt 前 300 字符:")
        print(repr(system_prompt[:300]))
        print()
        
        print(f"User prompt 長度: {len(user_prompt)} 字符")
        print("User prompt 前 500 字符:")
        print(repr(user_prompt[:500]))
        print("User prompt 後 500 字符:")
        print(repr(user_prompt[-500:]))
        print()
        
        # 4. 檢查完整的模型輸入
        print("📤 步驟 4: 檢查發送給模型的完整輸入")
        print("-" * 50)
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        full_prompt = self.llama_tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        json_instruction = "\n\nIMPORTANT: You must respond with ONLY a valid JSON object. Do not include any explanations, comments, or additional text outside the JSON structure."
        full_prompt += json_instruction
        
        debug_results['steps']['model_input'] = {
            'full_prompt_length': len(full_prompt),
            'has_json_instruction': 'IMPORTANT' in full_prompt
        }
        
        print(f"完整 prompt 長度: {len(full_prompt)} 字符")
        print("完整 prompt 前 800 字符:")
        print(repr(full_prompt[:800]))
        print("\n完整 prompt 後 800 字符:")
        print(repr(full_prompt[-800:]))
        print()
        
        # 5. 獲取模型原始輸出
        print("📥 步驟 5: 模型原始輸出")
        print("-" * 50)
        print("正在生成回應...")
        
        inputs = self.llama_tokenizer(
            full_prompt, 
            return_tensors="pt", 
            padding=True,
            truncation=True,
            max_length=4096
        ).to(self.llama_model.device)
        
        with torch.no_grad():
            outputs = self.llama_model.generate(
                **inputs,
                max_new_tokens=600,
                do_sample=True,
                temperature=0.1,
                top_p=0.9,
                top_k=50,
                repetition_penalty=1.05,
                pad_token_id=self.llama_tokenizer.eos_token_id,
                eos_token_id=self.llama_tokenizer.eos_token_id,
                use_cache=True
            )
        
        # 解碼完整輸出
        full_response = self.llama_tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        print("🤖 模型完整原始輸出:")
        print("="*60)
        print(full_response)
        print("="*60)
        print(f"原始輸出長度: {len(full_response)} 字符")
        print()
        
        debug_results['steps']['model_output'] = {
            'full_response_length': len(full_response),
            'full_response': full_response
        }
        
        # 6. 使用改進的回應提取邏輯
        print("🔧 步驟 6: 改進的回應提取")
        print("-" * 50)
        
        extracted_response = self._extract_model_response_from_full_output(full_response, full_prompt)
        
        print("提取的回應:")
        print("="*60)
        print(extracted_response)
        print("="*60)
        print(f"提取的回應長度: {len(extracted_response)} 字符")
        
        debug_results['steps']['response_extraction'] = {
            'extracted_length': len(extracted_response),
            'contains_example': self._contains_example_content(extracted_response)
        }
        
        if self._contains_example_content(extracted_response):
            print("⚠️ 警告：提取的回應包含範例內容!")
        print()
        
        # 7. 使用改進的 JSON 解析邏輯
        print("🧪 步驟 7: 改進的 JSON 解析")
        print("-" * 50)
        
        try:
            parsed_result = self._extract_json_from_response(extracted_response)
            print("✅ 解析成功!")
            print("解析結果:")
            print(f"- 問題: {parsed_result.get('question', 'NO QUESTION')}")
            print(f"- 參考資料數量: {len(parsed_result.get('references', []))}")
            
            debug_results['steps']['json_parsing'] = {
                'success': True,
                'question': parsed_result.get('question', ''),
                'references_count': len(parsed_result.get('references', []))
            }
            
            if 'references' in parsed_result:
                for i, ref in enumerate(parsed_result['references']):
                    print(f"  參考 {i+1}: {ref.get('content', 'NO CONTENT')[:100]}...")
                    if 'start_chunk' in ref:
                        print(f"    chunks: {ref['start_chunk']} - {ref['end_chunk']}")
            
        except Exception as e:
            print(f"❌ 解析失敗: {e}")
            debug_results['steps']['json_parsing'] = {
                'success': False,
                'error': str(e)
            }
            parsed_result = None
        
        # 8. 可選：保存到文件
        if save_to_file:
            debug_filename = f"debug_output_{corpus_id.replace('/', '_').replace('\\', '_')}.txt"
            with open(debug_filename, 'w', encoding='utf-8') as f:
                f.write("="*100 + "\n")
                f.write("完整輸入輸出調試報告 (增強版)\n")
                f.write("="*100 + "\n\n")
                f.write(f"文檔路徑: {corpus_id}\n")
                f.write(f"模式: {'APPROXIMATE' if use_approx else 'EXACT'}\n\n")
                f.write("調試結果摘要:\n")
                f.write(json.dumps(debug_results, indent=2, ensure_ascii=False) + "\n\n")
                f.write("完整 Prompt:\n")
                f.write(full_prompt + "\n\n")
                f.write("模型原始輸出:\n")
                f.write(full_response + "\n\n")
                f.write("提取的回應:\n")
                f.write(extracted_response + "\n")
            
            print(f"📁 調試結果已保存到: {debug_filename}")
        
        print("\n" + "="*100)
        print("🎯 調試完成")
        print("="*100)
        
        return {
            'document_content': corpus,
            'selected_fragment': document,
            'full_prompt': full_prompt,
            'model_full_response': full_response,
            'extracted_response': extracted_response,
            'parsed_result': parsed_result,
            'debug_results': debug_results
        }

    def repair_csv_references(self):
        """
        修復 CSV 中的雙重序列化問題
        """
        print("🔧 開始修復 CSV 中的 references 欄位...")
        
        if not os.path.exists(self.questions_csv_path):
            print("❌ CSV 文件不存在")
            return
        
        # 創建備份
        backup_path = self.questions_csv_path.replace('.csv', '_backup.csv')
        if not os.path.exists(backup_path):
            import shutil
            shutil.copy2(self.questions_csv_path, backup_path)
            print(f"✅ 已創建備份: {backup_path}")
        
        # 讀取並修復
        df = pd.read_csv(self.questions_csv_path)
        if 'corpus_id' in df.columns:
            df['corpus_id'] = df['corpus_id'].str.replace('\\', '/', regex=False)
            print("🔧 已修正 'corpus_id' 欄位中的路徑格式。")
        print(f"📊 讀取到 {len(df)} 條記錄")
        
        fixed_count = 0
        error_count = 0
        
        def fix_reference_field(refs_str):
            nonlocal fixed_count, error_count
            
            if pd.isna(refs_str) or refs_str == '':
                return json.dumps([], ensure_ascii=False)
            
            try:
                # 使用安全解析
                parsed_refs = self._safe_json_loads(refs_str)
                
                # 驗證解析結果
                if isinstance(parsed_refs, list):
                    # 驗證每個 reference 的格式
                    valid_refs = []
                    for ref in parsed_refs:
                        if isinstance(ref, dict) and 'content' in ref:
                            valid_refs.append(ref)
                    
                    if valid_refs:
                        fixed_count += 1
                        return json.dumps(valid_refs, ensure_ascii=False)
                    else:
                        error_count += 1
                        return json.dumps([], ensure_ascii=False)
                else:
                    error_count += 1
                    return json.dumps([], ensure_ascii=False)
                    
            except Exception as e:
                print(f"修復失敗: {e}")
                print(f"原始資料: {refs_str[:100]}...")
                error_count += 1
                return json.dumps([], ensure_ascii=False)
        
        print("正在修復 references 欄位...")
        df['references'] = df['references'].apply(fix_reference_field)
        
        # 保存修復後的文件
        df.to_csv(self.questions_csv_path, index=False)
        
        print(f"✅ 修復完成!")
        print(f"   成功修復: {fixed_count} 條")
        print(f"   修復失敗: {error_count} 條")
        print(f"   總計: {len(df)} 條")
        
        if error_count > 0:
            print(f"⚠️  有 {error_count} 條記錄無法修復，已設為空 list")

    def validate_csv_integrity(self):
        """
        驗證 CSV 文件的完整性
        """
        print("🔍 驗證 CSV 文件完整性...")
        
        if not os.path.exists(self.questions_csv_path):
            print("❌ CSV 文件不存在")
            return False
        
        try:
            if 'corpus_id' in df.columns:
                df['corpus_id'] = df['corpus_id'].str.replace('\\', '/', regex=False)
            df = pd.read_csv(self.questions_csv_path)
            print(f"📊 讀取到 {len(df)} 條記錄")
            
            # 檢查必要欄位
            required_columns = ['question', 'references', 'corpus_id']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                print(f"❌ 缺少必要欄位: {missing_columns}")
                return False
            
            # 檢查 references 欄位
            valid_count = 0
            invalid_count = 0
            
            for idx, refs_str in enumerate(df['references']):
                try:
                    parsed_refs = self._safe_json_loads(refs_str)
                    if isinstance(parsed_refs, list) and len(parsed_refs) > 0:
                        # 檢查第一個 reference 的格式
                        first_ref = parsed_refs[0]
                        if isinstance(first_ref, dict) and 'content' in first_ref:
                            valid_count += 1
                        else:
                            invalid_count += 1
                            if invalid_count <= 3:  # 只顯示前3個錯誤示例
                                print(f"❌ 第 {idx+1} 行格式錯誤: {first_ref}")
                    else:
                        invalid_count += 1
                        if invalid_count <= 3:
                            print(f"❌ 第 {idx+1} 行 references 為空或格式錯誤")
                except Exception as e:
                    invalid_count += 1
                    if invalid_count <= 3:
                        print(f"❌ 第 {idx+1} 行解析失敗: {e}")
            
            print(f"✅ 有效記錄: {valid_count}")
            print(f"❌ 無效記錄: {invalid_count}")
            
            if invalid_count == 0:
                print("🎉 CSV 文件完整性驗證通過!")
                return True
            else:
                print(f"⚠️  發現 {invalid_count} 條有問題的記錄")
                return False
                
        except Exception as e:
            print(f"❌ 驗證過程中發生錯誤: {e}")
            return False