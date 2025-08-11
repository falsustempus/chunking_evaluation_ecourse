import os
import re
from collections import defaultdict
from pathlib import Path
import pandas as pd
from bs4 import BeautifulSoup

class DatasetAnalyzer:
    def __init__(self, folder_path: str):
        self.folder_path = folder_path
        self.file_stats = []
        self.category_stats = defaultdict(int)
        
    def _clean_html(self, html_content: str) -> str:
        """清理HTML內容，提取純文本"""
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            # 移除script和style標籤
            for script_or_style in soup(['script', 'style']):
                script_or_style.decompose()
            # 提取純文本
            clean_text = soup.get_text()
            # 清理多餘空白
            clean_text = ' '.join(clean_text.split())
            return clean_text
        except Exception as e:
            print(f"HTML清理失敗: {e}")
            return html_content

    def _clean_text(self, text_content: str) -> str:
        """清理純文本內容"""
        return re.sub(r'\s+', ' ', text_content).strip()

    def _get_file_content_and_word_count(self, file_path: str) -> tuple:
        """讀取檔案內容並計算字數"""
        file_extension = os.path.splitext(file_path)[1].lower()
        
        try:
            # 嘗試UTF-8編碼
            with open(file_path, 'r', encoding='utf-8') as f:
                raw_content = f.read()
        except UnicodeDecodeError:
            try:
                # 嘗試latin-1編碼
                with open(file_path, 'r', encoding='latin-1') as f:
                    raw_content = f.read()
            except Exception as e:
                print(f"❌ 無法讀取檔案 {file_path}: {e}")
                return "", 0
        except Exception as e:
            print(f"❌ 讀取檔案失敗 {file_path}: {e}")
            return "", 0

        # 根據副檔名清理內容
        if file_extension in ['.html', '.htm']:
            cleaned_content = self._clean_html(raw_content)
        elif file_extension in ['.txt', '.md', '.markdown']:
            cleaned_content = self._clean_text(raw_content)
        else:
            # 未知格式，嘗試當作純文本處理
            cleaned_content = self._clean_text(raw_content)
        
        # 計算字數（中英文混合）
        # 中文字符計數
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', cleaned_content))
        # 英文單詞計數
        english_words = len(re.findall(r'\b[a-zA-Z]+\b', cleaned_content))
        # 數字計數
        numbers = len(re.findall(r'\b\d+\b', cleaned_content))
        
        total_word_count = chinese_chars + english_words + numbers
        
        return cleaned_content, total_word_count

    def _extract_category_from_filename(self, filename: str) -> str:
        """從檔名提取分類（第二個底線到第三個底線）"""
        try:
            # 移除副檔名
            name_without_ext = os.path.splitext(filename)[0]
            # 以底線分割
            parts = name_without_ext.split('_')
            
            if len(parts) >= 4:
                # 提取第二個底線到第三個底線的部分（索引2）
                category = parts[2]
                return category
            else:
                return "未分類"
        except Exception as e:
            print(f"⚠️ 無法解析檔名分類 {filename}: {e}")
            return "解析錯誤"

    def analyze_folder(self):
        """分析整個資料夾"""
        print(f"🔍 開始分析資料夾: {self.folder_path}")
        print("="*60)
        
        # 取得所有檔案路徑
        folder_path_obj = Path(self.folder_path)
        if not folder_path_obj.exists():
            print(f"❌ 資料夾不存在: {self.folder_path}")
            return
        
        # 找出所有檔案
        all_files = list(folder_path_obj.rglob('*'))
        file_paths = [f for f in all_files if f.is_file()]
        
        total_files = len(file_paths)
        print(f"📁 找到 {total_files} 個檔案")
        print()
        
        # 逐一處理每個檔案
        for i, file_path in enumerate(file_paths, 1):
            self._process_single_file(file_path, i, total_files)
            
            # 每處理50個檔案顯示一次進度
            if i % 50 == 0:
                print(f"📊 已處理 {i}/{total_files} 個檔案 ({i/total_files*100:.1f}%)")
        
        print(f"\n✅ 處理完成！總共處理了 {total_files} 個檔案")
        
    def _process_single_file(self, file_path: Path, current_idx: int, total_files: int):
        """處理單一檔案"""
        try:
            filename = file_path.name
            file_extension = file_path.suffix.lower()
            file_size = file_path.stat().st_size
            
            # 顯示當前處理的檔案
            if current_idx % 10 == 1:  # 每10個檔案顯示一次
                print(f"📄 [{current_idx}/{total_files}] 處理: {filename}")
            
            # 提取分類
            category = self._extract_category_from_filename(filename)
            self.category_stats[category] += 1
            
            # 讀取內容並計算字數
            content, word_count = self._get_file_content_and_word_count(str(file_path))
            
            # 記錄檔案統計資訊
            file_info = {
                'filename': filename,
                'file_path': str(file_path),
                'file_extension': file_extension,
                'file_size_bytes': file_size,
                'file_size_kb': round(file_size / 1024, 2),
                'word_count': word_count,
                'category': category,
                'content_length': len(content)
            }
            
            self.file_stats.append(file_info)
            
        except Exception as e:
            print(f"❌ 處理檔案時發生錯誤 {file_path}: {e}")

    def generate_report(self):
        """生成分析報告"""
        if not self.file_stats:
            print("❌ 沒有可用的統計資料")
            return
        
        print("\n" + "="*80)
        print("📊 資料集分析報告")
        print("="*80)
        
        # 基本統計
        total_files = len(self.file_stats)
        print(f"📁 檔案總數: {total_files}")
        
        # 檔案格式統計
        print("\n🎯 檔案格式分佈:")
        format_stats = defaultdict(int)
        for file_info in self.file_stats:
            format_stats[file_info['file_extension']] += 1
        
        for ext, count in sorted(format_stats.items()):
            percentage = count / total_files * 100
            print(f"   {ext or '無副檔名'}: {count} 個 ({percentage:.1f}%)")
        
        # 檔案大小統計
        print("\n📏 檔案大小統計:")
        sizes = [info['file_size_kb'] for info in self.file_stats]
        print(f"   平均大小: {sum(sizes)/len(sizes):.2f} KB")
        print(f"   最大檔案: {max(sizes):.2f} KB")
        print(f"   最小檔案: {min(sizes):.2f} KB")
        
        # 字數統計
        print("\n📝 字數統計:")
        word_counts = [info['word_count'] for info in self.file_stats]
        print(f"   平均字數: {sum(word_counts)/len(word_counts):.0f} 字")
        print(f"   最多字數: {max(word_counts):,} 字")
        print(f"   最少字數: {min(word_counts):,} 字")
        print(f"   總字數: {sum(word_counts):,} 字")
        
        # 分類統計（按檔案數量排序）
        print(f"\n🏫 分類統計 (共 {len(self.category_stats)} 個分類):")
        sorted_categories = sorted(self.category_stats.items(), key=lambda x: x[1], reverse=True)
        
        for category, count in sorted_categories:
            percentage = count / total_files * 100
            print(f"   {category}: {count} 筆 ({percentage:.1f}%)")
        
        print("\n" + "="*80)

    def save_detailed_report(self, output_path: str = "dataset_analysis_report.csv"):
        """保存詳細報告到CSV檔案"""
        if not self.file_stats:
            print("❌ 沒有可用的統計資料")
            return
        
        try:
            df = pd.DataFrame(self.file_stats)
            df.to_csv(output_path, index=False, encoding='utf-8-sig')
            print(f"✅ 詳細報告已保存到: {output_path}")
        except Exception as e:
            print(f"❌ 保存報告失敗: {e}")

    def save_category_summary(self, output_path: str = "category_summary.csv"):
        """保存分類摘要到CSV檔案"""
        if not self.category_stats:
            print("❌ 沒有分類統計資料")
            return
        
        try:
            # 準備分類摘要資料
            category_data = []
            total_files = len(self.file_stats)
            
            for category, count in sorted(self.category_stats.items(), key=lambda x: x[1], reverse=True):
                percentage = count / total_files * 100
                category_data.append({
                    'category': category,
                    'file_count': count,
                    'percentage': round(percentage, 2)
                })
            
            df = pd.DataFrame(category_data)
            df.to_csv(output_path, index=False, encoding='utf-8-sig')
            print(f"✅ 分類摘要已保存到: {output_path}")
            
        except Exception as e:
            print(f"❌ 保存分類摘要失敗: {e}")

    def get_stratified_sample(self, total_sample_size: int = 100, min_per_category: int = 1):
        """根據分類比例進行分層抽樣"""
        if not self.file_stats:
            print("❌ 沒有可用的統計資料")
            return []
        
        print(f"\n🎯 進行分層抽樣 (目標樣本數: {total_sample_size})")
        
        # 按分類分組檔案
        files_by_category = defaultdict(list)
        for file_info in self.file_stats:
            files_by_category[file_info['category']].append(file_info)
        
        # 計算每個分類應該抽取的數量
        total_files = len(self.file_stats)
        sampled_files = []
        remaining_sample_size = total_sample_size
        
        sorted_categories = sorted(self.category_stats.items(), key=lambda x: x[1], reverse=True)
        
        for i, (category, count) in enumerate(sorted_categories):
            if remaining_sample_size <= 0:
                break
                
            # 計算該分類的抽樣數量
            if i == len(sorted_categories) - 1:  # 最後一個分類，用剩餘的全部
                sample_size = remaining_sample_size
            else:
                # 按比例計算
                proportion = count / total_files
                sample_size = max(min_per_category, int(total_sample_size * proportion))
                sample_size = min(sample_size, count, remaining_sample_size)
            
            # 從該分類隨機抽樣
            import random
            category_files = files_by_category[category]
            if len(category_files) >= sample_size:
                selected = random.sample(category_files, sample_size)
            else:
                selected = category_files  # 如果檔案數不足，全部選取
            
            sampled_files.extend(selected)
            remaining_sample_size -= len(selected)
            
            print(f"   {category}: {len(selected)}/{count} 個檔案")
        
        print(f"\n✅ 完成分層抽樣，共抽取 {len(sampled_files)} 個檔案")
        return sampled_files

# 使用範例
if __name__ == "__main__":
    # 指定要分析的資料夾路徑
    folder_path = "./corpora"
    
    # 創建分析器
    analyzer = DatasetAnalyzer(folder_path)
    
    # 執行分析
    analyzer.analyze_folder()
    
    # 生成報告
    analyzer.generate_report()
    
    # 保存詳細報告
    analyzer.save_detailed_report("dataset_analysis_detailed.csv")
    analyzer.save_category_summary("dataset_analysis_categories.csv")
    
    # 進行分層抽樣
    sample = analyzer.get_stratified_sample(total_sample_size=100)
    
    # 保存抽樣結果
    if sample:
        sample_df = pd.DataFrame(sample)
        sample_df.to_csv("stratified_sample_100.csv", index=False, encoding='utf-8-sig')
        print("✅ 分層抽樣結果已保存到: stratified_sample_100.csv")