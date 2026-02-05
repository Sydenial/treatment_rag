import asyncio
import json
import os
import openai
from openai import AsyncOpenAI
from typing import List, Dict, Any
from tqdm import tqdm
from pathlib import Path
import json
import re
import pandas as pd

md_dir = '/home/syd/remote/rag/case_report_data'
new_dir = '/home/syd/remote/rag/case_report_data_cleaned'
output_jsonl_path = '/home/syd/remote/rag/cleaned_output.jsonl'

with open(output_jsonl_path, 'r', encoding='utf-8') as f:
    for line in f:
        result = json.loads(line)
        md_path = Path(result['custom_id'])
        a1, a2 = md_path.parts[-2], md_path.parts[-3]
        os.makedirs(os.path.join(new_dir, a2, a1), exist_ok=True)
        save_path = os.path.join(new_dir, a2, a1, 'pubmed_pdf.md')
        try:
            if result['success']:
                with open(save_path, "w", encoding="utf-8") as f2:
                    f2.write(result['response'])
                print(f"成功保存文件：{save_path}")
        except:
            print(f"{save_path}失败。")

