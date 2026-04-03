import json
import os
import glob
from typing import List
from pathlib import Path

def get_table_image_paths(folder_path) ->List[str]:
    table_paths = []
    folder = Path(folder_path)

    # Kiểm tra folder có tồn tại không
    if not folder.exists():
        print(f"❌ Folder không tồn tại: {folder_path}")
        return table_paths

    # Lấy tất cả file .json trong folder (không đệ quy)
    json_files = list(folder.glob("*.json"))

    print(f"📁 Tìm thấy {len(json_files)} file JSON trong folder.")

    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as file:
                data = json.load(file)

            # Kiểm tra có key "objects"
            if isinstance(data, dict) and "objects" in data:
                for obj in data.get("objects", []):
                    if isinstance(obj, dict):
                        # Kiểm tra cả class và original_class
                        obj_class = obj.get("class") or obj.get("original_class") or ""

                        if obj_class.lower() == "table":
                            crop_path = obj.get("crop_path")
                            if crop_path and isinstance(crop_path, str):
                                table_paths.append(crop_path)

        except json.JSONDecodeError:
            print(f"⚠️  File JSON bị lỗi: {json_file.name}")
        except Exception as e:
            print(f"⚠️  Lỗi khi xử lý file {json_file.name}: {e}")

    print(f"✅ Hoàn thành! Tìm thấy {len(table_paths)} bảng (Table).")
    return table_paths




