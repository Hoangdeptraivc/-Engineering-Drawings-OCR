import json
from pathlib import Path
from paddleocr import PaddleOCR
from PIL import Image
import cv2
import numpy as np
from tqdm import tqdm

# Khởi tạo OCR cho cả Việt và Anh
ocr_vi = PaddleOCR(
    use_angle_cls=True,
    lang='vi',
    show_log=False,
    use_gpu=True,
)

ocr_en = PaddleOCR(
    use_angle_cls=True,
    lang='en',
    show_log=False,
    use_gpu=True,
)


def ocr_cell(image_path):
    """OCR cho một ảnh cell, trả về text và confidence tốt nhất"""
    try:
        # Thử với cả 2 model
        result_vi = ocr_vi.ocr(str(image_path), cls=True)
        result_en = ocr_en.ocr(str(image_path), cls=True)

        best_text = ""
        best_conf = 0

        # Xử lý kết quả từ model Việt
        if result_vi and result_vi[0]:
            for line in result_vi[0]:
                text = line[1][0]
                conf = line[1][1]
                if conf > best_conf:
                    best_conf = conf
                    best_text = text

        # Xử lý kết quả từ model Anh và so sánh
        if result_en and result_en[0]:
            for line in result_en[0]:
                text = line[1][0]
                conf = line[1][1]
                if conf > best_conf:
                    best_conf = conf
                    best_text = text

        return best_text, best_conf
    except Exception as e:
        print(f"Lỗi OCR {image_path}: {e}")
        return "", 0


def process_all_jsons():
    # Đường dẫn
    path_out1 = Path(r"C:\Users\vanho\PycharmProjects\pythonProject2\Engineering Drawings\outputs")
    path_cropped_model2 = Path(
        r"C:\Users\vanho\PycharmProjects\pythonProject2\Engineering Drawings\outputs\cropped_model2")

    # Tìm tất cả JSON files
    json_files_model1 = list(path_out1.glob("*.json"))
    json_files_model2 = list(path_cropped_model2.glob("*_metadata.json"))

    print(f"Tìm thấy {len(json_files_model1)} JSON từ Model 1")
    print(f"Tìm thấy {len(json_files_model2)} JSON từ Model 2")

    # Kết quả tổng hợp
    final_result = {
        "metadata": {
            "total_model1_jsons": len(json_files_model1),
            "total_model2_jsons": len(json_files_model2),
            "ocr_languages": ["vi", "en"]
        },
        "model1_results": [],
        "model2_results": []
    }

    # XỬ LÝ MODEL 1 - Chỉ lấy Text và List-item
    print("\n=== XỬ LÝ MODEL 1 (Text & List-item) ===")
    for json_path in tqdm(json_files_model1, desc="Model 1"):
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        processed_objects = []

        for obj in data.get('objects', []):
            obj_class = obj.get('class', '')

            # Chỉ xử lý nếu là Text hoặc List-item
            if obj_class in ['Text', 'List-item']:
                crop_path = obj.get('crop_path')
                if crop_path and Path(crop_path).exists():
                    text, confidence = ocr_cell(crop_path)
                    obj['ocr_text'] = text
                    obj['ocr_confidence'] = confidence
                    print(f"  {obj_class}: '{text}' (conf: {confidence:.2f})")
                else:
                    obj['ocr_text'] = ""
                    obj['ocr_confidence'] = 0
                    print(f"  ⚠️ Không tìm thấy ảnh: {crop_path}")

            processed_objects.append(obj)

        data['objects'] = processed_objects
        final_result['model1_results'].append(data)

    # XỬ LÝ MODEL 2 - Xử lý tất cả cells
    print("\n=== XỬ LÝ MODEL 2 (Tất cả cells) ===")
    for json_path in tqdm(json_files_model2, desc="Model 2"):
        with open(json_path, 'r', encoding='utf-8') as f:
            cells_data = json.load(f)

        # cells_data có thể là list hoặc dict
        if isinstance(cells_data, dict):
            cells = cells_data.get('cells', [])
        else:
            cells = cells_data

        processed_cells = []

        for cell in cells:
            crop_path = cell.get('cropped_image_path')
            if crop_path and Path(crop_path).exists():
                text, confidence = ocr_cell(crop_path)
                cell['ocr_text'] = text
                cell['ocr_confidence'] = confidence
                print(f"  Cell {cell.get('cell_index', '?')}: '{text}' (conf: {confidence:.2f})")
            else:
                cell['ocr_text'] = ""
                cell['ocr_confidence'] = 0

            processed_cells.append(cell)

        if isinstance(cells_data, dict):
            cells_data['cells'] = processed_cells
            final_result['model2_results'].append(cells_data)
        else:
            final_result['model2_results'].append(processed_cells)

    # LƯU KẾT QUẢ
    output_dir = Path(r"C:\Users\vanho\PycharmProjects\pythonProject2\Engineering Drawings\outputs\ocr_results")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Lưu file JSON tổng hợp
    final_json_path = output_dir / "all_ocr_results.json"
    with open(final_json_path, 'w', encoding='utf-8') as f:
        json.dump(final_result, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Đã lưu kết quả tổng hợp tại: {final_json_path}")

    # Tạo file JSON riêng cho web (đơn giản hơn)
    web_json_path = output_dir / "web_demo_results.json"
    web_data = {
        "total_ocr_cells": 0,
        "results": []
    }

    # Gom tất cả kết quả OCR từ model 1
    for data in final_result['model1_results']:
        for obj in data.get('objects', []):
            if 'ocr_text' in obj and obj['ocr_text']:
                web_data['results'].append({
                    "source": "model1",
                    "class": obj.get('class'),
                    "bbox": obj.get('bbox'),
                    "text": obj['ocr_text'],
                    "confidence": obj.get('ocr_confidence', 0),
                    "image_path": obj.get('crop_path')
                })

    # Gom tất cả kết quả OCR từ model 2
    for cells in final_result['model2_results']:
        if isinstance(cells, dict):
            cells_list = cells.get('cells', [])
        else:
            cells_list = cells

        for cell in cells_list:
            if 'ocr_text' in cell and cell['ocr_text']:
                web_data['results'].append({
                    "source": "model2",
                    "class": "TableCell",
                    "bbox": cell.get('bbox'),
                    "text": cell['ocr_text'],
                    "confidence": cell.get('ocr_confidence', 0),
                    "image_path": cell.get('cropped_image_path')
                })

    web_data['total_ocr_cells'] = len(web_data['results'])

    with open(web_json_path, 'w', encoding='utf-8') as f:
        json.dump(web_data, f, ensure_ascii=False, indent=2)

    print(f"✅ Đã lưu file cho web demo tại: {web_json_path}")
    print(f"📊 Tổng số cells có OCR: {web_data['total_ocr_cells']}")

    return final_result


# Chạy xử lý
if __name__ == "__main__":
    result = process_all_jsons()