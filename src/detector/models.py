import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import json
import torch
torch.cuda.empty_cache()
from PIL import Image
from transformers import AutoImageProcessor, DetrForSegmentation,TableTransformerForObjectDetection
from pathlib import Path
import sys
from torchvision.ops import nms
from typing import  Union, List, Dict, Any, Optional
from collections import defaultdict

sys.path.append(str(Path(__file__).parent.parent))

from config.config import (

    PREPROCESSOR_CFG, IMAGE_SIZE
)


class BaseDetector:
    """Base class cho tất cả detectors"""

    def __init__(self, model_path=None, threshold=None):
        """
        Khởi tạo detector

        Args:
            model_path: Đường dẫn đến thư mục chứa model
            threshold: Ngưỡng confidence cho detection
        """
        self.model_path = Path(model_path)
        self.threshold = threshold






    def _setup_device(self):
        """Setup device"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        print(f"💻 Device: {self.device}")

    def preprocess(self, image: Union[str, Path, Image.Image]) -> Dict[str, torch.Tensor]:
        """
        TIỀN XỬ LÝ ẢNH - CHILD CLASS OVERRIDE

        Args:
            image: PIL Image hoặc đường dẫn đến ảnh

        Returns:
            Dict với các tensor đã được xử lý
        """
        # Load ảnh nếu là đường dẫn
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")

        # Mặc định dùng processor của transformers
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        return inputs

    def postprocess(self, outputs, original_size: tuple) -> List[Dict]:
        """
        HẬU XỬ LÝ - CHILD CLASS OVERRIDE

        Args:
            outputs: Output từ model
            original_size: (height, width) của ảnh gốc

        Returns:
            List các detections
        """
        results = self.processor.post_process_object_detection(
            outputs,
            threshold=self.threshold,
            target_sizes=[original_size]
        )[0]

        return self._format_detections(results)

    def _format_detections(self, results: Dict) -> List[Dict]:
        """Format kết quả detection"""
        detections = []
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            detections.append({
                "class_id": int(label),
                "class_name": self.classes[label],
                "confidence": float(score),
                "bbox": [int(x) for x in box.tolist()]
            })

        # Apply NMS
        detections = self._apply_nms(detections)

        return detections

    def _apply_nms(self, detections: List[Dict], iou_threshold: float = 0.5) -> List[Dict]:
        """Áp dụng NMS theo từng class"""
        boxes_by_class = defaultdict(list)

        for det in detections:
            boxes_by_class[det["class_name"]].append(det)

        final_detections = []

        for class_name, items in boxes_by_class.items():
            if not items:
                continue

            bboxes = [item["bbox"] for item in items]
            scores = [item["confidence"] for item in items]

            keep_indices = self._nms(bboxes, scores, iou_threshold)

            for idx in keep_indices:
                final_detections.append(items[idx])

        final_detections.sort(key=lambda x: x["confidence"], reverse=True)
        return final_detections

    def _nms(self, boxes, scores, iou_threshold=0.5):
        """Helper NMS"""
        if len(boxes) == 0:
            return []
        boxes_t = torch.tensor(boxes, dtype=torch.float32)
        scores_t = torch.tensor(scores, dtype=torch.float32)
        keep = nms(boxes_t, scores_t, iou_threshold)
        return keep.tolist()

    def detect(self, image: Union[str, Path, Image.Image]) -> List[Dict]:
        """
        Phát hiện object trong ảnh - METHOD CHÍNH, KHÔNG CẦN OVERRIDE

        Args:
            image: PIL Image hoặc đường dẫn đến ảnh

        Returns:
            List các object phát hiện được
        """
        # Lấy original size
        if isinstance(image, (str, Path)):
            pil_image = Image.open(image).convert("RGB")
        else:
            pil_image = image

        original_size = pil_image.size[::-1]  # (height, width)

        # Preprocess (CHILD CLASS CÓ THỂ OVERRIDE)
        inputs = self.preprocess(pil_image)

        # Inference
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Postprocess (CHILD CLASS CÓ THỂ OVERRIDE)
        detections = self.postprocess(outputs, original_size)

        return detections



    def __call__(self, image):
        """Gọi trực tiếp detector"""
        return self.detect(image)

class LayoutDetector(BaseDetector):
    """Wrapper cho DETR layout detection model"""

    def __init__(self, model_path=None, threshold=None, verbose=False):
        self.verbose = verbose
        super().__init__(model_path, threshold)
        self._load_classes()
        self._load_model()

    def _load_model(self):
        """Load DETR model với segmentation"""
        self.processor = AutoImageProcessor.from_pretrained(str(self.model_path))
        self.model = DetrForSegmentation.from_pretrained(str(self.model_path))
        self._setup_device()


    def _load_classes(self):
        """Load class names từ config.json"""
        config_path = self.model_path / "config.json"
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        id2label = config.get("id2label", {})
        self.classes = [id2label[str(i)] for i in range(len(id2label))]
        self.num_classes = len(self.classes)

        print(f"📋 Loaded {self.num_classes} classes from config")




    def get_model_info(self):
        """Lấy thông tin về model"""
        return {
            "model_path": str(self.model_path),
            "num_classes": self.num_classes,
            "classes": self.classes,
            "input_size": IMAGE_SIZE,
            "threshold": self.threshold,
            "device": str(self.device)
        }


class TableDetector(BaseDetector):
    """TableTransformer Model - chỉ cần override xử lý ảnh"""

    def __init__(self, model_path=None, threshold=None):
        super().__init__(model_path, threshold)
        self.classes = []
        self.num_classes = 0
        self._load_classes()
        self._load_model()

    def _load_classes(self):
        """Load class names từ config.json của TableTransformer"""
        try:
            config_path = self.model_path / "config.json"
            if not config_path.exists():
                print(f"⚠️ Config not found at {config_path}")
                self._set_default_classes()
                return

            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)

            # Lấy id2label từ cấp cao nhất (không phải trong backbone_config)
            id2label = config.get("id2label", {})

            if id2label:
                # Chuyển đổi id2label thành list classes theo đúng thứ tự
                # Tìm max key để biết số lượng classes
                max_key = max([int(k) for k in id2label.keys()]) if id2label else 0
                self.classes = [id2label.get(str(i), f"class_{i}") for i in range(max_key + 1)]
                self.num_classes = len(self.classes)
                print(f"📋 Loaded {self.num_classes} classes from config:")
                for i, cls in enumerate(self.classes):
                    print(f"   {i}: {cls}")
            else:
                # Fallback classes cho table detection
                print("⚠️ No id2label found in config, using default classes")
                self._set_default_classes()

        except Exception as e:
            print(f"⚠️ Error loading classes: {e}")
            self._set_default_classes()

    def _set_default_classes(self):
        """Set default classes cho table detection"""
        self.classes = [
            "table",
            "table column",
            "table row",
            "table column header",
            "table projected row header",
            "table spanning cell"
        ]
        self.num_classes = len(self.classes)
        print(f"📋 Using default {self.num_classes} classes")
    def _load_model(self):
        """Override: Load TableTransformer thay vì DETR"""
        self.processor = AutoImageProcessor.from_pretrained(str(self.model_path))
        self.model = TableTransformerForObjectDetection.from_pretrained(str(self.model_path))
        self._setup_device()