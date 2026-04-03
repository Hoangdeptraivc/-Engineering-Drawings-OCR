import sys
from pathlib import Path
import io
# Thêm thư mục hiện tại vào path
sys.path.append(str(Path(__file__).parent))
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
from pipelines.pipelines_inference import InferencePipeline


def get_first_image_from_upload():
    """Lấy ảnh đầu tiên từ thư mục uploaded_images"""
    UPLOAD_IMAGES_DIR = Path(__file__).parent / "uploaded_images"

    if not UPLOAD_IMAGES_DIR.exists():
        print(f"⚠️ Thư mục {UPLOAD_IMAGES_DIR} không tồn tại")
        return None

    # Tìm tất cả file ảnh
    image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff', '*.PNG', '*.JPG', '*.JPEG']
    images = []

    for ext in image_extensions:
        images.extend(UPLOAD_IMAGES_DIR.glob(ext))

    if not images:
        print(f"⚠️ Không tìm thấy ảnh nào trong thư mục {UPLOAD_IMAGES_DIR}")
        return None

    # Trả về ảnh đầu tiên
    return images[0]


def main():
    """Main function để chạy pipeline"""

    print("🚀 Starting Engineering Drawing OCR Pipeline")
    print("=" * 50)

    # Khởi tạo pipeline
    pipeline = InferencePipeline()

    # In thông tin pipeline
    info = pipeline.get_pipeline_info()
    print(f"\n📋 Pipeline Info:")
    print(f"   Model classes: {info['model']['num_classes']}")
    print(f"   Input size: {info['model']['input_size']['width']}x{info['model']['input_size']['height']}")
    print(f"   Class mapping: {info['class_mapping']}")

    # Lấy ảnh từ thư mục uploaded_images
    print("\n📂 Đang tìm ảnh trong thư mục uploaded_images...")
    test_image_path = get_first_image_from_upload()

    if test_image_path is None:
        print("\n❌ Không có ảnh nào để xử lý!")
        print("   Vui lòng upload ảnh qua web app trước")
        return

    test_image = str(test_image_path)
    print(f"📸 Tìm thấy ảnh: {test_image_path.name}")
    print(f"   Đường dẫn: {test_image}")

    if Path(test_image).exists():
        print("\n🔄 Đang xử lý OCR...")
        result = pipeline.process_image(
            test_image,
            save_crops=True,
            save_vis=True,
            verbose=True
        )

        # In kết quả tóm tắt
        print("\n" + "=" * 50)
        print("📊 KẾT QUẢ XỬ LÝ:")
        print("=" * 50)
        print(f"  📸 Ảnh: {result['image']}")
        print(f"  🔍 Objects found: {result['num_objects']}")
        print(f"  ⏱️ Processing time: {result['processing_time_ms']} ms")

        if result['num_objects'] > 0:
            print("\n  📋 Chi tiết các đối tượng phát hiện:")
            for i, obj in enumerate(result['objects'], 1):
                print(f"    {i}. {obj['class']} (original: {obj['original_class']}, conf: {obj['confidence']:.3f})")
                if obj.get('crop_path'):
                    print(f"       📁 Crop saved: {obj['crop_path']}")
        else:
            print("\n  ⚠️ Không phát hiện đối tượng nào!")

        print("\n✅ XỬ LÝ THÀNH CÔNG!")

    else:
        print(f"\n❌ Lỗi: Ảnh không tồn tại - {test_image}")


if __name__ == "__main__":
    main()