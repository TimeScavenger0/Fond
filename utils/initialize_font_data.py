import json
import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# parent directory of CURRENT_DIR
BASE_DIR = os.path.dirname(os.path.dirname(CURRENT_DIR))
BASE_DIR = os.path.dirname(CURRENT_DIR)

fox_text = "The quick brown\nfox jumps over\nthe lazy dog"
fox_text_four_lines = "The quick\nfox jumps over\nthe lazy dog"

font_dir = os.path.join(BASE_DIR, "gwfonts")
all_gray_scale_image_file_dir = os.path.join(
    BASE_DIR, "gwfonts_images"
)
all_json_path = os.path.join(
    BASE_DIR, "attributeData/all_font_to_attribute_values.json"
)
train_json_path = os.path.join(
    BASE_DIR,
    "attributeData/train_font_to_attribute_values.json",
)
test_json_path = os.path.join(
    BASE_DIR,
    "attributeData/test_font_to_attribute_values.json",
)
validation_json_path = os.path.join(
    BASE_DIR,
    "attributeData/validation_font_to_attribute_values.json",
)
all_gwfonts_json_path = os.path.join(BASE_DIR, "attributeData/all_font_to_attribute_values.json")


def _load_json_safe(path):
    """文件存在则正常加载；缺失或不可读时返回空 dict，不使导入崩溃。"""
    try:
        with open(path, "r") as f:
            return json.load(f)
    except (FileNotFoundError, OSError):
        return {}


# 拉丁全套数据：本地完整部署时正常加载；云端最小部署（仅混合训练）缺这些文件时
# 安全降级为空。train_cn_mixed 在 Config 中完全覆盖数据路径，不依赖这里的全局数据。
all_json = _load_json_safe(all_json_path)
train_json = _load_json_safe(train_json_path)
test_json = _load_json_safe(test_json_path)
validation_json = _load_json_safe(validation_json_path)
train_font_names = list(train_json.keys())
test_font_names = list(test_json.keys())
validation_font_names = list(validation_json.keys())
all_font_names = list(all_json.keys())
font_names = list(all_json.keys())

try:
    all_gwfonts_names = [
        os.path.basename(os.path.splitext(font_file_name)[0])
        for font_file_name in os.listdir(font_dir)
    ]
    all_gwfonts_names = [
        font_name for font_name in all_gwfonts_names if font_name not in ['.DS_Store']
    ]
except (FileNotFoundError, OSError):
    all_gwfonts_names = []


def retrieve_font_path(font_name: str, font_dir: str) -> str:
    """
    Retrieve font path from font name

    Parameters
    ----------
    font_name : str
    font_dir : str

    Returns
    -------
    font_path : str
    """
    font_paths = os.listdir(font_dir)
    for font_path in font_paths:
        if font_name == os.path.splitext(os.path.basename(font_path))[0]:
            return os.path.join(font_dir, font_path)


try:
    font_paths = [
        retrieve_font_path(font_name, font_dir=font_dir) for font_name in font_names
    ]
except (FileNotFoundError, OSError):
    font_paths = []


# 项目固定的 37 个属性（内置兜底，顺序锁定）
_FALLBACK_ATTRIBUTES = [
    "angular", "artistic", "attention-grabbing", "attractive", "bad",
    "boring", "calm", "capitals", "charming", "clumsy", "complex",
    "cursive", "delicate", "disorderly", "display", "dramatic", "formal",
    "fresh", "friendly", "gentle", "graceful", "happy", "italic",
    "legible", "modern", "monospace", "playful", "pretentious", "serif",
    "sharp", "sloppy", "soft", "strong", "technical", "thin", "warm", "wide",
]


def _resolve_all_attributes():
    """
    属性名集合必须完整（dataset 用它做索引、矩阵宽度与 assert）。
    依次回退：拉丁 train_json（本地原始行为）→ 混合训练 JSON（云端）→ 内置 37 属性。
    """
    candidate_paths = [
        train_json_path,
        os.path.join(BASE_DIR, "data_cn", "mixed",
                     "mixed_font_to_attribute_values.json"),
    ]
    for path in candidate_paths:
        try:
            with open(path, "r") as f:
                data = json.load(f)
            attrs = list(list(data.values())[0].keys())
            if attrs:
                return attrs
        except (FileNotFoundError, OSError, IndexError):
            continue
    return list(_FALLBACK_ATTRIBUTES)


# Attributes
all_attributes = _resolve_all_attributes()
exclusive_attributes = [
    "capitals",
    # "cursive",
    # "display",
    # "italic",
    "monospace",
    # "serif",
]
# exclusive_attributes = []
print("exclusive_attributes: ", exclusive_attributes)
inclusive_attributes = [
    attr for attr in all_attributes if attr not in exclusive_attributes
]
