import base64
import io
import os
from pathlib import Path

from PIL import Image

# 定义图片扩展名识别规则
image_extensions = ('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp')

def image_to_bytes(image):
    byte_arr = io.BytesIO()
    image.save(byte_arr, format='JPEG')  # 可以选择不同的格式，例如'JPEG'或'PNG'
    byte_data = byte_arr.getvalue()

    # 将字节数据编码为Base64
    return base64.b64encode(byte_data).decode('utf-8')

def load_from_file(folder_path: str):
    file_names = []
    if os.path.isdir(folder_path):
        for filename in os.listdir(folder_path):
            if os.path.isfile(os.path.join(folder_path, filename)):
                file_names.append(filename)
    elif os.path.splitext(folder_path)[1] in ['.jpg', 'jpg', '.png', 'png', 'jpeg', '.jpeg']:
        file_names.append(folder_path)
    else:
        raise FileNotFoundError
    images = []
    file_paths = []
    for filename in file_names:
        full_path = os.path.join(folder_path, filename)
        file_paths.append(full_path)
        images.append(Image.open(full_path).convert('RGB'))
    return file_paths, images

def genertate_output_md(template_path, output_path, doc_cls):
    """
    根据模板生成类的Markdown文档

    :param template_path: 模板文件路径
    :param output_path: 输出文件路径
    :param cls: 要生成文档的类
    """
    output_dict = {}
    keys = doc_cls.keys()
    # 遍历所有属性并筛选目标列表
    for attr_name in keys:
        output_dict.update({attr_name: get_content(doc_cls, attr_name)})
    # 读取模版内容
    with open(template_path, 'r', encoding='utf-8') as file:
        template = file.read()
    formatted_content = template.format_map(
        output_dict
    )

    # 读取模版内容
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(formatted_content)


def get_content(doc_data, obj):
    output_list = []
    for section in doc_data[obj]:
        # 获取文本
        if section["type"] == 'txt':
            if isinstance(section['content'], str) and len(section['content'].strip()) > 0:
                output_list.append(str(section['content']).strip())
        elif section["type"] == 'img':
            # 获取图片
            if isinstance(section['content'], str) and len(section['content'].strip()) > 0:
                dirpath = section['content']
                if isinstance(dirpath, str) and len(dirpath.strip()) > 0:
                    # 路径是否存在
                    if Path(dirpath).exists() and Path(dirpath).is_file() and Path(
                            dirpath).suffix.lower() in image_extensions:
                        # 图片路径处理（生成 Markdown 图片语法）
                        output_list.append(f"![image]({dirpath})")
                    else:
                        print("路径无效或文件未找到")
    return "\n\n".join(output_list)
