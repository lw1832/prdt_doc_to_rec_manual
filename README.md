## 整体介绍

- 按照requirements或者requirements-cpu配置环境，python版本推荐>3.10
- 代码主体在prdt_doc_to_rec_manual\project\doc_to_recommendation\prdt_doc_to_rec_md.py
- configs/ocr_layout.yaml是ocr+yolo任务配置文件
- llm部分需要额外配置token，具体在prdt_doc_to_rec_md.py里看
- 输出默认在assets中，输出文件格式为md文件和一堆图片截图