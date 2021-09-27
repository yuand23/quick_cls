## 基于预训练语言模型的通用分类器（目前只支持单句子短文本分类）
* to train: python quick_cls.py train
* to test: python quick_cls.py test
* to test single text: python quick_cls.py test_single
## 数据格式
train.csv, val.csv, test.csv  
第一行为headers，每一行为“text,tag”  

