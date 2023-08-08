from numpy.linalg import norm
from numpy import dot
from transformers import BertTokenizer, TFBertModel, AutoTokenizer, AutoModel
from langchain.text_splitter import RecursiveCharacterTextSplitter
from config.logger import logger
from config.config import config
from elasticsearch import Elasticsearch, helpers
from gensim.models import FastText


# 코사인 유사도를 구하는 함수
def cos_sim(a, b):
    return dot(a, b)/(norm(a)*norm(b))

# 스크립트를 실행하려면 여백의 녹색 버튼을 누릅니다.
if __name__ == '__main__':

    # tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-uncased")  # English and Korean BERT (소문자로 변경 후 처리)
    # model = AutoModel.from_pretrained("bert-base-multilingual-uncased")

    # tokenizer = AutoTokenizer.from_pretrained("skt/kobert-base-v1")  # English and Korean BERT (소문자로 변경 후 처리)
    # model = AutoModel.from_pretrained("skt/kobert-base-v1")

    tokenizer = AutoTokenizer.from_pretrained("snunlp/KR-BERT-char16424")
    model = AutoModel.from_pretrained("snunlp/KR-BERT-char16424")

    t1 = "아빠가 밥을 먹습니다."
    t2 = "아버지가 식사를 합니다."
    t3 = "엄마가 일을 합니다."

    # t1 = "아빠"
    # t2 = "아버지"
    # t3 = "엄마"

    # 문장 임베딩
    te1 = model(tokenizer.encode(t1, return_tensors='pt'))[0].squeeze(0).mean(dim=0).detach().numpy()
    te2 = model(tokenizer.encode(t2, return_tensors='pt'))[0].squeeze(0).mean(dim=0).detach().numpy()
    te3 = model(tokenizer.encode(t3, return_tensors='pt'))[0].squeeze(0).mean(dim=0).detach().numpy()

    print(cos_sim(te1, te1))
    print(cos_sim(te1, te2))
    print(cos_sim(te1, te3))
    print(cos_sim(te2, te3))
