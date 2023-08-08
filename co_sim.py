from numpy.linalg import norm
from numpy import dot
from transformers import BertTokenizer, TFBertModel, AutoTokenizer, AutoModel
from langchain.text_splitter import RecursiveCharacterTextSplitter
from config.logger import logger
from config.config import config
from elasticsearch import Elasticsearch, helpers

# 코사인 유사도를 구하는 함수
def cos_sim(a, b):
    return dot(a, b)/(norm(a)*norm(b))

# 스크립트를 실행하려면 여백의 녹색 버튼을 누릅니다.
if __name__ == '__main__':
    print('############### VOC Vector Search Start ###############')

    # Embadding 모델 셋팅
    # tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")  # English and Korean BERT (대소문자 구분)
    # model = AutoModel.from_pretrained("bert-base-multilingual-cased")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-uncased")  # English and Korean BERT (소문자로 변경 후 처리)
    model = AutoModel.from_pretrained("bert-base-multilingual-uncased")

    #t1 = "tax 시스템 구축 관련 미팅"
    #t1 = "DF, ECF-1 공정 Gas Detector 품의서"
    #t1 = "용접봉 관리 미흡"
    #t1 = "G80, GV60, 코나, 포터 등의 전기차 모델"
    t1="효성 화학 신제품 찾아줘"
    #t1="为汕头地区规模最大的织带厂"

    # 문장 임베딩
    te1 = model(tokenizer.encode(t1, return_tensors='pt'))[0].squeeze(0).mean(dim=0).detach().numpy()
    #te2 = model(tokenizer.encode(t2, return_tensors='pt'))[0].squeeze(0).mean(dim=0).detach().numpy()
    #te3 = model(tokenizer.encode(t3, return_tensors='pt'))[0].squeeze(0).mean(dim=0).detach().numpy()
    print(te1)

    es = Elasticsearch(hosts=config.es_host, http_auth=(config.es_user, config.es_password))

    res = es.search(index='em-voc-index2', body={
        "_source": ["vocid", "vocid_seq", "voc_text"],
        "query": {
            "script_score": {
                "query": {
                    "match_all": {}
                },
                "script": {
                    "source": "cosineSimilarity(params.query_vector, 'voc_vector') + 1.0",
                    "params": {
                        "query_vector": te1,
                    }
                }
            }
        }
    })
    # doc['voc_vector']
    # print(res['hits']['hits'])
    print("t1 => " + t1)
    for raw in res['hits']['hits']:
        print(str(raw["_score"]).ljust(10," ") + "- " + str(raw["_source"]["vocid"]).ljust(8," ") + "- " + raw["_source"]["voc_text"])
    # print(cos_sim(te1, te2))
    # print(cos_sim(te1, te3))
    # print(cos_sim(te2, te3))

