import torch
from transformers import BertTokenizer, TFBertModel, AutoTokenizer, AutoModel
import ssl
import requests
import os
from numpy.linalg import norm
from numpy import dot
from db_fn import db_config
from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tiktoken
import etl_voc_insert
from datetime import datetime

# 코사인 유사도를 구하는 함수
def cos_sim(a, b):
    return dot(a, b)/(norm(a)*norm(b))

# 스크립트를 실행하려면 여백의 녹색 버튼을 누릅니다.
if __name__ == '__main__':
    print('')
    print('############### VOC Vector Insert Start ###############')
    print(f'{datetime.now()}:Start')

    # Embadding 모델 셋팅   em-voc-bertmultilingual
    tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-uncased")  # English and Korean BERT
    model = AutoModel.from_pretrained("bert-base-multilingual-uncased")

    # Embadding 모델 셋팅   em-voc-krbert
    tokenizer = AutoTokenizer.from_pretrained("snunlp/KR-BERT-char16424")
    model = AutoModel.from_pretrained("snunlp/KR-BERT-char16424")

    # VOC 데이터 검색
    voc_list = db_config.select_voc()
    print("voc_list count : " + str(len(voc_list)))
    j=0
    for x in voc_list:
        vocid = x[0]
        voc = x[1].replace("'","`")

        # 문장을 일정 길이로 자르기
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
        texts = text_splitter.split_text(voc)
        j=j+1
        print("VOC Length : " + str(len(texts)) + "  Count : " + str(j))
        i = 0
        for y in texts:
            input_ids = tokenizer.encode(y, return_tensors='pt')
            outputs = model(input_ids)
            sentence_embedding = outputs[0].squeeze(0).mean(dim=0)
            i = i + 1
            result = db_config.insert_voc_vector(str(vocid), str(i), y, sentence_embedding.detach().numpy(),  'em-voc-bertmultilingual')
            etl_voc_insert.insertEsData(str(vocid)+"-"+str(i), vocid, i, y,  sentence_embedding.detach().numpy(), 'em-voc-bertmultilingual')

        result = db_config.update_voc(str(vocid), "S")

    print('############### VOC Vector Insert End ###############')
    print(f'{datetime.now()}:End')