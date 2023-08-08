from config.logger import logger
from config.config import config
from elasticsearch import Elasticsearch, helpers
from db_fn import db_config

def insertEsData(es_id, vocid, vocid_seq, voc_text, voc_vector, index_id):
    try:
        es = Elasticsearch(hosts=config.es_host, http_auth=(config.es_user, config.es_password))
        docs = []
        docs.append({
            '_index': index_id,
            '_type': '_doc',
            "_id": es_id,
            '_source': {
                "vocid_seq": int(vocid_seq),
                "vocid": int(vocid),
                "voc_text" : voc_text,
                "voc_vector" : voc_vector
            }
        })

        #"voc_vector": voc_vector.replace('[', '').replace(']', '').split(',')
        helpers.bulk(es, docs)
    except Exception as e:
        logger.error("Index: " + " em-voc-index2" + ", _id:" + "IIIII" + " : update db err ", e)

    return es

# 스크립트를 실행하려면 여백의 녹색 버튼을 누릅니다.
if __name__ == '__main__':

    print('############### VOC Vector Elasticsearch Insert Start ###############')
    # VOC 데이터 검색
    voc_list = db_config.select_voc_vector()
    print("voc_list count : " + str(len(voc_list)))
    for x in voc_list:
        insertEsData(x[0], x[1], x[2], x[3], x[4])

