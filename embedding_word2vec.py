import re
from lxml import etree  # 파서
from nltk.tokenize import word_tokenize, sent_tokenize
from gensim.models import Word2Vec

if __name__ == '__main__':
    print("")
    print("============>> embedding_word2vec.py start")
    result="■ 부평4구역 설계변경 협의1.현안   - 심의시 필로티 제연설비유지를 위한 방풍구조 요청에 따른 방화문 적용2. 문제제기  - 제연설비유지를 위한 방화문은 과한 스펙으로 법적으로 일반 도어 적용가능3. 설계사 의견  - 소방서 문제없음 확인 시 설계변경 가능4. 진행사항 및 향후 일정  - 감리 협의 진행 → 스펙하향에 대한 문제 제기로 본사 설계팀 지원 예정  - 소방서 협의 예정(현장)[공사관련]현장 원가 개선을 위한 추가 설계개선 사항 검토"

    data = []
    temp = []
    for j in word_tokenize(result):
        temp.append(j.lower())
    data.append(temp)

    model = Word2Vec(sentences=data, window=5, min_count=1, sg=0)
    print (model)