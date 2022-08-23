#gpt-neo tutorial: https://www.youtube.com/watch?v=GzHJ3NUVtV4
#example debate topic: https://www.academia.edu/36433939/ENGLISH_DEBATE_SCRIPT_The_Impact_of_Social_Networking_for_Students_

# 사용자가 입력한 완성된 글(논리적이거나 아니거나)
# AI 는 사용자가 입력한 문장들을 개별 문장으로 나누고, 각 문장을 입력으로 새로운 문장을 자체 생성한다. 생성한 문장은 논리적인 문장일 것이라는 가설.(테스트 완료)
# 사용자가 입력한 문장들을 개별 분장으로 분해해서 리스트에 담는다. 
# 문서유사도를 비교하는 함수를 작동시킨다. (AI가 생성한 문장 1개  VS. 인간입력문장중 1개)
# 문서유사도 결과는 확률값으로 나온다. 예) 0.7 —> 70% 유사하다. 
# 정교하게 만들기 위해서 AI 10개를 돌리고, 합산 후 비교평균값을 산출하여 개별문장 점수를 얻어내고
# 추가해야할 2가지 부분: 1) 완료 - 통계값나오는데 여기서 최대값, 최소값은 제거, 2)(해야할 과제) 입력문장의 양(수) 입력문장은 512token 512단어까지는 가능, 생성문장장을 10token 이지만, 완결된 하나의 문장이 필요, gpt 그것보다 더 많은 문장을 생성
# 출력문장에서 1개의 문장만 가져온다. 완결된 문장이어야 함. 이 결과를 가지고 (AI가 생성한 문장 1개  VS. 인간입력문장중 1개)
# 모든 인간 입력 문장에 대한 비교점수를 합산 후 평균을 내면, 결과점수는 바로 논리적인지 아닌지 알 수 있는 수치로 출력된다. 
# 지금까지는 두 문장(입력, 출력 - 인간입력 1, 입력 2)비교하였고, 이 기능이 잘 작동하면,
# 전체문장을 계산(개별분석 —> 전체분석(통계적 접근))

# 가설-검증, 발생한 문제 - 해결방안(가설) - 결과…
# 글의 논리성을 분석화는데 가능한 방법, 실험설계, 테스트, 다양한 문제(예상했거나 발견된 문제들)을 어떻게 해결해 갔고, 최종적으로 어떻게 각각의 가설을 적합하게 만드는가에 대한 연구가 논문이 될 것임

# 문법교정, 완결된 문장을 확인하고 처리하는 기술이 중요함.

#https://www.assemblyai.com/docs/core-transcription#filler-words
#https://www.assemblyai.com/docs/core-transcription#profanity-filtering

# 문장 예제 입력(인간 작성 문장)
#input_sent = """You need flour to bake bread. You have a sack of flour in the garage. When you get there, you find on top of it a hat that you thought you had lost months ago. So you """
input_sent = """We choose to be pro with this topic because nowadays social networking is very popular.There are many positive and negative impacts from this site in the internet. After discussing thistopic, we hope that we can increase the positive side and decrease the bad effect. All we know that social network has very much positive impact. In the educational sector, there are
some benefits, such as for the learning discussion media. Many students sometimes feel like they’re
not able to solve their task. So, they can solve their problem by discussions in social network withtheir friends or another people who could help."""


# 다음 문장 입력(인간입력) 논리 - 이 문장을  AI와 비교할 것임, 이 문장은 논리적인 추론 결과로 작성된 문장임. 
#input_sent_next = """So you take the hat and put it on your head. You go back to the kitchen and find that you have forgotten to buy eggs."""

input_sent_next = """it doesn't work."""

# 다음 문장 입력(인간입력) 비논리 - 이 문장을  AI와 비교할 것임, 이 문장은 비논리적인 추론 결과로 작성된 문장임. 
#input_sent_next ="""have to dry it out. To do that, you spread it out on a tarp in the sun."""
# 해석 : Flour that has gotten soaked has to be thrown out; drying it will not help.

# 문장을 sentence token으로 분리하여 리스트에 저장
import nltk
from nltk.tokenize import sent_tokenize
sentences  = sent_tokenize(input_sent)
#print(sentences)

# GPT 논리적 문장 생성
import requests
import json
import re
from happytransformer import HappyGeneration
from happytransformer import GENSettings

happy_gen = HappyGeneration("GPT-NEO", "EleutherAI/gpt-neo-2.7B")

def GenText(input_txt):
    # 문장생성시 냉정함 적용 - 즉, 논리적으로 창의적이 아님!
    tmp = [0.01]

    for temp in tmp:
        top_k_sampling_settings = GENSettings(do_sample=True, 
                                                top_k=120, 
                                                temperature=temp,  
                                                max_length=10, 
                                                no_repeat_ngram_size=2)

        result_top_k_sampling = happy_gen.generate_text(input_txt, args=top_k_sampling_settings)

        #전처리 필요함 GenerationResult(text=' 
        result_top_k_sampling_re = result_top_k_sampling.text
        result_top_k_sampling_re.strip("'")
        result_top_k_sampling_re.lstrip("GenerationResult(text=' ")
        result_top_k_sampling_re = re.sub(r"\n", "", result_top_k_sampling_re)

    return result_top_k_sampling_re


# Gramma-correction
from happytransformer import  HappyTextToText
grammaCorrection = HappyTextToText("T5", "vennify/t5-base-grammar-correction")
from happytransformer import TTSettings

beam_settings =  TTSettings(num_beams=5, min_length=1, max_length=100)

def grmmaCorrection(input_sentence):
    output_text= grammaCorrection.generate_text(input_sentence, args=beam_settings)
    return output_text


def listToString(str_list):
    result = ""
    for s in str_list:
        result += s + " "
    return result.strip()


# 6개의 샘플 문장 생성
def GenText_2nd(input_txt):
    tmp = [0.001, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] # 몇개 더 추가함
    resp = []
    resp_ = []
    resp__ = []
    for temp in tmp:
        top_k_sampling_settings = GENSettings(do_sample=True, 
                                                top_k=200, 
                                                temperature=temp,  
                                                max_length=50,
                                                min_length=5,
                                                no_repeat_ngram_size=2)

        result_top_k_sampling = happy_gen.generate_text(input_txt, args=top_k_sampling_settings)

        #전처리 필요함 GenerationResult(text=' 
        result_top_k_sampling_re = result_top_k_sampling.text
        result_top_k_sampling_re.strip("'")
        result_top_k_sampling_re.lstrip("GenerationResult(text=' ")
        result_top_k_sampling_re = re.sub(r"\n", "", result_top_k_sampling_re)


        
        
        # 생성된 글을 sentence tokenize 
        sent_token_re = sent_tokenize(result_top_k_sampling_re)
        


        # #----------
        # 마지막 문장 가져오기
        inp_sent = str(sent_token_re[-1])
        # 완결된 문장으로 수정하기
        fixed_sent = grmmaCorrection(inp_sent)
        
        #전처리 필요함 TextToTextResult(text=' 
        fixed_sentg_re = fixed_sent.text
        fixed_sentg_re.strip("'")
        fixed_sentg_re.lstrip("TextToTextResult(text=' ")
        fixed_sentg_re = re.sub(r"\n", "", result_top_k_sampling_re)
        
        
        
        fixed_sent_re_ = str(fixed_sentg_re)
        
        # 마지막 요소 제거 후
        sent_token_re.pop()
        # 완결문장 추가
        sent_token_re.append(fixed_sent_re_)
        #----------------
        
        for i in sent_token_re:
            resp_.append(i)
            
        result = listToString(resp_)
        print("Gen sent AI :", result)
        resp__.append(result)
        
    return resp__





# 문장 생성한 결과 
gen_txt  = GenText(input_sent)

# 인간입력문장 vs. AI 생성 문장 유사도 비교

sen = [input_sent_next, gen_txt]


from sentence_transformers import SentenceTransformer
model = SentenceTransformer('bert-base-nli-mean-tokens')
#Encoding:
sen_embeddings = model.encode(sen)

from sklearn.metrics.pairwise import cosine_similarity
#let's calculate cosine similarity for sentence 0:
sim_re = cosine_similarity(
    [sen_embeddings[0]],
    sen_embeddings[1:]
)

# 1개의 AI로 비교한 문장 유사도 측정 결과 input_sent_A. vs. input_sent_B
print("AI resoning result : ", sim_re[0][0])


# 6개의 AI로 비교한 문장 유사도 측정결과로 이것이 논리적 분석 값임
gen_txt_2nd = GenText_2nd(input_sent)

# 생성된 문장 확인
print("생성된 문장 확인 :" , gen_txt_2nd)

# 생성된 문장 개별 비교
sim_re_li = []
for itm in gen_txt_2nd:
    sent_ = [input_sent_next, itm]
    sen_embeddings_ = model.encode(sen)

    sim_re = cosine_similarity(
        [sen_embeddings_[0]],
        sen_embeddings_[1:]
    )
    sim_re_li.append(sim_re[0][0])

# 비교값 확인
print("유사문장 비교값 확인: " , sim_re_li)

# 보정을 위해서 최대, 최소값 삭제
min_value = min(sim_re_li)
max_value = max(sim_re_li)

sim_re_li.remove(min_value)
sim_re_li.remove(max_value)

print("최대 최소값 삭제 확인: " , sim_re_li)

# 개별 비교 분석 결과 평균값 산출하기
import numpy as np
a = np.array(sim_re_li)
AVG = np.mean(a)

print(" AI resoning result score :", AVG)