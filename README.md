# ADS
Automated Debate Scoring 

어떻게 하면 인공지능이 토론을 심사할 수 있을까? - 자연어 처리를 활용한 토론 심사 모델의 개발


gpt NEO
10개 정도의 인공지능 토론자를 만들어서 토론 대본을 생성
토론 토픽, 형식을 고려해서 시작하는 문장 리스트를 만들고 랜덤으로 모델에 적용
gpt NEO가 생성한 토론 대본과 실제 토론자의 대본의 문서 유사도를 측정하여 수치화
10개의 문서 유사도를 평균 내서 토론 점수 생성
추가로 인신공격 발언, filler words에 대한 평가 지표도 활용
gpt reasoning: https://cs.nyu.edu/~davise/papers/GPT3CompleteTests.html
문서 유사도: sentence similarity(gensim 등 여러 모델들을 모두 사용해보고 성능 비교)
논문: 모델들 모두 사용해보고 성능 비교, 인간 심사위원의 낸 토론 점수와 비교
깃헙에 포트폴리오로 정리(대학)
토론 동아리부원에게 테스트
인간의 노력과 시간, 비용이 들지 않고 데이터가 없어도 스스로 만들어서 할 수 있다는 점
문서 생성을 스핀오프해서 토론 모델
codeex -> 언급: 논리적인 NLP. codeex를 통해 토론에 활용할 수 있지 않을까 하는 아이디어 (논문 읽어보기)
flask로 웹에다가 grammarly처럼 만들어서 포트폴리오 영상으로도 활용 가능

zero shot learning
