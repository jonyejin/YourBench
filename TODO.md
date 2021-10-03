# 할일
1. 모델 불러오는 파트 함수들 정리하기
2. 실제 Adversarial Attack 모델, TestSet 가져와보기
  * CleverHans의 코드는 실제 모델이 아님... 예제를 넣으면 학습된 모델을 돌려주는것!
  * https://github.com/locuslab/fast_adversarial
  * 위 레포지토리 참고해서 만들게 될듯?
  
  * ImageNet 데이터 크기가 너무 큼...
  * 토렌트로 받기? 그냥 다른 데이터셋 쓰는 모델을 찾는게 나을듯

  * WideResNet  
    * https://pytorch.org/hub/pytorch_vision_wide_resnet/
    * WRN 뒤의 숫자의 의미  
      * ex) WRN-50-2에서 50은 깊이, 2는 넓이 인자 k를 의미.
      * 참조) https://deep-learning-study.tistory.com/519
    * TODO: 9강 듣기
  * 그 외.. 설치가 복잡해서 못돌려봄
3. 모델에다가 Attack해보기!
4. Attack 여러개를 병렬적으로 구성하기
5. 점수 확인하기
6. Advise작성
7. 파일화

+) 임의의 모델? 로드??

torch.jit.script로 도전가능해보인다..

예진: 커스텀, 논문에서 사용한 모델들 로드 하고 원후오빠의 공격과 합치기