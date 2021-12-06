## 개요

---

- Objection detection분야에서 최초로 딥러닝을 적용시킨 RCNN 모델에 대해 정리
- Fast-RCNN, Faster-RCNN, Mask-RCNN과 같이 RCNN 기반 모델에 대해 정리

![https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2F3I4GG%2Fbtqt1YgvsBE%2FffE10j1X4V6i7kWNWY8EAk%2Fimg.png](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2F3I4GG%2Fbtqt1YgvsBE%2FffE10j1X4V6i7kWNWY8EAk%2Fimg.png)

## 모델 설명

---

### 1. RCNN

- 배경
    - **[Rich feature hierarchies for accurate object detection and semantic segmentation](https://arxiv.org/pdf/1311.2524.pdf)**
    - 2014년 CVPR에 게재
    - Object detection 분야에서 최초로 딥러닝(CNN)을 적용
    - 이후에 등장할 다양한 object detection 모델의 기반이 됨
- 모델 구조
    
    ![https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FckQHkt%2FbtqudtS7Kef%2FOciaJlorLTKztFSRXFFDe1%2Fimg.png](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FckQHkt%2FbtqudtS7Kef%2FOciaJlorLTKztFSRXFFDe1%2Fimg.png)
    
    ![https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2Fl2uCa%2FbtquhiLhyUO%2FVkErzTU5MeibSSX7WE2rJk%2Fimg.png](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2Fl2uCa%2FbtquhiLhyUO%2FVkErzTU5MeibSSX7WE2rJk%2Fimg.png)
    
    - Object detection 문제를 1) 이미지에서 물체의 영역 탐지, 2) 영역의 물체를 분류 하는 2-stage 문제로 정의
    - 이미지를 입력으로 넣음
    - Object가 될 수 있는 region 후보들을 추출 (~2천 개)
    - 각각 다른 region의 크기를 분류 모델의 입력에 적합한 사이즈로 resizing
    - 분류 결과를 통해 labeling
    - Object 위치 탐색 → 분류 작업을 3가지 모듈로 분할
    
    1. Region Proposal (영역 제안)
        
        [https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FbPSomg%2FbtqugDuIClm%2F85f12kCkHxLUFMu8aI6fA0%2Fimg.png](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FbPSomg%2FbtqugDuIClm%2F85f12kCkHxLUFMu8aI6fA0%2Fimg.png)
        
        - Segmentation에서 많이 사용되는 selective search 알고리즘 사용
        - 위의 그림과 같이 색감(Color), 질감(Texture) 또는 에워싸임(Enclosed) 여부를 판단해 영역을 나눔
    2. CNN
        - 나뉘어진 2천개 이하의 영역들에 대해 Object 개수 만큼의 classification 문제로 만듦
            
            ![https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FmkwJ5%2Fbtqud6YY7mU%2FoabGpxlM0PJkoRZXoNndDK%2Fimg.png](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FmkwJ5%2Fbtqud6YY7mU%2FoabGpxlM0PJkoRZXoNndDK%2Fimg.png)
            
        - AlexNet의 구조를 사용했으며 class 개수만 조정하여 fine-tuning
    3. SVM
        - Classifier로 softmax를 쓰지 않고 SVM을 사용
        - 논문에서 실험적으로 softmax보다 SVM을 사용했을 때 성능이 높았다고 함
    4. Bounding Box Regression
        - Selective search를 통해 찾은 Object 박스의 위치는 상당히 부정확
        - 박스 위치를 조정하기 위한 과정
        - 탐색한 박스는 다음과 같이 표현 (x, y는 이미지의 중심점, w,h는 이미지의 너비 높이) → $P^i = (P^i_x, P^i_y, P^i_w, P^i_h)$
        - 실제 박스는 다음과 같이 표현 → $G=(G_x, G_y, G_w, G_h)$
        - P를 G와 최대한 가깝게 만드는 함수 d를 탐색 → $d_x(P), d_y(P), d_w(P), and\ d_h(P)$
        - 결과적으로 수정된 박스 $\hat{G} = (\hat{G}_x, \hat{G}_y, \hat{G}_w, \hat{G}_h)$는 다음과 같음
            
            ![https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FmbMJD%2Fbtquj1DepoF%2FPgEpgFpHpcWe8hwjmE3bu1%2Fimg.png](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FmbMJD%2Fbtquj1DepoF%2FPgEpgFpHpcWe8hwjmE3bu1%2Fimg.png)
            
        - (5)의 t는 P를 G로 이동시키기 위해 필요한 이동량을 의미함
        - $\phi_5$는 pool5 layer에서 얻어낸 특징 벡터를 의미함

- 장점
    - 이전 모델들에 비해 성능이 뛰어남
- 단점
    - 복잡하고 속도가 느림
    - Multi-stage 로 구성되어 있기 때문에 SVM, Bounding box regression에서 학습한 결과가 CNN을 업데이트 시키지 못함 (Back propagation이 불가)
    

### 2. Fast RCNN

- 배경
    - AlexNet을 사용하기 위해 이미지 크기를 변형 → 정보 손실이 존재
    - Selective search를 통해 뽑힌 후보들을 모두 분류하기 때문에 시간이 오래 걸림
    - Back propagation 불가
- 모델 구조
    1. CNN
        
        ![https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FGKiLC%2FbtqBuam3Ms2%2FeAAVlITAfKpLXA3QqLx2k1%2Fimg.png](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FGKiLC%2FbtqBuam3Ms2%2FeAAVlITAfKpLXA3QqLx2k1%2Fimg.png)
        
        ![https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FdeNqIx%2FbtqBuiLWi0l%2F2k75SqyKHLM5KsDe7qK641%2Fimg.png](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FdeNqIx%2FbtqBuiLWi0l%2F2k75SqyKHLM5KsDe7qK641%2Fimg.png)
        
        - Selective search를 통해 Region Proposal 추출
        - 뽑아낸 영역을 crop하지 않고 좌표와 크기 정보만 메모리에 저장 (RoI)
        - 전체 이미지를 CNN에 입력으로 집어넣음
        - 뽑아낸 RoI를 feature map에 맞게 위치 조정 (RoI projection)
    2. RoI (Region of Interest) Pooling
        
        ![https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FdOsmXy%2FbtqBxAkx2i0%2FIU2QyFPnANRKtPbkaXVQMk%2Fimg.png](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FdOsmXy%2FbtqBxAkx2i0%2FIU2QyFPnANRKtPbkaXVQMk%2Fimg.png)
        
        - Projection 시킨 RoI들을 fc layer에 입력으로 사용하기 위해서 공통된 크기를 갖는 feature map이 필요
        - 각각 다른 RoI 영역의 resolution을 맞추는 것이 RoI pooling
        - 크기가 다른 feature map의 region마다 stride를 다르게 pooling하여 크기를 맞춤
            
            ![https://blog.kakaocdn.net/dn/Izuvy/btqBuxoxz8c/1FxESvfKPLwFIdFrSHkNfk/img.gif](https://blog.kakaocdn.net/dn/Izuvy/btqBuxoxz8c/1FxESvfKPLwFIdFrSHkNfk/img.gif)
            
        - 7x5 크기의 region을 2x2로 만들기 위해 각각 다른 stride (7/2 = 3, 5/2 = 2)로 pooling하여 2x2의 output을 만듦
    3. Classification & Bounding Box Regression
        
        ![https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2Fb9fevh%2FbtqBublVOAw%2FceNpRDlR2Pj72qJKRRODZK%2Fimg.png](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2Fb9fevh%2FbtqBublVOAw%2FceNpRDlR2Pj72qJKRRODZK%2Fimg.png)
        
        - RoI pooling을 통해 구해진 fixed length feature vector를 fc layer에 입력
        - 출력된 값을 사용해 classification과 bouding box regression을 진행
        - 여기서는 softmax를 사용해 분류함
        - classification과 bbox regressor를 동시에 학습
        
        ![https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2Fb9p2tE%2FbtqBvudPTO1%2FMkBUI6r4hzeplAUgiq9GNk%2Fimg.png](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2Fb9p2tE%2FbtqBvudPTO1%2FMkBUI6r4hzeplAUgiq9GNk%2Fimg.png)
        

- 장점
    - RCNN에 비해 성능적 향상을 보임
    - 빠른 속도
    - End-to-End training으로 back propagation 가능
- 단점
    - Selection search의 경우 CPU를 사용하기 때문에 이 부분에서 속도가 느림
    

### 3. Faster RCNN

- 개요
    
    ![https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FcMqq0O%2FbtqBFNcS3vP%2FTkehwBGKq51Tx7SwLHkt7k%2Fimg.png](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FcMqq0O%2FbtqBFNcS3vP%2FTkehwBGKq51Tx7SwLHkt7k%2Fimg.png)
    
    - Region proposal 방법을 GPU를 통한 학습으로 진행하는 것을 목적으로 함
    - 빨간색으로 표시된 Region proposal network를 제외하면 Fast RCNN과 동일함
- 모델 구조
    1. RPN (Region Proposal Network)
        
        ![https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FbfDWks%2FbtqBFwJfwth%2F0VqH45QYD5vFL8mqrscBUk%2Fimg.png](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FbfDWks%2FbtqBFwJfwth%2F0VqH45QYD5vFL8mqrscBUk%2Fimg.png)
        
        - CNN을 통해 나온 feature map을 k개의 anchor box를 통해 영역을 정함
        - classification layer와 bbox regression을 거쳐 물체가 있는 곳을 학습
        - classification layer는 물체가 있는 지 없는 지만 확인하므로 2개의 class를 갖게 됨
        
        1. Anchor Targeting
            - CNN을 통해 생성된 feature map의 각 픽셀마다 k=9개의 anchor box를 생성
            - Anchor box는 3개의 scale(8, 16, 32)과 3개의 ratio(0.5, 1, 2)를 통해 9개를 생성
            - Input size가 800x800인 VGG-16을 예로 들면 feature map의 사이즈는 50x50x512 이므로, subsampling ratio는 16(800/50)임
            - 16x16 안의 중심 픽셀을 중심으로 50x50x9개(=22500개)의 anchor box가 생성됨
            
            ![https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FbOhhr4%2FbtqBE0RkFzd%2Fz1d20KVRX9YomD9AR38KSk%2Fimg.png](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FbOhhr4%2FbtqBE0RkFzd%2Fz1d20KVRX9YomD9AR38KSk%2Fimg.png)
            
            - 예시와 같이 800x800 크기의 이미지의 (400, 400) 위치에서 생성된 anchor box
            - 생성된 anchor box를 기준으로 그 안에 물체가 존재하는 지 여부를 학습할 예정
            - 실제 box와 비교했을 때 IoU가 0.7보다 크면 positive, 0.3보다 작으면 negative로 두고 나머지는 -1로 설정
            - 단, positive한 anchor box가 적을 수 있기 때문에 IoU가 가장 높은 anchor box 1개를 뽑아 positive로 라벨링
        2. Prediction
            - Bbox regression layer (50x50x512 → 50x50x9x4(좌표 4개))와 classfication layer (50x50x512 → 50x50x9x2 (존재 여부 2개 클래스))를 출력
            - 두 값은 실제 box 좌표와 Object 존재 여부로 loss를 계산하고 학습
            - 그리고 NMS를 거쳐 RoI sampling 된 후 Fast RCNN에 사용
        3. Loss Function for RPN
            
            ![https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2Fp1MNw%2FbtqBCU5Ofbo%2FYNbUFKrkf1sfO7XhR6BYM0%2Fimg.png](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2Fp1MNw%2FbtqBCU5Ofbo%2FYNbUFKrkf1sfO7XhR6BYM0%2Fimg.png)
            
    2. NMS (Non-Maximum Suppresion) & RoI Sampling
        
        ![https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FbmMJwY%2FbtqBHytwmdI%2FS4z5M3l7sXh2tKh7GkyAf0%2Fimg.png](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FbmMJwY%2FbtqBHytwmdI%2FS4z5M3l7sXh2tKh7GkyAf0%2Fimg.png)
        
        - 위 그림처럼 하나의 Object에 여러 개의 anchor box가 겹치는 것을 해결하기 위한 작업
        - prediction된 box들을 RoI score로 내림차순 정렬한 뒤, 높은 RoI score를 가진 box와 overlapping된 다른 box들을 제거 (overlapping threshold는 주로 0.6 ~ 0.9 사이를 사용) → 이것을 NMS라고 함
        - 제거된 RoI들 중 positive:negative 비율이 1:1이 되도록 sampling
        - positive의 개수가 부족하면 zero padding을 하거나 IoU 값이 높은 box를 positive로 사용
    3. Fast RCNN
        - 이후는 Fast RCNN과 동일
        
---

출처

- [https://nuggy875.tistory.com/21?category=860935](https://nuggy875.tistory.com/21?category=860935)
- [https://nuggy875.tistory.com/33](https://nuggy875.tistory.com/33)
- [https://tutorials.pytorch.kr/intermediate/torchvision_tutorial.html](https://tutorials.pytorch.kr/intermediate/torchvision_tutorial.html)