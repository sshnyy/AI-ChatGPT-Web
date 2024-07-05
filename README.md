# 💻 AI-ChatGPT-Web

### 피부 병변 조기 진단을 위한 이미지 분류(AI)와 ChatGPT 기반 웹 시스템

- 한국정보기술학회 KIIT2023 프로젝트
- 사용 언어: `Python`, `Java`, `JavaScript`, `MySQL`
- 배포 환경 : `Ubuntu 18.04.6`
  
<br>

## Project

    Date : 2023.03 ~ 2023.11 (7개월)
    Team: 3명
    
    My Roles:
    - 프로젝트 보조
    - 데이터 수집 및 AI 모델 구축
    - 피부 병변 조기 진단을 위한 API 연동

<br>

## Summary
  
- 제안하는 웹 시스템은 사용자가 피부 병변 이미지를 업로드하면, 해당 이미지를 분석하여 피부암 진단 정보를 제공하고, ChatGPT를 통해 추가적인 질문에 답변하며 즉각적인 답변을 제공합니다.


<br>

<img width="1551" alt="image" src="https://github.com/sshnyy/AI-ChatGPT-Web/assets/99328827/77c147b6-30b3-4bad-a1ba-1f4b79711bd7">


## 1️⃣ AI

![image](https://github.com/sohyunyg/Completed_Projects/assets/99328827/da7d27a5-0aee-47dc-b637-41770d9e3923)
![image](https://github.com/sohyunyg/Completed_Projects/assets/99328827/3589b0eb-0632-4f84-8e81-ff977c47b49e)
![image](https://github.com/sohyunyg/Completed_Projects/assets/99328827/7d57690d-9cf8-4f15-804a-352186d57106)
![image](https://github.com/sohyunyg/Completed_Projects/assets/99328827/e18ec188-0c4d-4b23-893c-8ef8cdd70575)


<br>

## 2️⃣ Web
![image](https://github.com/sohyunyg/Completed_Projects/assets/99328827/d2e1a10b-82cd-454f-80a7-ea66124122ab)
![image](https://github.com/sohyunyg/Completed_Projects/assets/99328827/b457f25d-2474-4bc2-ab2a-ada5bde54702)
![image](https://github.com/sohyunyg/Completed_Projects/assets/99328827/7b7dffda-11ed-4c11-8f6b-4d10c62a0443)
![image](https://github.com/sohyunyg/Completed_Projects/assets/99328827/ee4e232b-3018-4a7b-b20e-07fac1672490)


<br>

## How to run Python File

#### Train
```
git clone https://github.com/Kangsuyeon01/DermQA.git
CD DermQA_project/DL
```

```
python train.py
```
학습이 완료된 모델은 'models/saved_model' 에 저장됩니다.
#### Inference (test dataset)
```
python pipeline.py
```
#### Run Web application (Socket communication between Java Spring and Python)
```
python server.py --model_saved_path=[trained model path] --OPENAI_API_KEY=[OPENAI_API_KEY]
```
* java Spring Project 실행의 경우 `DermQA_project/java/project/src/main/resources
/application.properties`에서 MySQL 데이터 베이스 연결 후 사용
--- 
