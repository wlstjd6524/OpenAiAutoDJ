## 👨‍🏫 프로젝트 소개
OpenAI(ChatGPT) 를 통한 나만의 자동화 음악 큐레이션 시스템

## 💻 개발환경 
- ChatGPT API Model_gpt-3.5-turbo
- Python 3.11.1
- Stable Diffusion Model_dreamlike-diffusion-1.0
- Dalle2 API
- NVIDIA CUDA Toolkit 11.2
- Audio & Video Tool_ffmpeg
- yt-dlp ver 25.03.27
- Youtube-search Module 2.1.2
- Moviepy 2.1.2

## 📌 프로젝트개요
이 프로젝트는 OpenAI의 ChatGPT API를 활용하여 사용자의 취향에 맞는 음악을 자동으로 추천하고, 해당 곡을 YouTube에서 검색 및 다운로드하여 나만의 플레이리스트를 완성하는 자동화 음악 큐레이션 시스템입니다.

프로젝트의 출발점은 생성형 AI 프로그래밍을 보다 쉽게 체험할 수 있는 실용적인 아이디어를 찾는 것이었습니다. 최근 SNS와 유튜브 숏폼 등에서는 특정 장르나 상황에 어울리는 노래를 추천하는 콘텐츠가 인기를 끌고 있으며, 주차별 인기곡이나 특정 아티스트의 대표곡을 감성적인 이미지와 함께 보여주는 영상도 자주 접할 수 있습니다.

또한, 사람들은 자주 듣는 음악에 국한되지 않고 누군가의 추천을 통해 새로운 음악을 발견하는 경험을 점점 더 선호하고 있습니다. 실제로 유튜브 뮤직에서도 이런 시스템이 긍정적인 평가를 받고 있으며, 이러한 흐름을 생성형 AI에 접목시키면 재미있고 유용한 자동화 시스템이 되겠다는 생각에서 출발하게 되었습니다.

사용자는 단순히 자신의 기분이나 음악 취향을 입력하기만 하면, ChatGPT가 적절한 곡을 추천하고, 각 곡을 유튜브에서 검색하여 고품질 MP3로 자동 저장합니다. Stable Diffusion을 연동하여, 생성된 플레이리스트의 분위기에 맞는 커버 아트 이미지까지 생성할 수 있도록 구성되어 있습니다.

복잡한 수작업 없이, 텍스트 기반의 대화만으로 개인화된 음악 리스트를 만들 수 있는 점이 이 프로젝트의 가장 큰 장점입니다.

## ✒️ 프로젝트 목표
사용자는 단순히 명령어를 입력하는 것만으로도 아래와 같은 작업을 손쉽게 수행할 수 있습니다:

- 유튜브에서 음악을 검색 및 다운로드하고, MP3로 저장
- 곡별로 AI 이미지 생성(DALL·E 2, Dreamlike Diffusion 등 활용)
- 생성된 이미지들을 바탕으로 영상 제작
- 기존 MP4 기반의 영상 합성
- MP3 앨범 아트에 곡 정보 삽입
- HTML 기반의 콘텐츠 웹 UI 자동 생성

이를 통해 음악 감상 경험을 풍부하게 확장하고, 창작자가 직접 편집하지 않아도 AI가 콘텐츠를 구성해주는 자동화된 미디어 생성 흐름을 구현하는 것이 주요 목표입니다.

## 🔨 프로젝트 아키텍처
![Image](https://github.com/user-attachments/assets/50c3a5f3-4b00-43b9-843d-d0492bcb2d7b)

## 📱 기능 설명
- 생성형 AI의 GPT API 환경을 코드값으로 설정한 Main 화면 <br> <br>
![Image](https://github.com/user-attachments/assets/cf4173d0-64e2-4ae2-8535-2a238bae0650)

- 생성형 AI와 설정된 Prompt 값을 토대로 대화를 하며 자동화 큐레이션 시스템을 이용 <br> <br>
![Image](https://github.com/user-attachments/assets/55fbde24-617d-4e49-8d31-81d1c3938438)

- MP3 파일 경로가 존재하고 이미지 파일이 존재하였을 때 해당값들을 통해 통합된 영상을 제작 <br> <br>
![Image](https://github.com/user-attachments/assets/ac992028-12bb-4cc4-bee8-5d24ea50284b)

- dreamlike-diffusion 을 통해 만들어진 AI 자동화 이미지 모습들 <br> <br>
![Image](https://github.com/user-attachments/assets/066cec6f-3fd5-49f2-92d8-f7726d28effa)

- 해당 GPT API 환경을 통해 통합적으로 완성되었을 때 출력되는 HTML 환경 <br> <br>
![Image](https://github.com/user-attachments/assets/448a6a7c-18b0-4f89-8cb1-467d0b872bba)
