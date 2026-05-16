# Assets — 데모 입력 샘플 이미지

[HF Space 데모](https://huggingface.co/spaces/AD-Styles/mini-llava-v4-demo) 에
직접 업로드해 v4 모델을 바로 시험해 볼 수 있는 샘플 이미지입니다.

| 파일 | 설명 | 용도 |
|------|------|------|
| `source_dog.jpg` | 헬로키티 모자를 쓴 강아지 | 일반 객체 VQA (영어 / 한국어 질문) |
| `source_pikachu.png` | 모자 쓴 피카츄 | OOD 입력 — 학습 분포 밖 만화 캐릭터 |

정적 응답 스크린샷은 싣지 않습니다. 위 이미지를 데모에 올리면 누구나 같은 입력을
재현할 수 있고, 배포 모델의 실제 응답 점검은 `scripts/smoke_space_demo.py`
(→ `eval_results/v4_space_smoke.json`) 로 기록했습니다.
