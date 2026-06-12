import os
import io
import re
import json
import tempfile
from typing import List, Dict, Any

import streamlit as st
from PIL import Image
from google.cloud import vision


USER_ROLE = "을"


def setup_google_credentials():
    if os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
        return

    try:
        service_account_info = dict(st.secrets["gcp_service_account"])
    except Exception:
        st.error(
            "Google Vision 인증 정보가 없습니다. "
            "코랩에서는 서비스계정 JSON을 먼저 업로드하고, "
            "배포 후에는 Streamlit Cloud Secrets에 gcp_service_account를 등록해야 합니다."
        )
        st.stop()

    key_path = os.path.join(tempfile.gettempdir(), "gcp_service_account.json")

    with open(key_path, "w", encoding="utf-8") as f:
        json.dump(service_account_info, f, ensure_ascii=False)

    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = key_path


def normalize_text(text: str) -> str:
    if text is None:
        return ""

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)

    return text.strip()


def extract_text_from_image(image_bytes: bytes) -> str:
    setup_google_credentials()

    client = vision.ImageAnnotatorClient()
    image = vision.Image(content=image_bytes)

    response = client.document_text_detection(image=image)

    if response.error.message:
        raise Exception(response.error.message)

    return response.full_text_annotation.text or ""


def detect_document_type(text: str) -> Dict[str, Any]:
    text = normalize_text(text)

    keyword_sets = {
        "근로계약서": [
            "근로계약서", "근로자", "사업주", "임금", "근로개시일",
            "소정근로시간", "휴게", "연차", "4대 사회보험", "주휴일"
        ],
        "임대차계약서": [
            "임대차계약서", "임대인", "임차인", "보증금", "차임",
            "월세", "전세", "임대목적물", "원상복구", "계약갱신"
        ],
        "프리랜서계약서": [
            "프리랜서", "위촉", "원고료", "결과물", "저작권",
            "용역비", "업무위탁", "개별 업무", "성과물"
        ],
        "용역계약서": [
            "용역계약서", "용역", "도급인", "수급인", "용역대금",
            "검수", "납품", "하자보수", "과업", "완료보고"
        ],
        "개인정보동의서": [
            "개인정보", "수집", "이용", "제3자 제공", "보유기간",
            "동의", "민감정보", "처리위탁", "파기", "이용목적"
        ],
    }

    scores = {}

    for doc_type, keywords in keyword_sets.items():
        score = 0

        for keyword in keywords:
            if keyword in text:
                score += 1

        scores[doc_type] = score

    best_type = max(scores, key=scores.get)
    best_score = scores[best_type]

    total_possible = len(keyword_sets[best_type])

    if best_score == 0:
        return {
            "document_type": "기타",
            "confidence": 0,
            "scores": scores
        }

    confidence = round((best_score / total_possible) * 100)

    return {
        "document_type": best_type,
        "confidence": confidence,
        "scores": scores
    }


def split_sentences_ko(text: str) -> List[str]:
    text = normalize_text(text)
    sentences = []

    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue

        parts = re.split(
            r"(?<=[.!?])\s+|(?<=다)\s+|(?<=함)\s+|(?<=됩니다)\s+|(?<=합니다)\s+|(?<=한다)\s+",
            line
        )

        for part in parts:
            part = part.strip(" \t-")
            if part:
                sentences.append(part)

    return sentences


def clean_evidence_text(text: str) -> str:
    text = normalize_text(text)
    text = re.sub(r"^\s*[-–—]\s*", "", text)
    return text.strip()


def is_useful_evidence(text: str) -> bool:
    text = clean_evidence_text(text)

    if not text:
        return False

    if len(text) <= 1:
        return False

    if text in {"-", "원", "일", "월", "년", "시분"}:
        return False

    if len(text.replace("\n", "").strip()) < 4:
        return False

    return True


def evidence_score(text: str) -> int:
    score = 0
    line_count = text.count("\n") + 1
    length = len(text)

    if line_count == 1:
        score += 30
    else:
        score -= line_count * 4

    if 8 <= length <= 80:
        score += 20
    elif 81 <= length <= 140:
        score += 5
    else:
        score -= 10

    if ":" in text:
        score += 5

    return score


def find_matching_sentences(sentences: List[str], patterns: List[str], full_text: str = "", max_items: int = 2) -> List[str]:
    candidates = []

    for sentence in sentences:
        sentence = clean_evidence_text(sentence)

        if not is_useful_evidence(sentence):
            continue

        for pattern in patterns:
            if re.search(pattern, sentence, flags=re.IGNORECASE | re.MULTILINE):
                candidates.append(sentence)
                break

    if full_text:
        lines = [clean_evidence_text(line) for line in full_text.split("\n")]
        lines = [line for line in lines if is_useful_evidence(line)]

        for line in lines:
            for pattern in patterns:
                if re.search(pattern, line, flags=re.IGNORECASE | re.MULTILINE):
                    candidates.append(line)
                    break

    unique = []
    seen = set()

    for item in candidates:
        item = clean_evidence_text(item)
        key = re.sub(r"\s+", "", item)

        duplicate = False

        for old in seen:
            if key in old or old in key:
                duplicate = True
                break

        if not duplicate:
            unique.append(item)
            seen.add(key)

    unique.sort(key=evidence_score, reverse=True)

    return unique[:max_items]


FAVORABLE_RULES: List[Dict[str, Any]] = [
    {
        "title": "대금 또는 임금 지급 기준이 비교적 명확함",
        "severity": 3,
        "patterns": [r"지급일", r"지급\s*방법", r"계좌", r"입금", r"정산", r"지급한다", r"지급하여야"],
        "explanation": "돈을 언제, 어떤 방식으로 지급하는지 적혀 있으면 미지급이나 지연이 생겼을 때 근거로 삼기 좋습니다.",
        "easy": "돈을 언제 어떻게 받을 수 있는지 적혀 있으면 나중에 따지기 쉽습니다."
    },
    {
        "title": "근로개시일이 명시됨",
        "severity": 2,
        "patterns": [r"근로개시일", r"시작일", r"계약\s*시작"],
        "explanation": "언제부터 근로가 시작되는지 적혀 있으면 근로 시작 시점을 확인하는 근거가 됩니다.",
        "easy": "언제부터 일하기로 했는지 확인할 수 있습니다."
    },
    {
        "title": "상대방의 의무가 문서에 적혀 있음",
        "severity": 3,
        "patterns": [r"갑은.*하여야", r"갑은.*제공", r"사업주는.*교부", r"사업주는.*하여야", r"회사.*하여야", r"사업자.*하여야"],
        "explanation": "상대방이 해야 할 일이 명확히 적혀 있으면 사용자가 이행을 요구하기 쉽습니다.",
        "easy": "상대가 해야 할 일이 적혀 있으면 지키라고 말할 근거가 생깁니다."
    },
    {
        "title": "유급휴일 기준이 언급됨",
        "severity": 2,
        "patterns": [r"공휴일.*근로기준법", r"근로자의\s*날.*유급휴일", r"대체공휴일.*포함", r"유급휴일로\s*함"],
        "explanation": "공휴일이나 근로자의 날을 유급휴일로 본다는 기준이 적혀 있으면 휴일 처리와 수당 문제를 확인할 근거가 됩니다.",
        "easy": "쉬는 날을 유급으로 인정한다는 내용이 있으면 나중에 확인할 기준이 생깁니다."
    },
]


UNFAVORABLE_RULES: List[Dict[str, Any]] = [
    {
        "title": "연차유급휴가 사용 제한 가능성",
        "severity": 5,
        "patterns": [r"연차.*회사\s*사정", r"연차.*사용하지\s*못할\s*수", r"휴가.*사용하지\s*못할\s*수", r"연차휴가는.*사용하지\s*못"],
        "explanation": "연차유급휴가는 근로자의 중요한 권리인데, 회사 사정만으로 사용하지 못하게 할 수 있다는 표현은 근로자에게 매우 불리합니다.",
        "easy": "회사가 바쁘다는 이유로 내 연차를 못 쓰게 할 수 있다는 뜻일 수 있습니다."
    },
    {
        "title": "4대 사회보험 미적용 가능성",
        "severity": 5,
        "patterns": [r"4대\s*사회보험.*미적용", r"고용보험.*미적용", r"산재보험.*미적용", r"국민연금.*미적용", r"건강보험.*미적용", r"사회보험.*미가입", r"미적용을\s*원칙"],
        "explanation": "근로자라면 4대 사회보험 적용 여부가 중요합니다. 미적용을 원칙으로 한다는 문구는 사고, 실업, 건강 문제 발생 시 보호가 약해질 수 있어 불리합니다.",
        "easy": "일하다 다치거나 그만두게 됐을 때 보호를 못 받을 수 있습니다."
    },
    {
        "title": "포괄임금 또는 추가수당 미지급 가능성",
        "severity": 5,
        "patterns": [r"포괄\s*임금", r"월급에\s*포함", r"수당.*포함", r"별도\s*수당.*지급하지", r"별도\s*지급하지", r"연장.*야간.*휴일.*수당", r"초과근로수당.*포함"],
        "explanation": "근로계약서라면 추가 근무를 해도 수당을 제대로 받지 못할 가능성이 있어 중요합니다.",
        "easy": "야근이나 주말 근무를 해도 돈을 더 못 받을 수 있습니다."
    },
    {
        "title": "휴게시간이 비어 있거나 불명확함",
        "severity": 4,
        "patterns": [r"휴게:\s*시분", r"휴게\s*:\s*시분", r"휴게.*시분", r"휴게시간.*미기재", r"휴게.*공란"],
        "explanation": "근무시간이 긴데 휴게시간이 비어 있으면 실제 쉬는 시간이 보장되지 않을 수 있습니다.",
        "easy": "몇 시부터 몇 시까지 쉬는지 안 적혀 있으면 쉬는 시간이 애매해질 수 있습니다."
    },
    {
        "title": "근무장소 또는 업무 내용이 비어 있음",
        "severity": 4,
        "patterns": [r"근무장소\s*:\s*(\n|$)", r"업무의\s*내용\s*:\s*(\n|$)", r"근무장소\s*:\s*$", r"업무의\s*내용\s*:\s*$"],
        "explanation": "근무장소와 업무 내용이 비어 있으면 나중에 예상하지 못한 장소나 업무를 요구받을 수 있습니다.",
        "easy": "어디서 무슨 일을 하는지 안 적혀 있으면 나중에 다른 일을 시킬 여지가 생깁니다."
    },
    {
        "title": "근로시간이 길거나 법정 기준 초과 가능성",
        "severity": 4,
        "patterns": [r"1주\s*48", r"1주48", r"주\s*48", r"1주481간", r"10시00분.*20시\s*30분", r"10시.*20시"],
        "explanation": "근무 시간이 길거나 주당 근로시간이 높게 적혀 있으면 연장근로와 수당 문제가 생길 수 있습니다.",
        "easy": "일하는 시간이 길어서 추가수당을 받아야 하는 상황일 수 있습니다."
    },
    {
        "title": "주 6일 근무로 인한 부담 가능성",
        "severity": 4,
        "patterns": [r"매주\s*6일\s*근무", r"주\s*6일\s*근무", r"6일\s*근무"],
        "explanation": "주 6일 근무는 휴식 시간이 부족해질 수 있고, 실제 근로시간에 따라 연장근로수당 문제가 생길 수 있습니다.",
        "easy": "일주일에 6일 일하는 조건이면 쉬는 시간이 부족할 수 있습니다."
    },
    {
        "title": "면책 또는 책임 회피 조항",
        "severity": 5,
        "patterns": [r"면책", r"책임을\s*지지\s*않", r"책임\s*없", r"책임\s*제한", r"손해배상\s*책임\s*없", r"당사는.*책임.*없", r"갑은.*책임.*없"],
        "explanation": "문제가 생겼을 때 상대방이 책임을 피할 가능성이 커지므로 사용자에게 불리합니다.",
        "easy": "문제가 생겨도 상대가 책임지지 않겠다는 뜻일 수 있습니다."
    },
    {
        "title": "자동 갱신 또는 묵시적 연장 조항",
        "severity": 4,
        "patterns": [r"자동\s*갱신", r"자동\s*연장", r"묵시적\s*동의", r"통보하지\s*않으면", r"별도.*의사.*없", r"동일한\s*조건으로\s*갱신"],
        "explanation": "사용자가 따로 해지 의사를 밝히지 않으면 계약이 계속 이어질 수 있습니다.",
        "easy": "가만히 있으면 계약이 자동으로 계속될 수 있습니다."
    },
    {
        "title": "비용 부담이 사용자에게 치우침",
        "severity": 4,
        "patterns": [r"수수료", r"비용은\s*을", r"제반\s*비용", r"일체의\s*비용", r"부대\s*비용", r"실비", r"이용료"],
        "explanation": "비용 항목이 불명확하면 사용자가 예상하지 못한 돈을 부담할 수 있습니다.",
        "easy": "처음 생각한 돈보다 추가 비용이 더 붙을 수 있습니다."
    },
]


def analyze_contract_text(text: str) -> Dict[str, Any]:
    text = normalize_text(text)
    doc_info = detect_document_type(text)
    sentences = split_sentences_ko(text)

    pros = []
    cons = []

    for rule in FAVORABLE_RULES:
        evidence = find_matching_sentences(sentences, rule["patterns"], text, max_items=2)

        if evidence:
            pros.append({
                "title": rule["title"],
                "severity": rule["severity"],
                "evidence": evidence,
                "explanation": rule["explanation"],
                "easy": rule["easy"]
            })

    for rule in UNFAVORABLE_RULES:
        evidence = find_matching_sentences(sentences, rule["patterns"], text, max_items=2)

        if evidence:
            cons.append({
                "title": rule["title"],
                "severity": rule["severity"],
                "evidence": evidence,
                "explanation": rule["explanation"],
                "easy": rule["easy"]
            })

    pros.sort(key=lambda x: -x["severity"])
    cons.sort(key=lambda x: -x["severity"])

    questions = []

    if any("연차" in c["title"] for c in cons):
        questions.append("연차 사용 제한 사유와 미사용 시 보상 방식이 명확한지 확인해야 합니다.")

    if any("4대 사회보험" in c["title"] for c in cons):
        questions.append("실제 근로 형태가 4대 사회보험 적용 대상인지 확인해야 합니다.")

    if any("휴게시간" in c["title"] for c in cons):
        questions.append("휴게시간의 시작과 종료 시간이 비어 있지 않은지 확인해야 합니다.")

    if any("근무장소" in c["title"] or "업무 내용" in c["title"] for c in cons):
        questions.append("근무장소와 업무 내용을 구체적으로 적어야 합니다.")

    if any("근로시간" in c["title"] or "추가수당" in c["title"] or "포괄임금" in c["title"] for c in cons):
        questions.append("실제 근로시간, 연장근로 여부, 추가수당 계산 방식을 확인해야 합니다.")

    if any("주 6일" in c["title"] for c in cons):
        questions.append("주 6일 근무가 실제 근로시간과 휴일수당 문제로 이어지는지 확인해야 합니다.")

    if not questions:
        questions.append("계약기간, 지급조건, 해지조건, 책임범위, 개인정보, 비용 항목을 원문 기준으로 확인해야 합니다.")

    return {
        "document_type": doc_info["document_type"],
        "document_confidence": doc_info["confidence"],
        "document_scores": doc_info["scores"],
        "ocr_text": text,
        "user_role": USER_ROLE,
        "유리한_조항": pros,
        "불리한_조항": cons,
        "확인할_질문": questions
    }


st.set_page_config(page_title="법률문서 OCR 분석기", layout="wide")

st.title("법률문서 OCR 유리·불리 조항 분석기")
st.write("법률문서 이미지를 업로드하면 OCR로 텍스트를 추출하고, 사용자(을) 기준으로 유리한 조항과 불리한 조항을 분석합니다.")

uploaded_file = st.file_uploader("법률문서 이미지 업로드", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image_bytes = uploaded_file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("업로드한 이미지")
        st.image(image, use_container_width=True)

    with col2:
        st.subheader("분석 설정")
        st.write("분석 기준: 사용자(을)")
        run = st.button("OCR 및 분석 실행")

    if run:
        with st.spinner("OCR 처리 중입니다..."):
            ocr_text = extract_text_from_image(image_bytes)
            ocr_text = normalize_text(ocr_text)

        analysis_result = analyze_contract_text(ocr_text)

        st.divider()
        st.subheader("문서 종류 판별 결과")

        st.write(f"문서 종류: {analysis_result['document_type']}")
        st.write(f"판별 신뢰도: {analysis_result['document_confidence']}%")

        with st.expander("문서 종류별 점수 보기"):
            st.json(analysis_result["document_scores"])

        st.divider()
        st.subheader("OCR 추출 텍스트")
        st.text_area("추출된 텍스트", ocr_text, height=280)

        st.divider()
        st.subheader("유리한 조항")

        if not analysis_result["유리한_조항"]:
            st.info("탐지된 유리한 조항이 없습니다.")
        else:
            for item in analysis_result["유리한_조항"]:
                with st.container(border=True):
                    st.markdown(f"### {item['title']}")
                    st.write(f"이유: {item['explanation']}")
                    st.write(f"쉬운 설명: {item['easy']}")
                    for ev in item["evidence"]:
                        st.write(f"근거: {ev}")

        st.divider()
        st.subheader("불리한 조항")

        if not analysis_result["불리한_조항"]:
            st.info("탐지된 불리한 조항이 없습니다.")
        else:
            for item in analysis_result["불리한_조항"]:
                with st.container(border=True):
                    st.markdown(f"### {item['title']} / 위험도 {item['severity']}/5")
                    st.write(f"이유: {item['explanation']}")
                    st.write(f"쉬운 설명: {item['easy']}")
                    for ev in item["evidence"]:
                        st.write(f"근거: {ev}")

        st.divider()
        st.subheader("확인할 질문")

        for idx, q in enumerate(analysis_result["확인할_질문"], 1):
            st.write(f"{idx}. {q}")

        result_json = json.dumps(analysis_result, ensure_ascii=False, indent=2)

        st.download_button(
            label="분석 결과 JSON 다운로드",
            data=result_json,
            file_name="legal_analysis_result.json",
            mime="application/json"
        )

        st.warning("이 결과는 법률 자문이 아니라 문서 이해를 돕기 위한 자동 분석입니다. 중요한 계약은 전문가 검토가 필요합니다.")
