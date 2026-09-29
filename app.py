import io
import re
import json
from typing import List, Dict, Any
from xml.sax.saxutils import escape
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.lib.styles import ParagraphStyle
from openai import OpenAI

import streamlit as st
from PIL import Image
from google.cloud import vision
from google.oauth2 import service_account

from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet


USER_ROLE = "근로자"


# ---------------------------------------------------------
# Google Cloud Vision 인증
# ---------------------------------------------------------

def get_vision_client():
    try:
        gcp_info = dict(st.secrets["gcp_service_account"])

        # Streamlit Secrets에서 \n이 문자 그대로 저장된 경우 실제 줄바꿈으로 변환
        private_key = gcp_info.get("private_key", "")

        if "\\n" in private_key:
            private_key = private_key.replace("\\n", "\n")

        private_key = private_key.strip()

        # PEM 형식 확인
        if not private_key.startswith("-----BEGIN PRIVATE KEY-----"):
            raise ValueError(
                "private_key가 올바른 PEM 형식으로 시작하지 않습니다."
            )

        if not private_key.endswith("-----END PRIVATE KEY-----"):
            raise ValueError(
                "private_key가 올바른 PEM 형식으로 끝나지 않습니다."
            )

        gcp_info["private_key"] = private_key

        credentials = service_account.Credentials.from_service_account_info(
            gcp_info
        )

        return vision.ImageAnnotatorClient(
            credentials=credentials
        )

    except KeyError:
        st.error(
            "Streamlit Secrets에 [gcp_service_account] 인증 정보가 없습니다."
        )
        st.stop()

    except Exception as e:
        st.error(f"Google Cloud 인증 오류: {e}")
        st.stop()

# ---------------------------------------------------------
# 텍스트 정리
# ---------------------------------------------------------

def normalize_text(text: str) -> str:
    if text is None:
        return ""

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)

    return text.strip()


# ---------------------------------------------------------
# OCR
# ---------------------------------------------------------

def extract_text_from_image(image_bytes: bytes) -> str:
    client = get_vision_client()

    image = vision.Image(
        content=image_bytes
    )

    response = client.document_text_detection(
        image=image
    )

    if response.error.message:
        raise RuntimeError(
            f"Google Cloud Vision OCR 오류: {response.error.message}"
        )

    return response.full_text_annotation.text or ""


# ---------------------------------------------------------
# 문서 종류 판별
# ---------------------------------------------------------

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

    confidence = round(
        (best_score / total_possible) * 100
    )

    return {
        "document_type": best_type,
        "confidence": confidence,
        "scores": scores
    }


# ---------------------------------------------------------
# 문장 분리
# ---------------------------------------------------------

def split_sentences_ko(text: str) -> List[str]:
    text = normalize_text(text)
    sentences = []

    for line in text.split("\n"):
        line = line.strip()

        if not line:
            continue

        parts = re.split(
            r"(?<=[.!?])\s+|"
            r"(?<=다)\s+|"
            r"(?<=함)\s+|"
            r"(?<=됩니다)\s+|"
            r"(?<=합니다)\s+|"
            r"(?<=한다)\s+",
            line
        )

        for part in parts:
            part = part.strip(" \t-")

            if part:
                sentences.append(part)

    return sentences


# ---------------------------------------------------------
# 근거 문장 정리
# ---------------------------------------------------------

def clean_evidence_text(text: str) -> str:
    text = normalize_text(text)
    text = re.sub(
        r"^\s*[-–—]\s*",
        "",
        text
    )

    return text.strip()


def is_useful_evidence(text: str) -> bool:
    text = clean_evidence_text(text)

    if not text:
        return False

    if len(text) <= 1:
        return False

    if text in {
        "-", "원", "일", "월", "년", "시분"
    }:
        return False

    if len(
        text.replace("\n", "").strip()
    ) < 4:
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


# ---------------------------------------------------------
# 패턴 매칭
# ---------------------------------------------------------

def find_matching_sentences(
    sentences: List[str],
    patterns: List[str],
    full_text: str = "",
    max_items: int = 2
) -> List[str]:

    candidates = []

    for sentence in sentences:
        sentence = clean_evidence_text(
            sentence
        )

        if not is_useful_evidence(
            sentence
        ):
            continue

        for pattern in patterns:
            if re.search(
                pattern,
                sentence,
                flags=re.IGNORECASE | re.MULTILINE
            ):
                candidates.append(
                    sentence
                )
                break

    if full_text:
        lines = [
            clean_evidence_text(line)
            for line in full_text.split("\n")
        ]

        lines = [
            line
            for line in lines
            if is_useful_evidence(line)
        ]

        for line in lines:
            for pattern in patterns:

                if re.search(
                    pattern,
                    line,
                    flags=re.IGNORECASE | re.MULTILINE
                ):
                    candidates.append(
                        line
                    )
                    break

    unique = []
    seen = set()

    for item in candidates:
        item = clean_evidence_text(item)

        key = re.sub(
            r"\s+",
            "",
            item
        )

        duplicate = False

        for old in seen:
            if key in old or old in key:
                duplicate = True
                break

        if not duplicate:
            unique.append(item)
            seen.add(key)

    unique.sort(
        key=evidence_score,
        reverse=True
    )

    return unique[:max_items]


# ---------------------------------------------------------
# 유리한 조항 규칙
# ---------------------------------------------------------

FAVORABLE_RULES: List[Dict[str, Any]] = [

    {
        "title": "대금 또는 임금 지급 기준이 비교적 명확함",

        "severity": 3,

        "patterns": [
            r"지급일",
            r"지급\s*방법",
            r"계좌",
            r"입금",
            r"정산",
            r"지급한다",
            r"지급하여야"
        ],

        "explanation":
            "돈을 언제, 어떤 방식으로 지급하는지 적혀 있으면 "
            "미지급이나 지연이 생겼을 때 근거로 삼기 좋습니다.",

        "easy":
            "돈을 언제 어떻게 받을 수 있는지 적혀 있으면 "
            "나중에 따지기 쉽습니다."
    },

    {
        "title": "근로개시일이 명시됨",

        "severity": 2,

        "patterns": [
            r"근로개시일",
            r"시작일",
            r"계약\s*시작"
        ],

        "explanation":
            "언제부터 근로가 시작되는지 적혀 있으면 "
            "근로 시작 시점을 확인하는 근거가 됩니다.",

        "easy":
            "언제부터 일하기로 했는지 확인할 수 있습니다."
    },

    {
        "title": "상대방의 의무가 문서에 적혀 있음",

        "severity": 3,

        "patterns": [
            r"갑은.*하여야",
            r"갑은.*제공",
            r"사업주는.*교부",
            r"사업주는.*하여야",
            r"회사.*하여야",
            r"사업자.*하여야"
        ],

        "explanation":
            "상대방이 해야 할 일이 명확히 적혀 있으면 "
            "사용자가 이행을 요구하기 쉽습니다.",

        "easy":
            "상대가 해야 할 일이 적혀 있으면 "
            "지키라고 말할 근거가 생깁니다."
    },

    {
        "title": "유급휴일 기준이 언급됨",

        "severity": 2,

        "patterns": [
            r"공휴일.*근로기준법",
            r"근로자의\s*날.*유급휴일",
            r"대체공휴일.*포함",
            r"유급휴일로\s*함"
        ],

        "explanation":
            "공휴일이나 근로자의 날을 유급휴일로 본다는 기준이 "
            "적혀 있으면 휴일 처리와 수당 문제를 확인할 근거가 됩니다.",

        "easy":
            "쉬는 날을 유급으로 인정한다는 내용이 있으면 "
            "나중에 확인할 기준이 생깁니다."
    },
]


# ---------------------------------------------------------
# 불리한 조항 규칙
# ---------------------------------------------------------

UNFAVORABLE_RULES: List[Dict[str, Any]] = [

    {
        "title": "연차유급휴가 사용 제한 가능성",

        "severity": 5,

        "patterns": [
            r"연차.*회사\s*사정",
            r"연차.*사용하지\s*못할\s*수",
            r"휴가.*사용하지\s*못할\s*수",
            r"연차휴가는.*사용하지\s*못"
        ],

        "explanation":
            "연차유급휴가는 근로자의 중요한 권리인데, "
            "회사 사정만으로 사용하지 못하게 할 수 있다는 표현은 "
            "근로자에게 매우 불리합니다.",

        "easy":
            "회사가 바쁘다는 이유로 내 연차를 "
            "못 쓰게 할 수 있다는 뜻일 수 있습니다."
    },

    {
        "title": "4대 사회보험 미적용 가능성",

        "severity": 5,

        "patterns": [
            r"4대\s*사회보험.*미적용",
            r"고용보험.*미적용",
            r"산재보험.*미적용",
            r"국민연금.*미적용",
            r"건강보험.*미적용",
            r"사회보험.*미가입",
            r"미적용을\s*원칙"
        ],

        "explanation":
            "근로자라면 4대 사회보험 적용 여부가 중요합니다. "
            "미적용을 원칙으로 한다는 문구는 사고, 실업, 건강 문제 발생 시 "
            "보호가 약해질 수 있어 불리합니다.",

        "easy":
            "일하다 다치거나 그만두게 됐을 때 "
            "보호를 못 받을 수 있습니다."
    },

    {
        "title": "포괄임금 또는 추가수당 미지급 가능성",

        "severity": 5,

        "patterns": [
            r"포괄\s*임금",
            r"월급에\s*포함",
            r"수당.*포함",
            r"별도\s*수당.*지급하지",
            r"별도\s*지급하지",
            r"연장.*야간.*휴일.*수당",
            r"초과근로수당.*포함"
        ],

        "explanation":
            "근로계약서라면 추가 근무를 해도 "
            "수당을 제대로 받지 못할 가능성이 있어 중요합니다.",

        "easy":
            "야근이나 주말 근무를 해도 "
            "돈을 더 못 받을 수 있습니다."
    },

    {
        "title": "휴게시간이 비어 있거나 불명확함",

        "severity": 4,

        "patterns": [
            r"휴게:\s*시분",
            r"휴게\s*:\s*시분",
            r"휴게.*시분",
            r"휴게시간.*미기재",
            r"휴게.*공란"
        ],

        "explanation":
            "근무시간이 긴데 휴게시간이 비어 있으면 "
            "실제 쉬는 시간이 보장되지 않을 수 있습니다.",

        "easy":
            "몇 시부터 몇 시까지 쉬는지 안 적혀 있으면 "
            "쉬는 시간이 애매해질 수 있습니다."
    },

    {
        "title": "근무장소 또는 업무 내용이 비어 있음",

        "severity": 4,

        "patterns": [
            r"근무장소\s*:\s*(\n|$)",
            r"업무의\s*내용\s*:\s*(\n|$)",
            r"근무장소\s*:\s*$",
            r"업무의\s*내용\s*:\s*$"
        ],

        "explanation":
            "근무장소와 업무 내용이 비어 있으면 "
            "나중에 예상하지 못한 장소나 업무를 요구받을 수 있습니다.",

        "easy":
            "어디서 무슨 일을 하는지 안 적혀 있으면 "
            "나중에 다른 일을 시킬 여지가 생깁니다."
    },

    {
        "title": "근로시간이 길거나 법정 기준 초과 가능성",

        "severity": 4,

        "patterns": [
            r"1주\s*48",
            r"1주48",
            r"주\s*48",
            r"1주481간",
            r"10시00분.*20시\s*30분",
            r"10시.*20시"
        ],

        "explanation":
            "근무 시간이 길거나 주당 근로시간이 높게 적혀 있으면 "
            "연장근로와 수당 문제가 생길 수 있습니다.",

        "easy":
            "일하는 시간이 길어서 추가수당을 받아야 하는 "
            "상황일 수 있습니다."
    },

    {
        "title": "주 6일 근무로 인한 부담 가능성",

        "severity": 4,

        "patterns": [
            r"매주\s*6일\s*근무",
            r"주\s*6일\s*근무",
            r"6일\s*근무"
        ],

        "explanation":
            "주 6일 근무는 휴식 시간이 부족해질 수 있고, "
            "실제 근로시간에 따라 연장근로수당 문제가 생길 수 있습니다.",

        "easy":
            "일주일에 6일 일하는 조건이면 "
            "쉬는 시간이 부족할 수 있습니다."
    },

    {
        "title": "면책 또는 책임 회피 조항",

        "severity": 5,

        "patterns": [
            r"면책",
            r"책임을\s*지지\s*않",
            r"책임\s*없",
            r"책임\s*제한",
            r"손해배상\s*책임\s*없",
            r"당사는.*책임.*없",
            r"갑은.*책임.*없"
        ],

        "explanation":
            "문제가 생겼을 때 상대방이 책임을 피할 가능성이 "
            "커지므로 사용자에게 불리합니다.",

        "easy":
            "문제가 생겨도 상대가 책임지지 않겠다는 "
            "뜻일 수 있습니다."
    },

    {
        "title": "자동 갱신 또는 묵시적 연장 조항",

        "severity": 4,

        "patterns": [
            r"자동\s*갱신",
            r"자동\s*연장",
            r"묵시적\s*동의",
            r"통보하지\s*않으면",
            r"별도.*의사.*없",
            r"동일한\s*조건으로\s*갱신"
        ],

        "explanation":
            "사용자가 따로 해지 의사를 밝히지 않으면 "
            "계약이 계속 이어질 수 있습니다.",

        "easy":
            "가만히 있으면 계약이 자동으로 계속될 수 있습니다."
    },

    {
        "title": "비용 부담이 사용자에게 치우침",

        "severity": 4,

        "patterns": [
            r"수수료",
            r"비용은\s*을",
            r"제반\s*비용",
            r"일체의\s*비용",
            r"부대\s*비용",
            r"실비",
            r"이용료"
        ],

        "explanation":
            "비용 항목이 불명확하면 사용자가 예상하지 못한 "
            "돈을 부담할 수 있습니다.",

        "easy":
            "처음 생각한 돈보다 추가 비용이 더 붙을 수 있습니다."
    },
]


# ---------------------------------------------------------
# 계약서 분석
# ---------------------------------------------------------

def analyze_contract_text(
    text: str, role: str = "근로자"
) -> Dict[str, Any]:

    text = normalize_text(text)

    doc_info = detect_document_type(
        text
    )

    sentences = split_sentences_ko(
        text
    )

    pros = []
    cons = []

    for rule in FAVORABLE_RULES:

        evidence = find_matching_sentences(
            sentences,
            rule["patterns"],
            text,
            max_items=2
        )

        if evidence:
            pros.append({
                "title": rule["title"],
                "severity": rule["severity"],
                "evidence": evidence,
                "explanation": rule["explanation"],
                "easy": rule["easy"]
            })

    for rule in UNFAVORABLE_RULES:

        evidence = find_matching_sentences(
            sentences,
            rule["patterns"],
            text,
            max_items=2
        )

        if evidence:
            cons.append({
                "title": rule["title"],
                "severity": rule["severity"],
                "evidence": evidence,
                "explanation": rule["explanation"],
                "easy": rule["easy"]
            })

    pros.sort(
        key=lambda x: -x["severity"]
    )

    cons.sort(
        key=lambda x: -x["severity"]
    )

    questions = []

    if any(
        "연차" in c["title"]
        for c in cons
    ):
        questions.append(
            "연차 사용 제한 사유와 미사용 시 보상 방식이 "
            "명확한지 확인해야 합니다."
        )

    if any(
        "4대 사회보험" in c["title"]
        for c in cons
    ):
        questions.append(
            "실제 근로 형태가 4대 사회보험 적용 대상인지 "
            "확인해야 합니다."
        )

    if any(
        "휴게시간" in c["title"]
        for c in cons
    ):
        questions.append(
            "휴게시간의 시작과 종료 시간이 비어 있지 않은지 "
            "확인해야 합니다."
        )

    if any(
        "근무장소" in c["title"]
        or "업무 내용" in c["title"]
        for c in cons
    ):
        questions.append(
            "근무장소와 업무 내용을 구체적으로 적어야 합니다."
        )

    if any(
        "근로시간" in c["title"]
        or "추가수당" in c["title"]
        or "포괄임금" in c["title"]
        for c in cons
    ):
        questions.append(
            "실제 근로시간, 연장근로 여부, 추가수당 계산 방식을 "
            "확인해야 합니다."
        )

    if any(
        "주 6일" in c["title"]
        for c in cons
    ):
        questions.append(
            "주 6일 근무가 실제 근로시간과 휴일수당 문제로 "
            "이어지는지 확인해야 합니다."
        )

    if not questions:
        questions.append(
            "계약기간, 지급조건, 해지조건, 책임범위, "
            "개인정보, 비용 항목을 원문 기준으로 확인해야 합니다."
        )

    return {
        "document_type":
            doc_info["document_type"],

        "document_confidence":
            doc_info["confidence"],

        "document_scores":
            doc_info["scores"],

        "ocr_text":
            text,

        "user_role":
            role,

        "유리한_조항":
            pros,

        "불리한_조항":
            cons,

        "확인할_질문":
            questions
    }


# ---------------------------------------------------------
# PDF 생성
# ---------------------------------------------------------

def create_pdf_report(
    analysis_result
):
    pdf_buffer = io.BytesIO()

    doc = SimpleDocTemplate(
        pdf_buffer,
        pagesize=A4
    )

    pdfmetrics.registerFont(UnicodeCIDFont("HYGothic-Medium"))
    styles = getSampleStyleSheet()
    for style in styles.byName.values():
        style.fontName = "HYGothic-Medium"
    story = []

    story.append(
        Paragraph(
            "법률문서 OCR 분석 보고서",
            styles["Title"]
        )
    )

    story.append(
        Spacer(1, 12)
    )

    story.append(
        Paragraph(
            f"문서 종류: "
            f"{escape(str(analysis_result.get('document_type', '기타')))}",
            styles["Normal"]
        )
    )

    story.append(
        Paragraph(
            f"판별 신뢰도: "
            f"{analysis_result.get('document_confidence', 0)}%",
            styles["Normal"]
        )
    )

    story.append(
        Paragraph(
            f"분석 기준: "
            f"{escape(str(analysis_result.get('user_role', '근로자')))}",
            styles["Normal"]
        )
    )

    story.append(
        Spacer(1, 16)
    )

    story.append(
        Paragraph(
            "유리한 조항",
            styles["Heading2"]
        )
    )

    for idx, item in enumerate(
        analysis_result.get(
            "유리한_조항",
            []
        ),
        1
    ):

        story.append(
            Paragraph(
                f"{idx}. {escape(item['title'])}",
                styles["Heading3"]
            )
        )

        story.append(
            Paragraph(
                f"이유: {escape(item['explanation'])}",
                styles["Normal"]
            )
        )

        story.append(
            Paragraph(
                f"쉬운 설명: {escape(item['easy'])}",
                styles["Normal"]
            )
        )

        for ev in item.get(
            "evidence",
            []
        ):
            story.append(
                Paragraph(
                    f"근거: {escape(ev)}",
                    styles["Normal"]
                )
            )

        story.append(
            Spacer(1, 10)
        )

    story.append(
        Spacer(1, 12)
    )

    story.append(
        Paragraph(
            "불리한 조항",
            styles["Heading2"]
        )
    )

    for idx, item in enumerate(
        analysis_result.get(
            "불리한_조항",
            []
        ),
        1
    ):

        story.append(
            Paragraph(
                f"{idx}. {item['title']} "
                f"/ 위험도 {item['severity']}/5",
                styles["Heading3"]
            )
        )

        story.append(
            Paragraph(
                f"이유: {escape(item['explanation'])}",
                styles["Normal"]
            )
        )

        story.append(
            Paragraph(
                f"쉬운 설명: {escape(item['easy'])}",
                styles["Normal"]
            )
        )

        for ev in item.get(
            "evidence",
            []
        ):
            story.append(
                Paragraph(
                    f"근거: {escape(ev)}",
                    styles["Normal"]
                )
            )

        story.append(
            Spacer(1, 10)
        )

    story.append(
        Spacer(1, 12)
    )

    story.append(
        Paragraph(
            "확인할 질문",
            styles["Heading2"]
        )
    )

    for idx, q in enumerate(
        analysis_result.get(
            "확인할_질문",
            []
        ),
        1
    ):

        story.append(
            Paragraph(
                f"{idx}. {escape(q)}",
                styles["Normal"]
            )
        )

    story.append(Spacer(1, 12))
    story.append(Paragraph("AI 설명과 확인 질문", styles["Heading2"]))
    for item in analysis_result.get("ai_items", []):
        for line in (
            f"{item['impact']} / {item['page']}페이지",
            f"원문: {item['quote']}",
            f"쉬운 설명: {item['explanation']}",
            f"확인 질문: {item['question']}",
        ):
            story.append(Paragraph(escape(line), styles["Normal"]))
        story.append(Spacer(1, 8))

    doc.build(
        story
    )

    return pdf_buffer.getvalue()


# ---------------------------------------------------------
# AI 설명: 원문 인용을 검증해 근거 없는 결과를 제외
# ---------------------------------------------------------
ROLE_OPTIONS = {
    "근로계약서": ["근로자", "고용주"],
    "임대차계약서": ["임차인", "임대인"],
    "프리랜서계약서": ["수탁자", "발주자"],
    "용역계약서": ["수급인", "도급인"],
    "개인정보동의서": ["정보주체", "개인정보처리자"],
    "이용약관": ["이용자", "사업자"],
    "기타": ["문서 이용자", "문서 작성자"],
}


def ai_feedback(pages, role, document_type, rules):
    try:
        key = st.secrets.get("OPENAI_API_KEY")
    except Exception:
        key = None
    if not key:
        return [], "OPENAI_API_KEY가 없어 규칙 분석만 표시합니다."
    try:
        model = st.secrets.get("OPENAI_MODEL", "gpt-5")
    except Exception:
        model = "gpt-5"
    source = "\n\n".join(
        f"[페이지 {i}]\n{page[:18000]}" for i, page in enumerate(pages, 1)
    )
    source = source[:90000]
    prompt = (
        f"문서 종류: {document_type}; 사용자 입장: {role}\n"
        "아래 OCR 문서만 근거로 최대 8개의 중요한 조건을 분석하라. "
        "근로자·고용주 등 사용자의 입장에 따라 이익과 부담을 구분하되, "
        "법적 효력이나 위법 여부를 단정하지 말라. OCR 오류, 빈칸, 누락은 확인 필요로 표시하라. "
        "각 항목의 quote는 아래 문서에 실제로 존재하는 짧고 연속된 원문 구절이어야 한다. "
        "반드시 JSON 객체 하나만 출력하라: "
        '{"items":[{"page":1,"quote":"원문 그대로",'
        '"impact":"유리|불리|확인 필요","explanation":"쉬운 설명",'
        '"question":"확인 질문"}]}\n'
        f"기존 규칙 분석: {json.dumps({'유리': [x['title'] for x in rules['유리한_조항']], '불리': [x['title'] for x in rules['불리한_조항']]}, ensure_ascii=False)}\n"
        f"원문:\n{source}"
    )
    try:
        response = OpenAI(api_key=key).responses.create(
            model=model, input=prompt, store=False
        )
        raw = response.output_text.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        data = json.loads(raw)
        verified = []
        for item in data.get("items", []):
            page = item.get("page")
            quote = str(item.get("quote", "")).strip()
            if not isinstance(page, int) or not 1 <= page <= len(pages):
                continue
            compact = lambda value: re.sub(r"\s+", "", value)
            if len(compact(quote)) < 6 or compact(quote) not in compact(pages[page - 1]):
                continue
            if item.get("impact") not in ("유리", "불리", "확인 필요"):
                continue
            verified.append({
                "page": page, "quote": quote, "impact": item["impact"],
                "explanation": str(item.get("explanation", ""))[:700],
                "question": str(item.get("question", ""))[:350],
            })
        return verified[:8], None if verified else "AI 결과에서 원문 근거를 확인할 수 없어 표시하지 않았습니다."
    except Exception as exc:
        return [], f"AI 설명에 실패했습니다: {type(exc).__name__}. 규칙 분석 결과는 유지됩니다."


st.set_page_config(page_title="AI 문해력 브릿지", layout="wide")
st.title("AI 문해력 브릿지")
st.write("서류 전체 페이지를 순서대로 촬영하거나 업로드하고, 분석할 입장을 선택하세요.")

if "captured_pages" not in st.session_state:
    st.session_state.captured_pages = []

uploaded = st.file_uploader(
    "사진 여러 장 업로드 (선택한 순서대로 분석)",
    type=["png", "jpg", "jpeg"], accept_multiple_files=True
)
shot = st.camera_input("현재 페이지 촬영")
if shot is not None:
    shot_bytes = shot.getvalue()
    if st.button("촬영한 페이지 추가"):
        if shot_bytes not in st.session_state.captured_pages:
            st.session_state.captured_pages.append(shot_bytes)
        st.rerun()
if st.session_state.captured_pages and st.button("촬영 목록 비우기"):
    st.session_state.captured_pages = []
    st.rerun()

images = [f.getvalue() for f in (uploaded or [])] + st.session_state.captured_pages
st.write(f"현재 {len(images)}페이지. 순서가 맞는지 아래 미리보기로 확인하세요.")
if images:
    for index, image_bytes in enumerate(images, 1):
        with st.expander(f"{index}페이지 미리보기"):
            st.image(image_bytes, width=400)
    doc_choice = st.selectbox("문서 유형", list(ROLE_OPTIONS))
    role = st.selectbox("나의 입장", ROLE_OPTIONS[doc_choice])
    if st.button("전체 페이지 분석", type="primary"):
        page_texts = []
        for index, image_bytes in enumerate(images, 1):
            try:
                # EXIF 회전 및 이미지 인코딩 문제를 통일한다.
                with Image.open(io.BytesIO(image_bytes)) as im:
                    im.verify()
                with st.spinner(f"{index}/{len(images)}페이지 OCR 처리 중"):
                    page_texts.append(normalize_text(extract_text_from_image(image_bytes)))
            except Exception as exc:
                st.error(f"{index}페이지를 읽지 못했습니다: {exc}")
                st.stop()
        if not any(page_texts):
            st.error("읽힌 글자가 없습니다. 사진을 다시 촬영해 주세요.")
            st.stop()
        combined = "\n\n".join(
            f"[페이지 {i}]\n{text}" for i, text in enumerate(page_texts, 1)
        )
        result = analyze_contract_text(combined, role)
        result["user_role"] = role
        result["selected_document_type"] = doc_choice
        result["ocr_page_texts"] = page_texts
        with st.spinner("원문에 근거한 쉬운 설명 생성 중"):
            ai_items, ai_error = ai_feedback(page_texts, role, doc_choice, result)
        result["ai_items"] = ai_items
        result["ai_error"] = ai_error
        st.session_state.analysis_result = result

result = st.session_state.get("analysis_result")
if result:
    st.subheader("분석 결과")
    st.write(f"선택한 문서: {result['selected_document_type']} / 입장: {result['user_role']}")
    st.write(f"OCR 판별: {result['document_type']} (키워드 일치율 {result['document_confidence']}%)")
    if result['document_type'] != result['selected_document_type']:
        st.warning("선택한 문서 유형과 OCR 판별이 다릅니다. 문서와 촬영 순서를 확인하세요.")
    if result['ai_error']:
        st.info(result['ai_error'])
    st.subheader("AI 설명과 확인 질문")
    for item in result['ai_items']:
        with st.container(border=True):
            st.write(f"{item['impact']} / {item['page']}페이지")
            st.write(item['explanation'])
            st.write(f"원문: {item['quote']}")
            st.write(f"확인 질문: {item['question']}")
    st.subheader("기존 규칙으로 찾은 항목")
    st.caption("기존 규칙은 근로자·계약 상대방 관점에서 작성되어 있습니다. 다른 입장을 선택한 경우 유불리 판정 대신 탐지된 조항으로 검토하세요.")
    for group in ("유리한_조항", "불리한_조항"):
        with st.expander(f"{group}: {len(result[group])}개"):
            for item in result[group]:
                st.write(f"{item['title']}: {item['easy']}")
                for quote in item['evidence']:
                    st.write(f"근거: {quote}")
    with st.expander("페이지별 OCR 원문"):
        for i, page in enumerate(result['ocr_page_texts'], 1):
            st.text_area(f"{i}페이지", page, height=180, key=f"ocr_{i}")
    st.download_button(
        "분석 PDF 다운로드", create_pdf_report(result),
        file_name="bridge_analysis.pdf", mime="application/pdf"
    )
    st.caption("자동 분석은 문서 이해를 돕는 자료입니다. OCR 원문과 계약서 전체를 대조해 주세요.")
