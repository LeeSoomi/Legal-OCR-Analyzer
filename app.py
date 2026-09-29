            verified.append({
                "page": page, "quote": quote, "impact": item["impact"],
                "explanation": str(item.get("explanation", ""))[:700],
                "question": str(item.get("question", ""))[:350],
            })
        return verified[:8], None if verified else "AI 결과에서 원문 근거를 확인할 수 없어 표시하지 않았습니다."
    except Exception as exc:
        return [], f"AI 설명에 실패했습니다: {type(exc).__name__}. 규칙 분석 결과는 유지됩니다."


st.set_page_config(page_title="AI 문해력 브릿지", layout="wide")
st.markdown("<style>@media (max-width: 600px) {h1 {font-size: 2rem !important; line-height: 1.25 !important;}}</style>", unsafe_allow_html=True)
st.title("AI 문해력 브릿지")
st.write("서류 전체 페이지를 순서대로 촬영하거나 업로드하고, 분석할 입장을 선택하세요.")
st.caption("시연 촬영은 1080p를 요청합니다. 브라우저·기기에 따라 실제 해상도는 다를 수 있으므로 미리보기에서 글자를 확대해 확인하세요.")

if "captured_pages" not in st.session_state:
    st.session_state.captured_pages = []

uploaded = st.file_uploader(
    "사진 여러 장 업로드 (선택한 순서대로 분석)",
    type=["png", "jpg", "jpeg"], accept_multiple_files=True
)
shot = st.camera_input("현재 페이지 촬영 (고해상도 요청)", resolution="1080p")
if shot is not None:
    shot_bytes = shot.getvalue()
    with Image.open(io.BytesIO(shot_bytes)) as captured_image:
        st.caption(f"촬영 크기: {captured_image.width} × {captured_image.height} 픽셀")
        if captured_image.height < 1000:
            st.warning("촬영 해상도가 낮습니다. 기본 카메라로 촬영해 업로드하거나 다른 브라우저에서 다시 촬영하세요.")
    if st.button("촬영한 페이지 추가"):
        if shot_bytes not in st.session_state.captured_pages:
            st.session_state.captured_pages.append(shot_bytes)
        st.rerun()
if st.session_state.captured_pages and st.button("촬영 목록 비우기"):
    st.session_state.captured_pages = []
    st.rerun()

images = [f.getvalue() for f in (uploaded or [])] + st.session_state.captured_pages
st.write(f"현재 {len(images)}페이지. 순서가 맞는지 아래 미리보기로 확인하세요.")
if uploaded and st.session_state.captured_pages:
    st.warning("업로드 사진과 실시간 촬영 사진이 모두 포함됐습니다. 같은 페이지를 두 번 넣지 않았는지 확인하세요.")
if images:
    for index, image_bytes in enumerate(images, 1):
        with st.expander(f"{index}페이지 미리보기"):
            st.image(image_bytes, width=400)
            with Image.open(io.BytesIO(image_bytes)) as preview_image:
                st.caption(f"원본 크기: {preview_image.width} × {preview_image.height} 픽셀")
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
    if not result["유리한_조항"] and not result["불리한_조항"]:
        st.warning("기존 규칙에 일치하는 문구가 없습니다. 페이지별 OCR 원문에 글자가 정확히 읽혔는지 확인하세요.")
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
