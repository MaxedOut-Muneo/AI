def search_estimate_cases_for_reference(question: str):
    """
    견적서 사례는 가격 산정 목적이 아니라,
    견적서 양식/포함 항목/누락 가능 항목 참고용으로 사용한다.

    동료의 estimate_engine.py 함수명이 다를 수 있으므로
    여러 함수명을 순서대로 시도한다.
    """

    try:
        import estimate.estimate_engine as engine
    except Exception as e:
        return f"견적 사례 엔진을 불러오지 못했습니다: {e}"

    candidate_function_names = [
        "search_estimate_cases",
        "generate_estimate",
        "run_estimate_engine",
        "estimate_engine",
        "main"
    ]

    for name in candidate_function_names:
        fn = getattr(engine, name, None)

        if callable(fn):
            try:
                return fn(question)
            except TypeError:
                continue
            except Exception as e:
                return f"견적 사례 검색 중 오류가 발생했습니다: {e}"

    return (
        "estimate/estimate_engine.py에서 호출 가능한 함수를 찾지 못했습니다. "
        "search_estimate_cases(question) 형태의 함수를 만들어 연결하는 것을 권장합니다."
    )