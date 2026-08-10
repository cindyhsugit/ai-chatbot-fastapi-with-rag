import app.providers.gemini_provider


def test_gemini_llm_is_configured():
    assert app.providers.gemini_provider.gemini_llm is not None
