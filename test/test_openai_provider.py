import app.providers.openai_provider


def test_openai_llm_is_configured():
    assert app.providers.openai_provider.openai_llm is not None
