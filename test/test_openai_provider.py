import providers.openai_provider


def test_openai_llm_is_configured():
    assert providers.openai_provider.openai_llm is not None
