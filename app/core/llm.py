from crewai import LLM


def create_llm(provider, model_name, api_key):
    if provider == "gemini":
        return LLM(
            model=f"{provider}/{model_name}",
            temperature=0.0,
            api_key=api_key,
        )
