from crewai import Task, Agent


def create_insights_agent(llm, language: str) -> Agent:
    return Agent(
        role="Business Intelligence Agent",
        goal=f"Writing an analysis report in {language} that translate numerical insights into business language for decision makers that can not understand complex metrics but natural simple language.",
        backstory="Working on ERP domains, you accept statistics and insights about any unknown ERP module and apply business intelligence on it without making assumptions. you are so smart in correlating different dimensions to gain insights and show risks while explaining it in simple way.",
        tools=[],
        llm=llm,
        verbose=False,
    )


def create_insights_task(agent: Agent, language: str) -> Task:
    return Task(
        description="Analyze numerical report and write insights without any assumptions/hallucinations about data (facts driven) translating complex metrics into simple facts with advantages and risks.",
        expected_output=f"Analysis Report in {language} mentioning every dimention; Start with summary followed by insights in bullet points conclude by advices; without inroductions:\n\n - Translate complex metrics in business language without technical terms.\n- Report under 250 words.\n- Don't translate nouns.\n - Don't use ids instead of names.\n- Be number/percentage driven whenever available.\n- Consider anomalies risks/advantages, do not treat anomalies as wrong data entries.\n- Show super intelligence in numerical conflicts across various insight tables.",
        agent=agent,
    )
