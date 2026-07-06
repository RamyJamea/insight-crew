from crewai import Task, Agent


def create_analysis_agent(llm, tools: list) -> Agent:
    return Agent(
        role="CoT Insights Extraction Agent",
        goal="Perform data exploration and massive insights extraction via predefined tools. More tools used more accuracy/information gain.",
        backstory="working on an ERP data analysis, role is to explore data nad perform insgihts extraction to gather multi-dimension information. you excel in aggrigations and pandas expressions. you are a CoT persona.",
        tools=tools,
        llm=llm,
        verbose=True,
    )


def create_analysis_task(agent: Agent) -> Task:
    return Task(
        description="Use perdefined tools for data exploration to ask yourself insightful questions about data, clean and extract information, explore abnormal behaviours to help you adress questions to write massive numerical facts about the data. smart tool calling and observation for further tool calls until having a lot of insights to produce.",
        expected_output="Long Report that contains factual insights in tables. no analysis or assumtions just rich facts.",
        agent=agent,
    )
