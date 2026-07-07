from crewai import Task, Agent


def create_analysis_agent(llm, tools: list) -> Agent:
    return Agent(
        role="CoT Insights Extraction Agent",
        goal="Perform data exploration and massive insights extraction via predefined tools. More tools used more accuracy/information gain. Exploring excpected and unexpected asspects.",
        backstory="Working on an ERP data analysis while having over-thinking disorder, role is to explore data and perform insights extraction to gather multi-dimension information. Excel in running multiple aggregations and pandas expressions to gain rich insights.",
        tools=tools,
        llm=llm,
        verbose=True,
    )


def create_analysis_task(agent: Agent) -> Task:
    return Task(
        description="Use perdefined tools for data exploration asking about things needs to be discovered in this data, clean and extract information, explore abnormal behaviours.\n\n- Smart tool calls and observations exploring every dimension in the data (No insights left undiscovered).\n- Make sure to adapt data domain knowledge to know what exactly needs to be discovered.\n- Many tool calls must result in many discovered dimensions.",
        expected_output="Long Report that contains factual insights in tables. no analysis or assumptions just rich facts about the data.",
        agent=agent,
    )
