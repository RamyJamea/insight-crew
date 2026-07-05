import pandas as pd
from crewai import Crew, Process, LLM
from .agents import *
from .tools import *


def run_workflow(df: pd.DataFrame, llm: LLM, language: str):
    analysis_agent = create_analysis_agent(
        llm,
        tools=[
            RemoveColumnsTool(df=df),
            InspectionTool(df=df),
            AggregationTool(df=df),
            TimeSeriesTool(df=df),
            ExpressionTool(df=df),
            AnomalyDetectionTool(df=df),
            CorrelationAnalysisTool(df=df),
            ClusteringTool(df=df),
            ForecastTool(df=df),
            ParetoTool(df=df),
        ],
    )
    insights_agent = create_insights_agent(llm, language)

    analysis_task = create_analysis_task(analysis_agent)
    insights_task = create_insights_task(insights_agent, language)

    crew = Crew(
        agents=[analysis_agent, insights_agent],
        tasks=[analysis_task, insights_task],
        process=Process.sequential,
        verbose=False,
    )

    return crew.kickoff()
