import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from crewai.tools import BaseTool


class ExpressionResult(BaseModel):
    result_type: str
    shape: list[int] | None
    data: str | float | int


class ExpressionTool(BaseTool):
    name: str = "Execute Custom Pandas Expression"
    description: str = (
        "Executes a custom Python expression using df, pd and np. Returns strict JSON with type and shape metadata."
    )
    df: pd.DataFrame = Field(...)

    def _run(self, expression: str) -> str:
        try:
            result = eval(expression, {"df": self.df, "pd": pd, "np": np}, {})

            if isinstance(result, pd.DataFrame):
                return ExpressionResult(
                    result_type="DataFrame",
                    shape=list(result.shape),
                    data=result.head(5).round(2).to_csv(sep="|"),
                ).model_dump_json()

            elif isinstance(result, pd.Series):
                return ExpressionResult(
                    result_type="Series",
                    shape=[len(result)],
                    data=result.head(5).round(2).to_csv(sep="|", header=False),
                ).model_dump_json()

            elif isinstance(result, (float, np.floating)):
                return ExpressionResult(
                    result_type="Float", shape=None, data=round(float(result), 2)
                ).model_dump_json()

            else:
                return ExpressionResult(
                    result_type=type(result).__name__, shape=None, data=str(result)
                ).model_dump_json()

        except Exception as e:
            return f"{"error": 'execution_failed', 'details': '{str(e)}'}"
