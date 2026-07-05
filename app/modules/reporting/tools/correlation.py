import pandas as pd
import numpy as np
from crewai.tools import BaseTool
from pydantic import BaseModel, Field


class CorrelationResult(BaseModel):
    feature_a: str
    feature_b: str
    correlation: float


class CorrelationAnalysisTool(BaseTool):
    name: str = "Analyze Correlation"
    description: str = (
        "Calculates the Pearson correlation matrix for numeric columns and returns top correlated pairs."
    )
    df: pd.DataFrame = Field(...)

    def _run(self, threshold: float = 0.5) -> str:
        try:
            numeric_df = self.df.select_dtypes(include=np.number)
            if numeric_df.shape[1] < 2:
                return '{"error": "insufficient_numeric_columns"}'

            corr_matrix = numeric_df.corr().abs()

            upper_tri = corr_matrix.where(
                np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
            )

            results = []
            for col in upper_tri.columns:
                for row in upper_tri.index:
                    val = upper_tri.loc[row, col]
                    if not pd.isna(val) and val >= threshold:
                        results.append(
                            CorrelationResult(
                                feature_a=row,
                                feature_b=col,
                                correlation=round(float(val), 3),
                            )
                        )

            results.sort(key=lambda x: x.correlation, reverse=True)

            if not results:
                return '{"result": "no_strong_correlations"}'

            return f"[{','.join(r.model_dump_json() for r in results)}]"

        except Exception as e:
            return f"{'error': '{str(e)}'}"
