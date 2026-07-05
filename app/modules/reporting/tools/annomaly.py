import pandas as pd
import numpy as np
from crewai.tools import BaseTool
from pydantic import Field


class AnomalyDetectionTool(BaseTool):
    name: str = "Detect Anomalies"
    description: str = (
        "Finds outliers in a numeric column using Z-score thresholding. Returns pipe-delimited outlier rows."
    )
    df: pd.DataFrame = Field(...)

    def _run(
        self,
        target_column: str,
        threshold: float = 3.0,
        return_columns: list[str] = None,
    ) -> str:
        if target_column not in self.df.columns:
            return '{"error": "column_not_found"}'

        if not pd.api.types.is_numeric_dtype(self.df[target_column]):
            return '{"error": "column_not_numeric"}'

        try:
            col_data = self.df[target_column]
            mean, std = col_data.mean(), col_data.std()

            if std == 0:
                return '{"result": "no_variance"}'

            z_scores = np.abs((col_data - mean) / std)
            outliers = self.df[z_scores > threshold]

            if outliers.empty:
                return '{"result": "no_anomalies_found"}'

            cols_to_return = (
                return_columns if return_columns else self.df.columns.tolist()
            )
            return (
                outliers[cols_to_return].head(20).round(2).to_csv(index=False, sep="|")
            )

        except Exception as e:
            return f"{'error': '{str(e)}'}"
