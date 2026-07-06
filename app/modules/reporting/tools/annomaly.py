import json
import numpy as np
import pandas as pd
from crewai.tools import BaseTool
from pydantic import Field


class AnomalyDetectionTool(BaseTool):
    name: str = "Detect Abnormal Behaviours"
    description: str = "Use this tool to perform anomaly detection using Z-Scores."
    df: pd.DataFrame = Field(...)

    def _run(
        self,
        target_column: str,
        threshold: float = 3.0,
        use_log_transform: bool = True,
        return_columns: list[str] = None,
    ) -> str:
        if target_column not in self.df.columns:
            return json.dumps(
                {
                    "error": "column_not_found",
                    "message": f"Column '{target_column}' not found.",
                }
            )

        if not pd.api.types.is_numeric_dtype(self.df[target_column]):
            return json.dumps(
                {
                    "error": "column_not_numeric",
                    "message": f"Column '{target_column}' is not numeric.",
                }
            )

        try:
            col_data = self.df[target_column]

            # Apply transformation if enabled
            if use_log_transform:
                # Sign-preserving log transform: handles 0 and negative values safely
                transformed_data = np.sign(col_data) * np.log1p(np.abs(col_data))
            else:
                transformed_data = col_data

            mean = transformed_data.mean()
            std = transformed_data.std()

            if std == 0:
                return json.dumps(
                    {
                        "result": "no_variance",
                        "message": "Standard deviation is zero. No anomalies can be computed.",
                    }
                )

            # Calculate Z-scores based on the transformed distribution
            z_scores = np.abs((transformed_data - mean) / std)
            outliers = self.df[z_scores > threshold]

            if outliers.empty:
                return json.dumps(
                    {
                        "result": "no_anomalies_found",
                        "message": f"No anomalies detected at threshold {threshold}.",
                    }
                )

            cols_to_return = (
                return_columns if return_columns else self.df.columns.tolist()
            )

            # Return pipe-delimited string of the raw matching rows
            return (
                outliers[cols_to_return].head(20).round(2).to_csv(index=False, sep="|")
            )

        except Exception as e:
            return json.dumps({"error": "execution_failed", "message": str(e)})
