import pandas as pd
from pydantic import Field
from crewai.tools import BaseTool


class ParetoTool(BaseTool):
    name: str = "Pareto Analysis"
    description: str = (
        "Performs Pareto (80/20) analysis on a categorical column against a numeric value. Returns top drivers."
    )
    df: pd.DataFrame = Field(...)

    def _run(
        self,
        category_column: str,
        value_column: str,
        cumulative_threshold: float = 80.0,
    ) -> str:
        if (
            category_column not in self.df.columns
            or value_column not in self.df.columns
        ):
            return '{"error": "columns_not_found"}'

        try:
            pareto_df = (
                self.df.groupby(category_column)[value_column].sum().reset_index()
            )
            pareto_df = pareto_df.sort_values(by=value_column, ascending=False)

            total_value = pareto_df[value_column].sum()
            if total_value == 0:
                return '{"error": "sum_of_values_is_zero"}'

            pareto_df["percentage"] = (pareto_df[value_column] / total_value) * 100
            pareto_df["cumulative_percentage"] = pareto_df["percentage"].cumsum()

            top_drivers = pareto_df[
                pareto_df["cumulative_percentage"] <= cumulative_threshold
            ]

            if top_drivers.empty and not pareto_df.empty:
                top_drivers = pareto_df.head(1)

            return top_drivers.round(2).to_csv(index=False, sep="|")

        except Exception as e:
            return f"{'error': '{str(e)}'}"
