import pandas as pd
from crewai.tools import BaseTool
from pydantic import Field


class AggregationTool(BaseTool):
    name: str = "Aggregate Data"
    description: str = (
        "Aggregates a numeric column grouped by another column. Returns pipe-delimited data."
    )
    df: pd.DataFrame = Field(...)

    def _run(
        self,
        groupby_column: str,
        value_column: str,
        agg_operation: str = "sum",
        sort_descending: bool = True,
        limit: int = 10,
    ) -> str:
        if groupby_column not in self.df.columns or value_column not in self.df.columns:
            return '{"error": "columns_not_found"}'

        try:
            agg_df = (
                self.df.groupby(groupby_column)[value_column]
                .agg(agg_operation)
                .reset_index()
            )

            if sort_descending:
                agg_df = agg_df.sort_values(by=value_column, ascending=False)

            return agg_df.head(limit).round(2).to_csv(index=False, sep="|")

        except Exception as e:
            return f"{'error': '{str(e)}'}"
