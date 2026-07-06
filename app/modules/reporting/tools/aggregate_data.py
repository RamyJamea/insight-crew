import pandas as pd
from crewai.tools import BaseTool
from pydantic import Field


class AggregationTool(BaseTool):
    name: str = "Aggregate Data"
    description: str = (
        "Aggregates a numeric column grouped by one or multiple categorical columns. "
        "For entity analysis, always pass both the ID and Name columns together. Returns pipe-delimited data."
    )
    df: pd.DataFrame = Field(...)

    def _run(
        self,
        groupby_columns: list[str],
        value_column: str,
        agg_operation: str = "sum",
        sort_descending: bool = True,
        limit: int = 10,
    ) -> str:
        missing_cols = [
            col
            for col in groupby_columns + [value_column]
            if col not in self.df.columns
        ]
        if missing_cols:
            return f'{{"error": "columns_not_found", "missing": {missing_cols}}}'

        try:
            agg_df = self.df.groupby(groupby_columns, as_index=False)[value_column].agg(
                agg_operation
            )

            if sort_descending:
                agg_df = agg_df.sort_values(by=value_column, ascending=False)

            return agg_df.head(limit).round(2).to_csv(index=False, sep="|")

        except Exception as e:
            return f"{'error': '{str(e)}'}"
