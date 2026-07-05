import pandas as pd
from crewai.tools import BaseTool
from pydantic import BaseModel, Field


class ColumnRemovalResult(BaseModel):
    dropped_columns: list[str]
    ignored_columns: list[str]
    remaining_columns_count: int


class RemoveColumnsTool(BaseTool):
    name: str = "Remove Irrelevant Columns"
    description: str = (
        "Removes specified columns from the dataset. Returns JSON metadata of the action."
    )
    df: pd.DataFrame = Field(...)

    def _run(self, columns_to_remove: list[str]) -> str:
        if self.df.empty:
            return '{"error": "empty_dataframe"}'

        existing_cols = set(self.df.columns)
        valid_cols = [c for c in columns_to_remove if c in existing_cols]
        missing_cols = [c for c in columns_to_remove if c not in existing_cols]

        if not valid_cols:
            return ColumnRemovalResult(
                dropped_columns=[],
                ignored_columns=missing_cols,
                remaining_columns_count=len(self.df.columns),
            ).model_dump_json()

        try:
            self.df.drop(columns=valid_cols, inplace=True)

            result = ColumnRemovalResult(
                dropped_columns=valid_cols,
                ignored_columns=missing_cols,
                remaining_columns_count=len(self.df.columns),
            )
            return result.model_dump_json()

        except Exception as e:
            return f"{'error': '{str(e)}'}"
