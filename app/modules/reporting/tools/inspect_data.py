import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from crewai.tools import BaseTool


class DataProfile(BaseModel):
    rows: int
    cols: int
    dtypes: dict[str, list[str]]
    missing: dict[str, int]
    stats: dict[str, dict[str, float]] | None


class InspectionTool(BaseTool):
    name: str = "Inspect Data Profile"
    description: str = (
        "Returns strict JSON containing schema, dtypes, missing values, and summary statistics."
    )
    df: pd.DataFrame = Field(...)

    def _run(self) -> str:
        if self.df.empty:
            return '{"error": "empty_dataframe"}'

        dtype_dict = {}
        for col, dtype in self.df.dtypes.items():
            dtype_dict.setdefault(str(dtype), []).append(col)

        missing = self.df.isnull().sum()
        missing_dict = missing[missing > 0].to_dict()

        numeric_df = self.df.select_dtypes(include=np.number)
        stats_dict = None
        if not numeric_df.empty:

            stats_dict = (
                numeric_df.describe()
                .T[["min", "mean", "max"]]
                .round(2)
                .to_dict(orient="index")
            )

        profile = DataProfile(
            rows=self.df.shape[0],
            cols=self.df.shape[1],
            dtypes=dtype_dict,
            missing=missing_dict,
            stats=stats_dict,
        )

        return profile.model_dump_json(exclude_none=True)
