import pandas as pd
from crewai.tools import BaseTool
from pydantic import BaseModel, Field


class TimeSeriesMetadata(BaseModel):
    start_date: str
    end_date: str
    periods: int
    overall_growth_pct: float | None


class TimeSeriesResult(BaseModel):
    metadata: TimeSeriesMetadata
    data_csv: str


class TimeSeriesTool(BaseTool):
    name: str = "Analyze Time Series Trend"
    description: str = (
        "Aggregates a metric over time intervals and calculates period-over-period growth. Returns strict JSON."
    )
    df: pd.DataFrame = Field(...)

    def _run(
        self,
        date_column: str,
        value_column: str,
        frequency: str = "ME",
        limit: int = 12,
    ) -> str:
        if date_column not in self.df.columns or value_column not in self.df.columns:
            return '{"error": "columns_not_found"}'

        try:
            self.df[date_column] = pd.to_datetime(self.df[date_column])
            trend_df = (
                self.df.groupby(pd.Grouper(key=date_column, freq=frequency))[
                    value_column
                ]
                .sum()
                .reset_index()
            )

            if trend_df.empty:
                return '{"error": "empty_time_series"}'

            trend_df["pop_growth_pct"] = (
                trend_df[value_column].pct_change().fillna(0) * 100
            )
            trend_df[date_column] = trend_df[date_column].dt.strftime("%Y-%m-%d")

            initial_val = trend_df[value_column].iloc[0]
            final_val = trend_df[value_column].iloc[-1]
            overall_growth = (
                ((final_val - initial_val) / initial_val * 100)
                if initial_val != 0
                else None
            )

            metadata = TimeSeriesMetadata(
                start_date=trend_df[date_column].iloc[0],
                end_date=trend_df[date_column].iloc[-1],
                periods=len(trend_df),
                overall_growth_pct=round(overall_growth, 2) if overall_growth else None,
            )

            csv_data = trend_df.tail(limit).round(2).to_csv(index=False, sep="|")

            result = TimeSeriesResult(metadata=metadata, data_csv=csv_data)
            return result.model_dump_json(exclude_none=True)

        except Exception as e:
            return f"{'error': '{str(e)}'}"
