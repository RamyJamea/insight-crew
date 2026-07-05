import pandas as pd
from pydantic import BaseModel, Field
from crewai.tools import BaseTool
from statsmodels.tsa.holtwinters import ExponentialSmoothing


class ForecastResult(BaseModel):
    forecast_periods: int
    predictions: dict[str, float]


class ForecastTool(BaseTool):
    name: str = "Forecast Time Series"
    description: str = (
        "Predicts future values for a time series using Exponential Smoothing. Returns forecasted data points."
    )
    df: pd.DataFrame = Field(...)

    def _run(
        self,
        date_column: str,
        value_column: str,
        periods_to_forecast: int = 3,
        frequency: str = "ME",
    ) -> str:
        if date_column not in self.df.columns or value_column not in self.df.columns:
            return '{"error": "columns_not_found"}'

        try:
            # Prepare time series data
            ts_df = self.df[[date_column, value_column]].copy()
            ts_df[date_column] = pd.to_datetime(ts_df[date_column])
            ts_df = ts_df.groupby(pd.Grouper(key=date_column, freq=frequency))[
                value_column
            ].sum()

            if len(ts_df) < 4:
                return '{"error": "insufficient_data_for_forecasting"}'

            model = ExponentialSmoothing(
                ts_df, trend="add", seasonal=None, initialization_method="estimated"
            )
            fit_model = model.fit()

            forecast = fit_model.forecast(periods_to_forecast)

            predictions_dict = {
                date.strftime("%Y-%m-%d"): round(float(val), 2)
                for date, val in forecast.items()
            }

            result = ForecastResult(
                forecast_periods=periods_to_forecast, predictions=predictions_dict
            )
            return result.model_dump_json()

        except Exception as e:
            return f"{'error': '{str(e)}'}"
