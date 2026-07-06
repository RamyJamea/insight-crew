import pandas as pd
import numpy as np
from typing import Literal
from pydantic import BaseModel, Field
from crewai.tools import BaseTool


class CleanAndAnalyzeTimeSeriesArgs(BaseModel):
    date_column: str = Field(..., description="The exact name of the date column.")
    value_column: str = Field(
        ..., description="The exact name of the numeric column to clean and analyze."
    )
    frequency: str = Field(
        default="ME",
        description="Resampling frequency (e.g., 'ME' for month-end, 'YE' for year-end, 'W' for weekly).",
    )
    outlier_method: Literal["nullify", "drop"] = Field(
        default="nullify",
        description="Action for outliers: 'nullify' (set to NaN, recommended for forecasting) or 'drop' (remove row).",
    )
    outlier_factor: float = Field(
        default=1.5,
        description="IQR multiplier factor. 1.5 catches mild outliers; 3.0 catches only extreme outliers.",
    )


class TimeSeriesTool(BaseTool):
    name: str = "Clean and Analyze Time Series Data"
    description: str = (
        "Cleans outliers from a time series dataset, calculates mathematical trends (slope, R², volatility), "
        "and translates the findings into a natural language observation for the agent."
    )
    args_schema: type[BaseModel] = CleanAndAnalyzeTimeSeriesArgs
    df: pd.DataFrame = Field(...)

    def _run(
        self,
        date_column: str,
        value_column: str,
        frequency: str = "ME",
        outlier_method: str = "nullify",
        outlier_factor: float = 1.5,
    ) -> str:

        if date_column not in self.df.columns or value_column not in self.df.columns:
            return f"Error: Missing columns. Available columns in dataset: {list(self.df.columns)}"

        try:
            # ==========================================
            # 1. OUTLIER CLEANING (Modifies self.df)
            # ==========================================
            self.df[value_column] = pd.to_numeric(
                self.df[value_column], errors="coerce"
            )

            q1 = self.df[value_column].quantile(0.25)
            q3 = self.df[value_column].quantile(0.75)
            iqr = q3 - q1

            lower_bound = q1 - (outlier_factor * iqr)
            upper_bound = q3 + (outlier_factor * iqr)

            outliers_mask = (self.df[value_column] < lower_bound) | (
                self.df[value_column] > upper_bound
            )
            num_outliers = int(outliers_mask.sum())

            if num_outliers > 0:
                if outlier_method == "nullify":
                    self.df.loc[outliers_mask, value_column] = np.nan
                elif outlier_method == "drop":
                    # Keep non-outliers and reassign to preserve dataframe reference
                    cleaned_df = self.df[~outliers_mask]
                    self.df.drop(self.df.index, inplace=True)
                    for col in cleaned_df.columns:
                        self.df[col] = cleaned_df[col]

            # ==========================================
            # 2. TIME SERIES MATH (On cleaned data)
            # ==========================================
            local_df = self.df[[date_column, value_column]].copy()
            local_df[date_column] = pd.to_datetime(
                local_df[date_column], errors="coerce"
            )
            local_df = (
                local_df.dropna()
            )  # Drop NaNs (including nullified outliers) for math calculations

            # Resample
            trend_df = (
                local_df.groupby(pd.Grouper(key=date_column, freq=frequency))[
                    value_column
                ]
                .sum()
                .reset_index()
            )

            if trend_df.empty or len(trend_df) < 2:
                return "Error: Insufficient data to calculate a trend after aggregation. Need at least 2 periods."

            y = trend_df[value_column].values
            x = np.arange(len(y))

            # Linear Regression Math
            slope, intercept = np.polyfit(x, y, 1)
            y_pred = slope * x + intercept
            y_mean = np.mean(y)

            # R-Squared
            ss_tot = np.sum((y - y_mean) ** 2)
            ss_res = np.sum((y - y_pred) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0

            # Volatility & Growth
            volatility = (np.std(y) / y_mean) if y_mean != 0 else 0.0
            initial_val = y[0]
            final_val = y[-1]
            global_growth = (
                ((final_val - initial_val) / initial_val * 100)
                if initial_val != 0
                else 0.0
            )

            # Peaks and Troughs
            max_idx = np.argmax(y)
            min_idx = np.argmin(y)
            peak_val = y[max_idx]
            peak_date = trend_df[date_column].iloc[max_idx].strftime("%Y-%m-%d")
            trough_val = y[min_idx]
            trough_date = trend_df[date_column].iloc[min_idx].strftime("%Y-%m-%d")

            # ==========================================
            # 3. IF/ELSE AGENT TRANSLATION
            # ==========================================

            # Outlier Translation
            if num_outliers == 0:
                outlier_text = (
                    f"No statistical outliers were found in '{value_column}'."
                )
            else:
                outlier_text = f"Found and {outlier_method}ed {num_outliers} outlier(s) outside normal bounds ({round(lower_bound, 2)} to {round(upper_bound, 2)})."

            # Trend Direction Translation
            if slope > 0.05:
                trend_dir = "an upward (growing)"
            elif slope < -0.05:
                trend_dir = "a downward (shrinking)"
            else:
                trend_dir = "a relatively flat/stagnant"

            # Predictability (R²) Translation
            if r_squared > 0.7:
                rel_text = "highly predictable and reliable"
            elif r_squared > 0.4:
                rel_text = "moderately predictable"
            else:
                rel_text = "noisy and lacks a definitive mathematical trend"

            # Volatility Translation
            if volatility > 0.5:
                vol_text = "highly volatile (expect drastic swings)"
            elif volatility > 0.2:
                vol_text = "moderately volatile"
            else:
                vol_text = "relatively stable"

            # Construct the final observation string
            observation = f"""
### Data Preparation
{outlier_text} The remaining data was grouped by '{frequency}' resulting in {len(trend_df)} total periods.

### Trend Analysis Observation
- **Overall Direction:** The dataset exhibits {trend_dir} trend. Total growth across the timespan is {global_growth:.1f}%.
- **Mathematical Slope:** The value changes by an average of {slope:.2f} units per period.
- **Predictability (R² = {r_squared:.2f}):** The trend is considered {rel_text}.
- **Volatility (Index = {volatility:.2f}):** The data behavior is {vol_text}.

### Historical Landmarks
- **Peak:** Reached a maximum of {peak_val:.2f} on {peak_date}.
- **Trough:** Hit a low of {trough_val:.2f} on {trough_date}.
            """

            return observation.strip()

        except Exception as e:
            return f"Error executing analysis: {str(e)}"
