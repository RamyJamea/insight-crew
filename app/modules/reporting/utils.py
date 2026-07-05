import pandas as pd


def convert_to_df(payload: dict) -> pd.DataFrame:
    data_list = payload.get("dataSrcList", [])
    if not data_list:
        raise ValueError("The 'dataSrcList' in the payload is empty or missing.")

    columns_info = payload.get("cloumns", payload.get("columns", []))

    visible_cols_mapping = {
        col["field"]: col["headerText"]
        for col in columns_info
        if col.get("visible") is True
    }

    df = pd.DataFrame(data_list)

    cols_to_keep = [col for col in visible_cols_mapping.keys() if col in df.columns]
    df = df[cols_to_keep]

    for col in df.columns:
        if df[col].dtype == "object":
            numeric_converted = pd.to_numeric(df[col], errors="ignore")
            if numeric_converted.dtype != "object":
                df[col] = numeric_converted
                continue

        col_lower = col.lower()
        if "date" in col_lower or "time" in col_lower or col.endswith("_clk"):
            try:
                df[col] = pd.to_datetime(
                    df[col], format="mixed", dayfirst=True, errors="coerce"
                )
            except Exception:
                pass
        elif df[col].dtype == "object":
            try:
                converted = pd.to_datetime(
                    df[col], format="mixed", dayfirst=True, errors="coerce"
                )
                if converted.notna().sum() > (len(df) * 0.70):
                    df[col] = converted
            except Exception:
                pass

    df = df.rename(columns=visible_cols_mapping)

    return df
