import pandas as pd
from pydantic import BaseModel, Field
from crewai.tools import BaseTool
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


class ClusterSummary(BaseModel):
    cluster_id: int
    size: int
    centers: dict[str, float]


class ClusteringTool(BaseTool):
    name: str = "Segment Data"
    description: str = (
        "Applies K-Means clustering on numeric columns to segment data. Returns cluster centroids and sizes."
    )
    df: pd.DataFrame = Field(...)

    def _run(self, feature_columns: list[str], n_clusters: int = 3) -> str:
        missing_cols = [c for c in feature_columns if c not in self.df.columns]
        if missing_cols:
            return f"{'error': 'columns_not_found', 'missing': {missing_cols}}"

        try:
            data = self.df[feature_columns].dropna()
            if data.empty:
                return '{"error": "empty_data_after_dropping_nas"}'

            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(data)

            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
            kmeans.fit(scaled_data)

            original_scale_centers = scaler.inverse_transform(kmeans.cluster_centers_)

            results = []
            cluster_counts = pd.Series(kmeans.labels_).value_counts().to_dict()

            for i in range(n_clusters):
                center_dict = {
                    feat: round(float(original_scale_centers[i][j]), 2)
                    for j, feat in enumerate(feature_columns)
                }
                results.append(
                    ClusterSummary(
                        cluster_id=i, size=cluster_counts.get(i, 0), centers=center_dict
                    )
                )

            return f"[{','.join(r.model_dump_json() for r in results)}]"

        except Exception as e:
            return f"{'error': '{str(e)}'}"
