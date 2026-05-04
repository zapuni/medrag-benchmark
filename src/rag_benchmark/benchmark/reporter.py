from pathlib import Path

import pandas as pd
import plotly.graph_objects as go


class BenchmarkReporter:
    def __init__(self, results_dir: str = "./results") -> None:
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        (self.results_dir / "csv").mkdir(exist_ok=True)
        (self.results_dir / "charts").mkdir(exist_ok=True)

    def save_csv(self, df: pd.DataFrame, name: str) -> Path:
        path = self.results_dir / "csv" / f"{name}.csv"
        df.to_csv(path, index=False)
        print(f"Saved: {path}")
        return path

    def plot_latency_vs_recall(self, df: pd.DataFrame, title: str, filename: str):
        fig = go.Figure()
        colors = {"Flat": "gray", "HNSW": "blue", "IVF": "green", "IVF+PQ": "orange"}

        for index_type in df["index_type"].unique():
            sub = df[df["index_type"] == index_type].sort_values("recall_at_k")
            fig.add_trace(
                go.Scatter(
                    x=sub["latency_ms"],
                    y=sub["recall_at_k"],
                    mode="lines+markers",
                    name=index_type,
                    marker=dict(size=8, color=colors.get(index_type, "black")),
                    text=sub.get("params_str", [""] * len(sub)),
                    hovertemplate=(
                        f"<b>{index_type}</b><br>"
                        "Latency: %{x:.2f}ms<br>"
                        "Recall@k: %{y:.3f}<br>"
                        "%{text}"
                    ),
                )
            )

        fig.update_layout(
            title=title,
            xaxis_title="Latency (ms/query)",
            yaxis_title="Recall@k",
            legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="center", x=0.5),
        )

        img_path = self.results_dir / "charts" / f"{filename}.png"
        html_path = self.results_dir / "charts" / f"{filename}.html"
        fig.write_image(str(img_path))
        fig.write_html(str(html_path))
        print(f"Saved chart: {img_path}")
        return fig
