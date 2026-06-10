import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

DARK_TEMPLATE = "plotly_dark"
PAPER_BG = "#0a0a0f"
PLOT_BG = "#0a0a0f"
COLORS = ["#00d4ff", "#7b2fff", "#ff6b6b", "#00ff9d", "#ffb347", "#ff69b4"]


def _style(fig, title):
    fig.update_layout(
        title=dict(text=title, font=dict(family="Space Mono", size=13, color="#aaaacc")),
        paper_bgcolor=PAPER_BG,
        plot_bgcolor=PLOT_BG,
        font=dict(family="DM Sans", color="#888899"),
        margin=dict(l=40, r=40, t=50, b=40),
        legend=dict(bgcolor="#14141f", bordercolor="#2a2a3a", borderwidth=1),
    )
    fig.update_xaxes(gridcolor="#1a1a2a", zerolinecolor="#2a2a3a")
    fig.update_yaxes(gridcolor="#1a1a2a", zerolinecolor="#2a2a3a")
    return fig


def auto_visualize(df):
    """Generate smart visualizations based on data types."""
    charts = []
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    # 1. Distribution of numeric columns
    if len(num_cols) >= 1:
        col = num_cols[0]
        fig = px.histogram(df, x=col, nbins=30, template=DARK_TEMPLATE,
                           color_discrete_sequence=[COLORS[0]],
                           title=f"Distribution of {col}")
        fig = _style(fig, f"📊 Distribution — {col}")
        charts.append(fig)

    # 2. Bar chart for categorical column
    if len(cat_cols) >= 1 and len(num_cols) >= 1:
        cat = cat_cols[0]
        num = num_cols[0]
        top = df.groupby(cat)[num].sum().nlargest(10).reset_index()
        fig = px.bar(top, x=cat, y=num, template=DARK_TEMPLATE,
                     color_discrete_sequence=[COLORS[1]],
                     title=f"Top 10 {cat} by {num}")
        fig = _style(fig, f"🏆 Top 10 {cat} by {num}")
        charts.append(fig)

    # 3. Correlation heatmap
    if len(num_cols) >= 3:
        corr = df[num_cols[:8]].corr()
        fig = go.Figure(data=go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.index,
            colorscale=[[0, "#7b2fff"], [0.5, "#0a0a0f"], [1, "#00d4ff"]],
            text=np.round(corr.values, 2),
            texttemplate="%{text}",
        ))
        fig = _style(fig, "🔗 Correlation Heatmap")
        fig.update_layout(title=dict(text="🔗 Correlation Heatmap", font=dict(family="Space Mono", size=13, color="#aaaacc")))
        charts.append(fig)

    # 4. Scatter plot
    if len(num_cols) >= 2:
        x_col, y_col = num_cols[0], num_cols[1]
        color_col = cat_cols[0] if cat_cols else None
        fig = px.scatter(df.head(500), x=x_col, y=y_col, color=color_col,
                         template=DARK_TEMPLATE,
                         color_discrete_sequence=COLORS,
                         opacity=0.7)
        fig = _style(fig, f"🔵 {x_col} vs {y_col}")
        charts.append(fig)

    # 5. Pie chart for categorical
    if len(cat_cols) >= 1:
        cat = cat_cols[0]
        counts = df[cat].value_counts().head(8).reset_index()
        counts.columns = [cat, "count"]
        fig = px.pie(counts, names=cat, values="count",
                     template=DARK_TEMPLATE,
                     color_discrete_sequence=COLORS,
                     hole=0.4)
        fig = _style(fig, f"🥧 {cat} Distribution")
        charts.append(fig)

    # 6. Line chart (if numeric cols exist — treat index as x)
    if len(num_cols) >= 1:
        col = num_cols[0]
        fig = px.line(df.head(100), y=col, template=DARK_TEMPLATE,
                      color_discrete_sequence=[COLORS[0]])
        fig = _style(fig, f"📈 {col} Trend (first 100 rows)")
        charts.append(fig)

    if not charts:
        # Fallback: just show value counts of first column
        counts = df[df.columns[0]].value_counts().head(10).reset_index()
        counts.columns = ["value", "count"]
        fig = px.bar(counts, x="value", y="count", template=DARK_TEMPLATE,
                     color_discrete_sequence=[COLORS[0]])
        fig = _style(fig, f"📊 {df.columns[0]} Value Counts")
        charts.append(fig)

    return charts
