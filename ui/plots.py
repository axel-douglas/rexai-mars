# ui/plots.py
import plotly.io as pio
import plotly.graph_objects as go

def use_plotly_theme():
    tmpl = go.layout.Template(
        layout=go.Layout(
            font=dict(family="Inter, system-ui, -apple-system, Segoe UI, Roboto", size=14, color="#101316"),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="#FFFFFF",
            colorway=["#2FD3C6", "#7C3AED", "#17B26A", "#F59E0B", "#EF4444", "#5E6B78"],
            margin=dict(l=40, r=20, t=30, b=40),
            xaxis=dict(showgrid=True, gridcolor="#EEF1F4", zeroline=False, ticks="outside"),
            yaxis=dict(showgrid=True, gridcolor="#EEF1F4", zeroline=False, ticks="outside"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        )
    )
    pio.templates["rex"] = tmpl
    pio.templates.default = "plotly_white+rex"
