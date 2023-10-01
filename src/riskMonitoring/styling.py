plot_layout = dict(
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1,
        
    ),
    autosize=True,
    yaxis_title=None,
    xaxis_title=None,
    margin=dict(l=10, r=10, t=10, b=10),
    uniformtext_mode='hide',
    font=dict(size=16),

)

barplot_trace = dict(
    marker_line_width=0,
    selector=dict(type="bar"),
)
card_style = dict(
    flex='1 1 calc(100% / 3)',
)
