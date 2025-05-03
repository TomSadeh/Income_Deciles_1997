"""
Streamlit App for Household Income Distribution Visualizations
------------------------------------------------------------
This Streamlit app displays visualizations of household income distribution
across deciles from 1997-2022, showing how the distribution has changed over time.

The app includes:
- Static visualizations (line charts, bar charts, heatmaps)
- Interactive charts
- Animated visualizations
- Data tables and statistics

To run:
streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from matplotlib.colors import LinearSegmentedColormap
import base64
from io import BytesIO
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time

# Set page configuration
st.set_page_config(
    page_title="Household Income Distribution Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Add custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.8rem;
        font-weight: bold;
        color: #26A69A;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #5E35B1;
        margin-top: 1.5rem;
        margin-bottom: 0.8rem;
    }
    .caption {
        font-size: 0.9rem;
        font-style: italic;
        color: #78909C;
        margin-top: 0.2rem;
    }
    .highlight {
        background-color: #F9FBE7;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 0.5rem solid #CDDC39;
    }
    .card {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #F5F5F5;
        margin-bottom: 1rem;
    }
    .metric-card {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #E8F5E9;
        text-align: center;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Helper functions
def load_data(file_path):
    """Load the ratio data from CSV file"""
    try:
        # Skip initial comment lines if present
        with open(file_path, 'r') as f:
            line = f.readline()
            skip_rows = 0
            while line.startswith('#'):
                skip_rows += 1
                line = f.readline()
        
        # Load data
        data = pd.read_csv(file_path, skiprows=skip_rows)
        
        # Clean column names
        data.rename(columns={data.columns[0]: 'year'}, inplace=True)
        
        return data
    except FileNotFoundError:
        st.error(f"Error: File {file_path} not found.")
        return None

def get_gif_html(gif_path):
    """Convert GIF to HTML for Streamlit display"""
    try:
        with open(gif_path, "rb") as file:
            contents = file.read()
            data_url = base64.b64encode(contents).decode("utf-8")
        
        return f'<img src="data:image/gif;base64,{data_url}" alt="Animated visualization" style="width: 100%;">'
    except FileNotFoundError:
        return f"<p>Animation not found at {gif_path}</p>"

def calculate_inequality_metrics(data):
    """Calculate inequality metrics for the data"""
    decile_cols = [col for col in data.columns if col.startswith('decile_')]
    
    # Calculate metrics
    metrics = pd.DataFrame(index=data['year'])
    
    # Top 10% to bottom 10% ratio
    metrics['top10_to_bottom10'] = data['decile_10'] / data['decile_1']
    
    # Top 20% to bottom 20% ratio
    metrics['top20_to_bottom20'] = (data['decile_9'] + data['decile_10']) / (data['decile_1'] + data['decile_2'])
    
    # Palma ratio (top 10% to bottom 40%)
    metrics['palma_ratio'] = data['decile_10'] / (data['decile_1'] + data['decile_2'] + data['decile_3'] + data['decile_4'])
    
    # Weighted average decile (center of mass)
    weighted_avg = []
    for _, row in data.iterrows():
        decile_values = np.array([float(col.split('_')[1]) for col in decile_cols])
        weights = row[decile_cols].values
        avg = np.sum(decile_values * weights) / np.sum(weights)
        weighted_avg.append(avg)
    
    metrics['weighted_avg_decile'] = weighted_avg
    
    return metrics

def create_plotly_line_chart(data, title, y_axis_title):
    """Create a Plotly line chart for decile trends"""
    decile_cols = [col for col in data.columns if col.startswith('decile_')]
    
    fig = go.Figure()
    
    # Color scale
    colors = px.colors.sequential.Viridis
    color_scale = [colors[int(i * (len(colors)-1) / (len(decile_cols)-1))] for i in range(len(decile_cols))]
    
    # Add traces for each decile
    for i, col in enumerate(decile_cols):
        decile_num = col.split('_')[1]
        fig.add_trace(go.Scatter(
            x=data['year'], 
            y=data[col],
            mode='lines+markers',
            name=f'Decile {decile_num}',
            line=dict(color=color_scale[i], width=3),
            marker=dict(size=8)
        ))
    
    # Add equal distribution reference line
    fig.add_trace(go.Scatter(
        x=[data['year'].min(), data['year'].max()],
        y=[0.1, 0.1],
        mode='lines',
        name='Equal Distribution (10%)',
        line=dict(color='grey', width=2, dash='dash'),
    ))
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title='Year',
        yaxis_title=y_axis_title,
        yaxis_tickformat='.0%',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        hovermode="x unified",
        margin=dict(l=20, r=20, t=50, b=20),
    )
    
    return fig

def create_plotly_heatmap(data, title):
    """Create a Plotly heatmap for decile distribution"""
    decile_cols = [col for col in data.columns if col.startswith('decile_')]
    
    # Prepare data for heatmap
    z_data = data[decile_cols].values
    x_labels = [f'Decile {col.split("_")[1]}' for col in decile_cols]
    y_labels = data['year'].astype(int).astype(str).tolist()
    
    # Create figure
    fig = go.Figure(data=go.Heatmap(
        z=z_data,
        x=x_labels,
        y=y_labels,
        colorscale='RdBu_r',
        zmid=0.1,  # Center colorscale at equal distribution value
        text=[[f'{val:.1%}' for val in row] for row in z_data],
        texttemplate='%{text}',
        textfont={"size": 10},
    ))
    
    # Update layout
    fig.update_layout(
        title=title,
        yaxis=dict(
            title='Year',
            autorange='reversed'  # To show earliest year at the top
        ),
        margin=dict(l=20, r=20, t=50, b=20),
    )
    
    return fig

def create_plotly_area_chart(data, title, y_axis_title):
    """Create a Plotly stacked area chart"""
    decile_cols = [col for col in data.columns if col.startswith('decile_')]
    
    fig = go.Figure()
    
    # Color scale
    colors = px.colors.sequential.RdBu_r
    color_scale = [colors[int(i * (len(colors)-1) / (len(decile_cols)-1))] for i in range(len(decile_cols))]
    
    # Add traces for each decile (in reverse order for proper stacking)
    for i, col in enumerate(reversed(decile_cols)):
        decile_num = col.split('_')[1]
        fig.add_trace(go.Scatter(
            x=data['year'], 
            y=data[col],
            mode='lines',
            name=f'Decile {decile_num}',
            line=dict(width=0.5, color=color_scale[i]),
            fill='tonexty',
            stackgroup='one',
        ))
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title='Year',
        yaxis_title=y_axis_title,
        yaxis=dict(
            tickformat='.0%',
            range=[0, 1]
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        hovermode="x unified",
        margin=dict(l=20, r=20, t=50, b=20),
    )
    
    return fig

def create_plotly_bar_comparison(data, title):
    """Create a Plotly bar chart comparing first and last year"""
    decile_cols = [col for col in data.columns if col.startswith('decile_')]
    
    # Get first and last years
    first_year = data['year'].iloc[0]
    last_year = data['year'].iloc[-1]
    
    # Extract data for first and last years
    first_year_data = data[data['year'] == first_year][decile_cols].values.flatten()
    last_year_data = data[data['year'] == last_year][decile_cols].values.flatten()
    
    # Calculate change
    change = last_year_data - first_year_data
    
    # Create figure
    fig = go.Figure()
    
    # Add bars for first year
    fig.add_trace(go.Bar(
        x=[f'Decile {i+1}' for i in range(len(decile_cols))],
        y=first_year_data,
        name=f'{int(first_year)}',
        marker_color='lightskyblue',
        text=[f'{val:.1%}' for val in first_year_data],
        textposition='outside',
    ))
    
    # Add bars for last year
    fig.add_trace(go.Bar(
        x=[f'Decile {i+1}' for i in range(len(decile_cols))],
        y=last_year_data,
        name=f'{int(last_year)}',
        marker_color='coral',
        text=[f'{val:.1%}' for val in last_year_data],
        textposition='outside',
    ))
    
    # Add arrows or annotations to show change
    for i, change_val in enumerate(change):
        sign = '+' if change_val > 0 else ''
        color = 'green' if change_val > 0 else 'red'
        
        fig.add_annotation(
            x=i,
            y=last_year_data[i] + 0.01,
            text=f"{sign}{change_val:.1%}",
            showarrow=False,
            font=dict(
                color=color,
                size=12,
                family="Arial Black"
            )
        )
    
    # Add equal distribution reference line
    fig.add_shape(
        type="line",
        x0=-0.5,
        y0=0.1,
        x1=len(decile_cols) - 0.5,
        y1=0.1,
        line=dict(
            color="grey",
            width=2,
            dash="dash",
        )
    )
    
    fig.add_annotation(
        x=len(decile_cols) - 1,
        y=0.1,
        text="Equal distribution (10%)",
        showarrow=False,
        font=dict(color="grey"),
        xanchor="right",
    )
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title='Decile',
        yaxis_title='Proportion of Households',
        yaxis=dict(
            tickformat='.0%',
            range=[0, max(max(first_year_data), max(last_year_data)) * 1.15]
        ),
        barmode='group',
        bargap=0.15,
        bargroupgap=0.1,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    
    return fig

def create_plotly_inequality_metrics(metrics, title):
    """Create a Plotly line chart for inequality metrics"""
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add traces for each metric except weighted_avg_decile
    fig.add_trace(
        go.Scatter(
            x=metrics.index, 
            y=metrics['top10_to_bottom10'],
            mode='lines+markers',
            name='Top 10% / Bottom 10%',
            line=dict(color='crimson', width=3),
            marker=dict(size=8)
        ),
        secondary_y=False,
    )
    
    fig.add_trace(
        go.Scatter(
            x=metrics.index, 
            y=metrics['top20_to_bottom20'],
            mode='lines+markers',
            name='Top 20% / Bottom 20%',
            line=dict(color='navy', width=3),
            marker=dict(size=8)
        ),
        secondary_y=False,
    )
    
    fig.add_trace(
        go.Scatter(
            x=metrics.index, 
            y=metrics['palma_ratio'],
            mode='lines+markers',
            name='Palma Ratio (Top 10% / Bottom 40%)',
            line=dict(color='darkorange', width=3),
            marker=dict(size=8)
        ),
        secondary_y=False,
    )
    
    # Add weighted average decile on secondary axis
    fig.add_trace(
        go.Scatter(
            x=metrics.index, 
            y=metrics['weighted_avg_decile'],
            mode='lines+markers',
            name='Weighted Avg Decile',
            line=dict(color='darkgreen', width=3, dash='dot'),
            marker=dict(size=8)
        ),
        secondary_y=True,
    )
    
    # Add horizontal reference line at 5.5 for weighted average
    fig.add_trace(
        go.Scatter(
            x=[metrics.index.min(), metrics.index.max()],
            y=[5.5, 5.5],
            mode='lines',
            name='Equal Distribution (5.5)',
            line=dict(color='grey', width=2, dash='dash'),
        ),
        secondary_y=True,
    )
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title='Year',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        hovermode="x unified",
        margin=dict(l=20, r=20, t=50, b=20),
    )
    
    # Set y-axes titles
    fig.update_yaxes(title_text="Ratio (Higher = More Inequality)", secondary_y=False)
    fig.update_yaxes(title_text="Weighted Average Decile", secondary_y=True, range=[1, 10])
    
    return fig

def create_animated_decile_chart(data):
    """Create an animated chart for Streamlit"""
    decile_cols = [col for col in data.columns if col.startswith('decile_')]
    years = data['year'].unique()
    
    # Create a base figure that will be updated
    fig = go.Figure(
        layout=go.Layout(
            updatemenus=[
                dict(
                    type="buttons",
                    buttons=[
                        dict(
                            label="Play",
                            method="animate",
                            args=[None, {"frame": {"duration": 800, "redraw": True}, "fromcurrent": True}]
                        ),
                        dict(
                            label="Pause",
                            method="animate",
                            args=[[None], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}]
                        ),
                    ],
                    direction="left",
                    pad={"r": 10, "t": 10},
                    showactive=False,
                    x=0.1,
                    xanchor="right",
                    y=0,
                    yanchor="top"
                )
            ],
            sliders=[{
                "steps": [
                    {
                        "method": "animate",
                        "label": str(int(year)),
                        "args": [[str(int(year))], {
                            "mode": "immediate",
                            "frame": {"duration": 500, "redraw": True},
                            "transition": {"duration": 300}
                        }]
                    }
                    for year in years
                ],
                "active": 0,
                "x": 0.1,
                "y": 0,
                "currentvalue": {
                    "prefix": "Year: ",
                    "visible": True,
                    "xanchor": "right"
                },
                "transition": {"duration": 300},
                "pad": {"b": 10, "t": 50},
                "len": 0.9,
                "xanchor": "left"
            }]
        )
    )
    
    # Add frames for animation
    frames = []
    for year in years:
        year_data = data[data['year'] == year]
        
        # Extract data for this year
        values = year_data[decile_cols].values.flatten()
        
        frame = go.Frame(
            data=[
                go.Bar(
                    x=[f"Decile {i+1}" for i in range(len(decile_cols))],
                    y=values,
                    text=[f"{val:.1%}" for val in values],
                    textposition="outside",
                    marker_color=[
                        'green' if val > 0.1 else 'red' for val in values
                    ],
                    name=str(int(year))
                )
            ],
            name=str(int(year))
        )
        frames.append(frame)
    
    fig.frames = frames
    
    # Add initial data
    initial_data = data[data['year'] == years[0]]
    initial_values = initial_data[decile_cols].values.flatten()
    
    fig.add_trace(
        go.Bar(
            x=[f"Decile {i+1}" for i in range(len(decile_cols))],
            y=initial_values,
            text=[f"{val:.1%}" for val in initial_values],
            textposition="outside",
            marker_color=[
                'green' if val > 0.1 else 'red' for val in initial_values
            ]
        )
    )
    
    # Add horizontal line at 0.1
    fig.add_shape(
        type="line",
        x0=-0.5,
        y0=0.1,
        x1=len(decile_cols) - 0.5,
        y1=0.1,
        line=dict(
            color="grey",
            width=2,
            dash="dash",
        )
    )
    
    fig.add_annotation(
        x=len(decile_cols) - 1,
        y=0.1,
        text="Equal distribution (10%)",
        showarrow=False,
        font=dict(color="grey"),
        xanchor="right",
    )
    
    # Update layout
    fig.update_layout(
        title="Household Distribution by 1997 Deciles (Animated)",
        xaxis_title="Decile",
        yaxis_title="Proportion of Households",
        yaxis=dict(
            tickformat='.0%',
            range=[0, data[decile_cols].max().max() * 1.15]
        ),
        margin=dict(l=20, r=20, t=100, b=80),
    )
    
    return fig

def main():
    """Main function for the Streamlit app"""
    
    st.markdown('<div class="main-header">Household Income Distribution Analysis (1997-2022)</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="highlight">
    This app visualizes how household income distribution in Israel has changed over time, 
    using 1997 income deciles adjusted for inflation as a constant reference point.
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.header("Navigation")
    page = st.sidebar.radio(
        "Select a page",
        ["Overview", "Net Income Analysis", "Expenditure Analysis", "Inequality Metrics", "Animated Visualizations", "Data Tables"]
    )
    
    # Load data
    try:
        net_data = load_data('net_ratio_in_1997_deciles.csv')
        c3_data = load_data('c3_ratio_in_1997_deciles.csv')
        
        # Calculate inequality metrics
        net_metrics = calculate_inequality_metrics(net_data)
        c3_metrics = calculate_inequality_metrics(c3_data)
        
        # Check for visualization directory
        vis_dir = 'visualizations'
        has_visualizations = os.path.exists(vis_dir)
        
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()
    
    # Overview page
    if page == "Overview":
        st.markdown('<div class="sub-header">Overview of Household Income Distribution</div>', unsafe_allow_html=True)
        
        st.markdown("""
        This analysis examines how household income distribution in Israel has evolved from 1997 to 2022. 
        Instead of using each year's own deciles, we use the 1997 deciles (adjusted for inflation) as a constant 
        reference point. This allows us to see how the overall distribution has shifted over time.
        
        ### Key Questions This Analysis Answers:
        
        1. Has the proportion of households in higher income brackets increased over time?
        2. Are more households falling into what were considered lower income brackets in 1997?
        3. How has income inequality changed from 1997 to 2022?
        4. Which income deciles have grown or shrunk the most?
        """)
        
        st.markdown('<div class="section-header">Summary of Findings</div>', unsafe_allow_html=True)
        
        # Calculate key metrics for summary
        first_year = int(net_data['year'].iloc[0])
        last_year = int(net_data['year'].iloc[-1])
        
        # Net income key changes
        net_first_d1 = net_data[net_data['year'] == first_year]['decile_1'].values[0]
        net_last_d1 = net_data[net_data['year'] == last_year]['decile_1'].values[0]
        net_d1_change = (net_last_d1 - net_first_d1) / net_first_d1 * 100
        
        net_first_d10 = net_data[net_data['year'] == first_year]['decile_10'].values[0]
        net_last_d10 = net_data[net_data['year'] == last_year]['decile_10'].values[0]
        net_d10_change = (net_last_d10 - net_first_d10) / net_first_d10 * 100
        
        # Metrics for weighted average decile
        net_first_wavg = net_metrics['weighted_avg_decile'].iloc[0]
        net_last_wavg = net_metrics['weighted_avg_decile'].iloc[-1]
        net_wavg_change = net_last_wavg - net_first_wavg
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric(
                "Bottom Decile Change", 
                f"{net_last_d1:.1%}", 
                f"{net_d1_change:.1f}%",
                delta_color="inverse" if net_d1_change > 0 else "normal"
            )
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric(
                "Top Decile Change", 
                f"{net_last_d10:.1%}", 
                f"{net_d10_change:.1f}%"
            )
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric(
                "Weighted Avg Decile", 
                f"{net_last_wavg:.2f}", 
                f"{net_wavg_change:+.2f}"
            )
            st.markdown('</div>', unsafe_allow_html=True)
            
        # Summary visualization
        st.markdown('<div class="section-header">Distribution Overview</div>', unsafe_allow_html=True)
        
        # Bar comparison chart
        fig = create_plotly_bar_comparison(
            net_data, 
            f"Changes in Household Distribution by 1997 Net Income Deciles ({first_year} vs {last_year})"
        )
        st.plotly_chart(fig, use_container_width=True)
            
        st.markdown('<div class="caption">This chart shows how the distribution has changed from the beginning to the end of the study period.</div>', unsafe_allow_html=True)
        
        # Explanation of metrics
        st.markdown('<div class="section-header">Understanding the Metrics</div>', unsafe_allow_html=True)
        
        st.markdown("""
        - **Decile Ratios**: The proportion of households that fall into each of the 1997 income deciles (adjusted for inflation).
        - **Weighted Average Decile**: Represents where the "center of mass" of the distribution is. A value above 5.5 indicates the distribution is skewed toward higher incomes.
        - **Top-to-Bottom Ratios**: Measure inequality by comparing the proportion of households in top deciles vs. bottom deciles.
        
        Navigate through the sections in the sidebar to explore detailed visualizations and analysis.
        """)
        
    # Net Income Analysis page
    elif page == "Net Income Analysis":
        
        st.markdown('<div class="caption">This chart shows how the proportion of households in each 1997 decile has changed over time. Rising lines indicate growing proportions, falling lines indicate shrinking proportions.</div>', unsafe_allow_html=True)
        
        # Stacked area chart
        st.markdown('<div class="section-header">Overall Distribution Change</div>', unsafe_allow_html=True)
        fig = create_plotly_area_chart(
            net_data, 
            "Household Distribution by 1997 Net Income Deciles Over Time",
            "Proportion of Households"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown('<div class="caption">This stacked area chart shows the overall distribution across all deciles. The thickness of each band represents the proportion of households in that decile.</div>', unsafe_allow_html=True)
        
        # Heatmap
        st.markdown('<div class="section-header">Distribution Heatmap</div>', unsafe_allow_html=True)
        fig = create_plotly_heatmap(
            net_data, 
            "Heatmap of Household Distribution by 1997 Net Income Deciles"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown('<div class="caption">This heatmap shows the intensity of household concentration in each decile. Darker red indicates higher proportion than equal distribution (10%), darker blue indicates lower proportion.</div>', unsafe_allow_html=True)
        
        # Bar comparison
        st.markdown('<div class="section-header">First vs. Last Year Comparison</div>', unsafe_allow_html=True)
        first_year = int(net_data['year'].iloc[0])
        last_year = int(net_data['year'].iloc[-1])
        
        fig = create_plotly_bar_comparison(
            net_data, 
            f"Changes in Household Distribution by 1997 Net Income Deciles ({first_year} vs {last_year})"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown('<div class="caption">This chart directly compares the first and last years of the study period, showing which deciles have grown or shrunk.</div>', unsafe_allow_html=True)
        
        # Display key insights
        st.markdown('<div class="section-header">Key Insights</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="card">', unsafe_allow_html=True)
        
        # Calculate which deciles saw the biggest changes
        decile_cols = [col for col in net_data.columns if col.startswith('decile_')]
        first_year_data = net_data[net_data['year'] == net_data['year'].iloc[0]][decile_cols]
        last_year_data = net_data[net_data['year'] == net_data['year'].iloc[-1]][decile_cols]
        changes = (last_year_data.values - first_year_data.values).flatten()
        decile_changes = [(i+1, changes[i]) for i in range(len(changes))]
        
        # Sort by absolute change
        biggest_increase = sorted(decile_changes, key=lambda x: -x[1])[0]
        biggest_decrease = sorted(decile_changes, key=lambda x: x[1])[0]
        
        st.markdown(f"""
        - **Most Growth**: Decile {biggest_increase[0]} saw the largest increase ({biggest_increase[1]:.1%} points)
        - **Most Decline**: Decile {biggest_decrease[0]} saw the largest decrease ({biggest_decrease[1]:.1%} points)
        - **Overall Trend**: The weighted average decile moved from {net_metrics['weighted_avg_decile'].iloc[0]:.2f} to {net_metrics['weighted_avg_decile'].iloc[-1]:.2f}
        """)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
    # Expenditure Analysis page
    elif page == "Expenditure Analysis":
        
        # Stacked area chart
        st.markdown('<div class="section-header">Overall Distribution Change</div>', unsafe_allow_html=True)
        fig = create_plotly_area_chart(
            c3_data, 
            "Household Distribution by 1997 Expenditure Deciles Over Time",
            "Proportion of Households"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Heatmap
        st.markdown('<div class="section-header">Distribution Heatmap</div>', unsafe_allow_html=True)
        fig = create_plotly_heatmap(
            c3_data, 
            "Heatmap of Household Distribution by 1997 Expenditure Deciles"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Bar comparison
        st.markdown('<div class="section-header">First vs. Last Year Comparison</div>', unsafe_allow_html=True)
        first_year = int(c3_data['year'].iloc[0])
        last_year = int(c3_data['year'].iloc[-1])
        
        fig = create_plotly_bar_comparison(
            c3_data, 
            f"Changes in Household Distribution by 1997 Expenditure Deciles ({first_year} vs {last_year})"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Display key insights
        st.markdown('<div class="section-header">Key Insights</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="card">', unsafe_allow_html=True)
        
        # Calculate which deciles saw the biggest changes
        decile_cols = [col for col in c3_data.columns if col.startswith('decile_')]
        first_year_data = c3_data[c3_data['year'] == c3_data['year'].iloc[0]][decile_cols]
        last_year_data = c3_data[c3_data['year'] == c3_data['year'].iloc[-1]][decile_cols]
        changes = (last_year_data.values - first_year_data.values).flatten()
        decile_changes = [(i+1, changes[i]) for i in range(len(changes))]
        
        # Sort by absolute change
        biggest_increase = sorted(decile_changes, key=lambda x: -x[1])[0]
        biggest_decrease = sorted(decile_changes, key=lambda x: x[1])[0]
        
        st.markdown(f"""
        - **Most Growth**: Decile {biggest_increase[0]} saw the largest increase ({biggest_increase[1]:.1%} points)
        - **Most Decline**: Decile {biggest_decrease[0]} saw the largest decrease ({biggest_decrease[1]:.1%} points)
        - **Overall Trend**: The weighted average decile moved from {c3_metrics['weighted_avg_decile'].iloc[0]:.2f} to {c3_metrics['weighted_avg_decile'].iloc[-1]:.2f}
        """)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Compare with income
        st.markdown('<div class="section-header">Comparing Income vs. Expenditure</div>', unsafe_allow_html=True)
        
        st.markdown("""
        Comparing income and expenditure distributions can provide insights into consumption patterns and saving behaviors across different income groups.
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("<center><strong>Net Income Weighted Avg Decile</strong></center>", unsafe_allow_html=True)
            fig = px.line(
                net_metrics, y='weighted_avg_decile',
                title="Net Income Distribution Center",
                labels={'index': 'Year', 'weighted_avg_decile': 'Weighted Avg Decile'}
            )
            fig.add_hline(y=5.5, line_dash="dash", line_color="gray", annotation_text="Equal (5.5)")
            fig.update_layout(margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            st.markdown("<center><strong>Expenditure Weighted Avg Decile</strong></center>", unsafe_allow_html=True)
            fig = px.line(
                c3_metrics, y='weighted_avg_decile',
                title="Expenditure Distribution Center",
                labels={'index': 'Year', 'weighted_avg_decile': 'Weighted Avg Decile'}
            )
            fig.add_hline(y=5.5, line_dash="dash", line_color="gray", annotation_text="Equal (5.5)")
            fig.update_layout(margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(fig, use_container_width=True)
    
    # Inequality Metrics page
    elif page == "Inequality Metrics":
        st.markdown('<div class="sub-header">Inequality Metrics</div>', unsafe_allow_html=True)
        
        st.markdown("""
        This section presents various metrics that quantify income inequality based on the distribution
        of households across the 1997 deciles (adjusted for inflation).
        """)
        
        # Net Income inequality metrics
        st.markdown('<div class="section-header">Net Income Inequality Metrics</div>', unsafe_allow_html=True)
        
        fig = create_plotly_inequality_metrics(
            net_metrics,
            "Income Inequality Metrics Based on 1997 Net Income Deciles"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown('<div class="caption">This chart shows various inequality metrics over time. Rising values indicate increasing inequality, falling values indicate decreasing inequality.</div>', unsafe_allow_html=True)
        
        # Expenditure inequality metrics
        st.markdown('<div class="section-header">Expenditure Inequality Metrics</div>', unsafe_allow_html=True)
        
        fig = create_plotly_inequality_metrics(
            c3_metrics,
            "Expenditure Inequality Metrics Based on 1997 Expenditure Deciles"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Explain the metrics
        st.markdown('<div class="section-header">Understanding Inequality Metrics</div>', unsafe_allow_html=True)
        
        st.markdown("""
        - **Top 10% / Bottom 10% Ratio**: Ratio of households in the highest decile to those in the lowest decile. Higher values indicate greater inequality.
        
        - **Top 20% / Bottom 20% Ratio**: Ratio of households in the top two deciles to those in the bottom two deciles.
        
        - **Palma Ratio**: Ratio of households in the top 10% to those in the bottom 40%. This metric focuses on the extremes of the distribution.
        
        - **Weighted Average Decile**: Represents the "center of mass" of the distribution. Values above 5.5 indicate skew toward higher incomes.
        """)
        
        # Compare inequality metrics between income and expenditure
        st.markdown('<div class="section-header">Inequality: Income vs. Expenditure</div>', unsafe_allow_html=True)
        
        # Create comparison dataframe
        compare_df = pd.DataFrame({
            'Year': net_metrics.index,
            'Income Top10/Bottom10': net_metrics['top10_to_bottom10'],
            'Expenditure Top10/Bottom10': c3_metrics['top10_to_bottom10']
        })
        
        fig = px.line(
            compare_df, x='Year', y=['Income Top10/Bottom10', 'Expenditure Top10/Bottom10'],
            title="Inequality Comparison: Income vs. Expenditure",
            labels={'value': 'Ratio (Higher = More Inequality)', 'variable': 'Metric'}
        )
        fig.update_layout(legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ))
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        Comparing inequality in income versus expenditure can reveal interesting patterns:
        
        - Expenditure inequality is typically lower than income inequality, as saving rates tend to increase with income.
        - Convergence between the two may suggest changes in saving behavior or credit availability.
        - Divergence might indicate growing wealth disparities or changes in consumption patterns.
        """)
    
    # Animated Visualizations page
    elif page == "Animated Visualizations":
        st.markdown('<div class="sub-header">Animated Visualizations</div>', unsafe_allow_html=True)
        
        st.markdown("""
        This section presents animated visualizations that show how the distribution of households 
        across the 1997 deciles (adjusted for inflation) has changed over time.
        """)
        
        # Tabs for different animated visualizations
        tab1, tab2 = st.tabs(["Net Income", "Expenditure"])
        
        with tab1:
            st.markdown('<div class="section-header">Net Income Distribution (Animated)</div>', unsafe_allow_html=True)
            
            # Interactive animated chart (Plotly)
            st.markdown("### Interactive Animation")
            fig = create_animated_decile_chart(net_data)
            st.plotly_chart(fig, use_container_width=True)
            
            # Animated GIFs (if available)
            if has_visualizations:
                st.markdown("### Animated Stacked Area Chart")
                html = get_gif_html(f'{vis_dir}/net_animated_stacked_area.gif')
                st.markdown(html, unsafe_allow_html=True)
                
                st.markdown("### Animated Bar Chart")
                html = get_gif_html(f'{vis_dir}/net_animated_bar_chart.gif')
                st.markdown(html, unsafe_allow_html=True)
                
                st.markdown("### Animated Race Chart")
                html = get_gif_html(f'{vis_dir}/net_animated_race_chart.gif')
                st.markdown(html, unsafe_allow_html=True)
            else:
                st.warning("Pre-rendered animated visualizations not found. Run the visualization script to generate them.")
        
        with tab2:
            st.markdown('<div class="section-header">Expenditure Distribution (Animated)</div>', unsafe_allow_html=True)
            
            # Interactive animated chart (Plotly)
            st.markdown("### Interactive Animation")
            fig = create_animated_decile_chart(c3_data)
            st.plotly_chart(fig, use_container_width=True)
            
            # Animated GIFs (if available)
            if has_visualizations:
                st.markdown("### Animated Stacked Area Chart")
                html = get_gif_html(f'{vis_dir}/c3_animated_stacked_area.gif')
                st.markdown(html, unsafe_allow_html=True)
                
                st.markdown("### Animated Bar Chart")
                html = get_gif_html(f'{vis_dir}/c3_animated_bar_chart.gif')
                st.markdown(html, unsafe_allow_html=True)
                
                st.markdown("### Animated Race Chart")
                html = get_gif_html(f'{vis_dir}/c3_animated_race_chart.gif')
                st.markdown(html, unsafe_allow_html=True)
            else:
                st.warning("Pre-rendered animated visualizations not found. Run the visualization script to generate them.")
    
    # Data Tables page
    elif page == "Data Tables":
        st.markdown('<div class="sub-header">Data Tables</div>', unsafe_allow_html=True)
        
        st.markdown("""
        This section presents the raw data and calculated metrics used in the visualizations.
        """)
        
        # Tabs for different data types
        tab1, tab2, tab3, tab4 = st.tabs(["Net Income Ratios", "Expenditure Ratios", "Net Income Metrics", "Expenditure Metrics"])
        
        with tab1:
            st.markdown('<div class="section-header">Net Income Distribution Ratios</div>', unsafe_allow_html=True)
            st.dataframe(net_data, use_container_width=True)
            
            csv = net_data.to_csv(index=False)
            st.download_button(
                "Download Net Income Data (CSV)",
                csv,
                "net_income_ratios.csv",
                "text/csv",
                key='download-net-csv'
            )
        
        with tab2:
            st.markdown('<div class="section-header">Expenditure Distribution Ratios</div>', unsafe_allow_html=True)
            st.dataframe(c3_data, use_container_width=True)
            
            csv = c3_data.to_csv(index=False)
            st.download_button(
                "Download Expenditure Data (CSV)",
                csv,
                "expenditure_ratios.csv",
                "text/csv",
                key='download-c3-csv'
            )
        
        with tab3:
            st.markdown('<div class="section-header">Net Income Inequality Metrics</div>', unsafe_allow_html=True)
            st.dataframe(net_metrics.reset_index(), use_container_width=True)
            
            csv = net_metrics.reset_index().to_csv(index=False)
            st.download_button(
                "Download Net Income Metrics (CSV)",
                csv,
                "net_income_metrics.csv",
                "text/csv",
                key='download-net-metrics-csv'
            )
        
        with tab4:
            st.markdown('<div class="section-header">Expenditure Inequality Metrics</div>', unsafe_allow_html=True)
            st.dataframe(c3_metrics.reset_index(), use_container_width=True)
            
            csv = c3_metrics.reset_index().to_csv(index=False)
            st.download_button(
                "Download Expenditure Metrics (CSV)",
                csv,
                "expenditure_metrics.csv",
                "text/csv",
                key='download-c3-metrics-csv'
            )
        
        # Documentation section
        st.markdown('<div class="section-header">Data Documentation</div>', unsafe_allow_html=True)
        
        st.markdown("""
        ### Ratio Data Columns:
        - **year**: Survey year
        - **decile_1** through **decile_10**: Proportion of households in each 1997 decile (adjusted for inflation)
        
        ### Metrics Columns:
        - **top10_to_bottom10**: Ratio of households in the highest decile to those in the lowest decile
        - **top20_to_bottom20**: Ratio of households in the top two deciles to those in the bottom two deciles
        - **palma_ratio**: Ratio of households in the top 10% to those in the bottom 40%
        - **weighted_avg_decile**: Weighted average decile (center of mass of the distribution)
        
        All ratios are calculated based on the 1997 decile thresholds, adjusted for inflation using 
        Israel's CPI for each subsequent year.
        """)

if __name__ == "__main__":
    main()