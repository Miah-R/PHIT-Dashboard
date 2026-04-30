import dash
from dash import dcc, html, Input, Output, State, callback_context
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
import json
import urllib.request
import ssl
from shapely.geometry import shape, Point
import requests
import numpy as np

# SSL context for geojson download
ssl_context = ssl.create_default_context()
ssl_context.check_hostname = False
ssl_context.verify_mode = ssl.CERT_NONE

# Helper functions
def normalize_fips(v):
    try:
        return f"{int(float(v)):05d}"
    except:
        return None

def parse_val(v):
    try:
        if v is None:
            return None
        s = str(v).strip()
        return float(s) if s else None
    except:
        return None

def compute_regression(x, y):
    x = np.array(x)
    y = np.array(y)

    # Remove NaNs properly
    mask = ~np.isnan(x) & ~np.isnan(y)
    x = x[mask]
    y = y[mask]

    if len(x) < 2:
        return None

    # Linear regression
    slope, intercept = np.polyfit(x, y, 1)

    # Predictions
    y_pred = slope * x + intercept

    # R^2
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

    return slope, intercept, r2, x, y, y_pred

# Categorical water-violation values -> numeric code for choropleth
WATERVIO_MAP = {"no": 0, "yes": 1}

def encode_watervio(series):
    """Convert Yes/No/NaN series to numeric codes for choropleth."""
    return series.map(lambda v: WATERVIO_MAP.get(str(v).strip().lower(), None))

# Load comorbidity data
def load_comorb():
    data = []
    with open("comorbidities2.csv", encoding="utf-8-sig") as f:
        df = pd.read_csv(f)
        # Normalize column names
        df.columns = [c.lower().strip() for c in df.columns]
        print(f"[comorbidities.csv] Columns detected: {df.columns.tolist()}")
        
        # Find FIPS column
        fips_col = None
        for col in df.columns:
            if 'fips' in col or 'co_id' in col:
                fips_col = col
                break
        
        # Find county column
        county_col = None
        for col in df.columns:
            if 'county' in col:
                county_col = col
                break
        
        for idx, row in df.iterrows():
            fips = normalize_fips(row.get(fips_col))
            county = row.get(county_col, "")
            if pd.isna(county):
                county = ""
            county = str(county).strip()
            
            if not fips or not county:
                continue
            
            def find_col(cols, year_prefix, keyword):
                """Find a column matching year prefix and keyword (case-insensitive)."""
                for c in cols:
                    if c.startswith(str(year_prefix)) and keyword in c:
                        return c
                return None

            cols = df.columns.tolist()

            def safe_val(col_name):
                return parse_val(row[col_name]) if col_name and col_name in row.index else None

            record = {
                "county": county,
                "fips": fips,
                "uninsured_2018": safe_val(find_col(cols, 18, "uninsured")),
                "uninsured_2019": safe_val(find_col(cols, 19, "uninsured")),
                "uninsured_2020": safe_val(find_col(cols, 20, "uninsured")),
                "watervio_2018": str(row[find_col(cols, 18, "watervio")]).strip() if find_col(cols, 18, "watervio") else None,
                "watervio_2019": str(row[find_col(cols, 19, "watervio")]).strip() if find_col(cols, 19, "watervio") else None,
                "watervio_2020": str(row[find_col(cols, 20, "watervio")]).strip() if find_col(cols, 20, "watervio") else None,
                "airpol_2018": safe_val(find_col(cols, 18, "airpol")),
                "airpol_2019": safe_val(find_col(cols, 19, "airpol")),
                "airpol_2020": safe_val(find_col(cols, 20, "airpol")),
                "somecol_2018": safe_val(find_col(cols, 18, "somecol")),
                "somecol_2019": safe_val(find_col(cols, 19, "somecol")),
                "somecol_2020": safe_val(find_col(cols, 20, "somecol")),
            }
            data.append(record)
    
    print(f"Loaded {len(data)} comorbidity records")
    return data

# Load cancer data
def load_cancer():
    data = []
    with open("incd_2.csv", encoding="utf-8-sig") as f:
        df = pd.read_csv(f)
        df.columns = [c.lower().strip() for c in df.columns]
        
        # Find FIPS column
        fips_col = None
        for col in df.columns:
            if 'fips' in col or 'co_id' in col:
                fips_col = col
                break
        
        for idx, row in df.iterrows():
            fips = normalize_fips(row.get(fips_col))
            if not fips:
                continue
            
            record = {"fips": fips}
            
            # Extract cancer metrics
            for col in df.columns:
                if 'cancer' in col:
                    if 'all' in col or 'total' in col:
                        record["ALLCancer"] = parse_val(row[col])
                    elif 'colon' in col:
                        record["COLONCancer"] = parse_val(row[col])
                    elif 'lung' in col:
                        record["LUNGCancer"] = parse_val(row[col])
                    elif 'breast' in col:
                        record["BREASTCancer"] = parse_val(row[col])
                    elif 'prostate' in col:
                        record["PROSTATECancer"] = parse_val(row[col])
            
            # If specific metrics not found, try to find any cancer column
            if len(record) == 1:  # Only fips
                for col in df.columns:
                    if 'cancer' in col:
                        record["ALLCancer"] = parse_val(row[col])
                        break
            
            data.append(record)
    
    print(f"Loaded {len(data)} cancer records")
    return data

# Load GeoJSON
def load_geo():
    url = "https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json"
    response = requests.get(url, verify=False)
    geo = response.json()
    
    # Filter to Mississippi only (FIPS starts with 28)
    geo["features"] = [f for f in geo["features"] if str(f["id"]).zfill(5).startswith("28")]
    return geo

# Load all data
comorb_data = load_comorb()
cancer_data = load_cancer()
geojson = load_geo()

# Merge datasets
cancer_dict = {c["fips"]: c for c in cancer_data}
merged_data = []
for r in comorb_data:
    row = r.copy()
    cancer_info = cancer_dict.get(r["fips"], {})
    row["ALLCancer"] = cancer_info.get("ALLCancer", 0)
    row["COLONCancer"] = cancer_info.get("COLONCancer", 0)
    row["LUNGCancer"] = cancer_info.get("LUNGCancer", 0)
    row["BREASTCancer"] = cancer_info.get("BREASTCancer", 0)
    row["PROSTATECancer"] = cancer_info.get("PROSTATECancer", 0)
    merged_data.append(row)

# Create DataFrame for easier manipulation
df = pd.DataFrame(merged_data)

# Define metrics
COMORB_METRICS = {
    "uninsured": {"label": "Uninsured Rate (%)", "years": [2018, 2019, 2020]},
    "watervio": {"label": "Water Violations", "years": [2018, 2019, 2020]},
    "airpol": {"label": "Air Pollution Index", "years": [2018, 2019, 2020]},
    "somecol": {"label": "Some College Attainment (%)", "years": [2018, 2019, 2020]},
}

CANCER_METRICS = {
    "ALLCancer": {"label": "All Cancers (Rate per 100k)", "years": [2018, 2019, 2020]},
    "COLONCancer": {"label": "Colon Cancer (Rate per 100k)", "years": [2018, 2019, 2020]},
    "LUNGCancer": {"label": "Lung Cancer (Rate per 100k)", "years": [2018, 2019, 2020]},
    "BREASTCancer": {"label": "Breast Cancer (Rate per 100k)", "years": [2018, 2019, 2020]},
    "PROSTATECancer": {"label": "Prostate Cancer (Rate per 100k)", "years": [2018, 2019, 2020]},
}

ALL_METRICS = {**COMORB_METRICS, **CANCER_METRICS}

# Initialize Dash app
app = dash.Dash(__name__, title="Mississippi Health Dashboard")

# App layout
app.layout = html.Div([
    html.Div([
        # Sidebar
        html.Div([
            html.H2("Mississippi Health Dashboard", style={"marginBottom": "30px"}),
            
            html.Div([
                html.Label("Select Metric:", style={"fontWeight": "bold", "marginBottom": "10px", "display": "block", "color": "#1C2446"}),
                dcc.Dropdown(
                    id="metric-dropdown",
                    options=[{"label": meta["label"], "value": metric} for metric, meta in ALL_METRICS.items()],
                    value="uninsured",
                    clearable=False,
                    style={"marginBottom": "80px"}
                ),
            ]),
            
            html.Div([
                html.Label("Select Year:", style={"fontWeight": "bold", "marginBottom": "10px", "display": "block"}),
                dcc.Slider(
                    id="year-slider",
                    min=2018,
                    max=2020,
                    step=1,
                    value=2019,
                    marks={2018: "2018", 2019: "2019", 2020: "2020"},
                    tooltip={"placement": "bottom", "always_visible": True}
                ),
            ], style={"marginBottom": "30px"}),
            
            html.Div([
                html.H4("About", style={"marginBottom": "10px"}),
                html.P("Click on any county in the map or table to view detailed trends."),
                html.P("Data source: County Health Rankings and Roadmaps (www.countyhealthrankings.org) & NIH State Cancer Profiles (statecancerprofiles.cancer.gov)", style={"fontSize": "12px", "color": "#bdc4c4"}),
            ], style={"marginTop": "50px", "padding": "15px", "backgroundColor": "#1C2446", "borderRadius": "5px"}),
            
        ], style={
            "width": "300px",
            "backgroundColor": "#2c3e50",
            "color": "white",
            "padding": "20px",
            "height": "100vh",
            "position": "fixed",
            "left": 0,
            "top": 0,
            "overflowY": "auto",
            "zIndex": 1000
        }),
        
        # Main content
        html.Div([
            html.Div([
                html.H3("All Cancer Per 100K Choropleth Map", style={"marginBottom": "10px"}),
                dcc.Graph(id="choropleth-map", config={"displayModeBar": True}, style={"height": "500px"}),
            ], style={"marginBottom": "20px"}),
            
            html.Div([
                html.Div([
                    html.H3("Time Series Trend", style={"marginBottom": "10px"}),
                    html.Div(id="selected-county-info", style={"marginBottom": "10px", "color": "#555"}),
                    dcc.Graph(id="trend-chart", config={"displayModeBar": True}),
                ], style={"width": "48%", "display": "inline-block", "verticalAlign": "top"}),
                
                html.Div([
                    html.H3("County Data Table", style={"marginBottom": "10px"}),
                    dcc.Input(
                        id="table-search",
                        type="text",
                        placeholder="Search county...",
                        style={"width": "100%", "padding": "8px", "marginBottom": "10px", "borderRadius": "4px", "border": "1px solid #ddd"}
                    ),
                    html.Div(id="county-table", style={"height": "350px", "overflowY": "auto"}),
                ], style={"width": "48%", "display": "inline-block", "float": "right"}),
            ]),
        ], style={
            "marginLeft": "350px",
            "padding": "20px",
            "backgroundColor": "#f5f6fa",
            "minHeight": "100vh"
        }),
    ]),
    
    # Store components for callback state
    dcc.Store(id="selected-fips", data=None),
    dcc.Store(id="click-data", data=None),
])

# Callback for choropleth map
@app.callback(
    Output("choropleth-map", "figure"),
    Input("metric-dropdown", "value"),
    Input("year-slider", "value")
)
def update_map(metric, year):
    """Update choropleth map based on selected metric and year"""
    customdata = [[df.iloc[i]['county'], df.iloc[i]['fips']] for i in range(len(df))]    

    # Get values for each county
    if metric in COMORB_METRICS:
        col_name = f"{metric}_{year}"

        if col_name not in df.columns:
            print(f"Missing column: {col_name}")
            values = [0] * len(df)
        else:
            if metric == "watervio":
                values = df[col_name].tolist()
            else:
                values = df[col_name].apply(lambda x: x if x is not None else 0).tolist()
    else:
        if metric not in df.columns:
            print(f"Missing cancer column: {metric}")
            values = [0] * len(df)
        else:
            values = df[metric].apply(lambda x: x if x is not None else 0).tolist()

 

    if metric == "watervio":
        encoded = encode_watervio(pd.Series(values))
        label_vals = pd.Series(values).map(lambda v: str(v).strip() if v else "N/A")
        fig = go.Figure(go.Choroplethmapbox(
            geojson=geojson,
            locations=df["fips"],
            z=encoded.fillna(-1),
            colorscale=[
                [0.0,  "#cccccc"],
                [0.25, "#cccccc"],
                [0.25, "#4daf4a"],
                [0.5,  "#4daf4a"],
                [0.5,  "#e41a1c"],
                [1.0,  "#e41a1c"],
            ],
            zmin=-1, zmax=1,
            marker_opacity=0.8,
            marker_line_width=0.5,
            colorbar=dict(
                title="Water Violations",
                tickvals=[-1, 0, 1],
                ticktext=["Unknown", "No", "Yes"],
                lenmode="fraction", len=0.4,
            ),
            customdata=customdata,
            hovertemplate="<b>%{customdata[0]}</br><b>FIPS: %{customdata[1]}</br><br>Water Violations: %{customdata}<extra></extra>"
        ))
        fig.data[0].customdata = [[c[0], c[1], label_vals.iloc[i]] for i, c in enumerate(customdata)]

    else:
        fig = go.Figure(go.Choroplethmapbox(
            geojson=geojson,
            locations=df["fips"],
            z=values,
            colorscale="Purples",
            marker_opacity=0.7,
            marker_line_width=0.5,
            colorbar_title=ALL_METRICS[metric]["label"],
            customdata=customdata,
            hovertemplate="<b>%{customdata[0]}</br><b>FIPS: %{customdata[1]}</br><br>" +
                          ALL_METRICS[metric]["label"] + ": %{z:.2f}<br>" +
                          "<extra></extra>"
        ))
    
    fig.update_layout(
        mapbox_style="carto-positron",
        mapbox_zoom=5.5,
        mapbox_center={"lat": 32.7, "lon": -89.7},
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        height=500
    )
    
    return fig

# Callback for table with search
@app.callback(
    Output("county-table", "children"),
    Input("metric-dropdown", "value"),
    Input("year-slider", "value"),
    Input("table-search", "value")
)
def update_table(metric, year, search_term):
    """Update county data table with search functionality"""
    
    # Prepare data
    if metric in COMORB_METRICS:
        col_name = f"{metric}_{year}"
        if metric == "watervio":
            values = df[col_name]  # keep as strings (Yes/No)
        else:
            values = pd.to_numeric(df[col_name], errors="coerce").fillna(0)
    else:
        col_name = metric
        values = pd.to_numeric(df[col_name], errors="coerce").fillna(0)
    
    table_df = pd.DataFrame({
        "County": df["county"],
        "FIPS": df["fips"],
        "Value": values
    })
    
    # Filter by search term
    if search_term and search_term.strip():
        table_df = table_df[
            table_df["County"].str.contains(search_term, case=False, na=False)
        ]
    
    # Sort by county name
    table_df = table_df.sort_values("County")
    
    # Create HTML table
    table_rows = []
    for idx, row in table_df.iterrows():
        
        # Safe display formatting
        if metric == "watervio":
            display_val = str(row["Value"]) if pd.notna(row["Value"]) else "N/A"
        else:
            try:
                display_val = f"{float(row['Value']):.2f}"
            except:
                display_val = "N/A"
        
        table_rows.append(
            html.Tr([
                html.Td(row["County"], style={"padding": "8px", "borderBottom": "1px solid #ddd"}),
                html.Td(row["FIPS"], style={"padding": "8px", "borderBottom": "1px solid #ddd"}),
                html.Td(display_val, style={"padding": "8px", "borderBottom": "1px solid #ddd", "textAlign": "right"}),
            ], style={"cursor": "pointer"}, id={"type": "table-row", "index": row["FIPS"]})
        )
    
    return html.Table([
        html.Thead(html.Tr([
            html.Th("County", style={"padding": "8px", "textAlign": "left", "backgroundColor": "#34495e", "color": "white", "position": "sticky", "top": 0 }),
            html.Th("FIPS", style={
                "padding": "8px",
                "textAlign": "left",
                "backgroundColor": "#34495e",
                "color": "white",
                "position": "sticky",
                "top": 0
            }),
            html.Th(ALL_METRICS[metric]["label"], style={"padding": "8px", "textAlign": "right","backgroundColor": "#34495e", "color": "white", "position": "sticky", "top": 0 }),
        ])),
        html.Tbody(table_rows)
    ], style={"width": "100%", "borderCollapse": "collapse"})

# Callback for click interaction (map or table)
@app.callback(
    Output("selected-fips", "data"),
    Output("click-data", "data"),
    Input("choropleth-map", "clickData"),
    Input({"type": "table-row", "index": dash.ALL}, "n_clicks"),
    prevent_initial_call=True
)
def handle_click(map_click, table_clicks):
    """Handle clicks on map or table to select a county"""
    ctx = callback_context
    if not ctx.triggered:
        return dash.no_update, dash.no_update
    
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
    
    # Handle map click
    if trigger_id == "choropleth-map" and map_click:
        points = map_click.get("points", [])
        if points:
            location = points[0].get("location")
            if location:
                return location, {"source": "map", "fips": location}
    
    # Handle table row click
    elif "table-row" in trigger_id:
        # Find which row was clicked
        for i, clicks in enumerate(table_clicks):
            if clicks and clicks > 0:
                return table_clicks[i], {"source": "table", "fips": table_clicks[i]}
    
    return dash.no_update, dash.no_update

# Callback for trend chart
@app.callback(
    Output("trend-chart", "figure"),
    Output("selected-county-info", "children"),
    Input("selected-fips", "data"),
    Input("metric-dropdown", "value")
)
def update_trend(selected_fips, metric):
    """Update trend chart for selected county"""
    
    if not selected_fips:
        return go.Figure(), "Click on a county to see trend"
    
    # Get county data
    county_data = df[df["fips"] == selected_fips]
    if county_data.empty:
        return go.Figure(), "County not found"
    
    county_name = county_data.iloc[0]["county"]
    
    # Prepare trend data
    years = [2018, 2019, 2020]
    
    if metric in COMORB_METRICS:
        values = [
            county_data.iloc[0].get(f"{metric}_2018", 0),
            county_data.iloc[0].get(f"{metric}_2019", 0),
            county_data.iloc[0].get(f"{metric}_2020", 0)
        ]
    else:
        # Cancer metrics are constant across years
        value = county_data.iloc[0].get(metric, 0)
        values = [value, value, value]
    
    # Create trend chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=years,
        y=values,
        mode="lines+markers",
        name=ALL_METRICS[metric]["label"],
        line=dict(color="#e74c3c", width=3),
        marker=dict(size=8, color="#c0392b")
    ))
    
    fig.update_layout(
        title=f"{county_name} County - {ALL_METRICS[metric]['label']}",
        xaxis_title="Year",
        yaxis_title=ALL_METRICS[metric]["label"],
        hovermode="closest",
        height=350,
        margin=dict(l=40, r=40, t=60, b=40)
    )
    
    info_text = f"Selected: {county_name} County (FIPS: {selected_fips})"
    
    return fig, info_text
html.Div([
    html.H3("Regression: Comorbidity vs Cancer"),
    dcc.Graph(id="regression-chart")
], style={"marginTop": "20px"})

@app.callback(
    Output("regression-chart", "figure"),
    Input("metric-dropdown", "value"),
    Input("year-slider", "value")
)
def update_regression(metric, year):

    if metric not in COMORB_METRICS:
        return go.Figure()

    x_col = f"{metric}_{year}"
    y_col = "ALLCancer"

    if x_col not in df.columns or y_col not in df.columns:
        return go.Figure()

    reg_df = df[[x_col, y_col, "county"]].copy()

    reg_df[x_col] = pd.to_numeric(reg_df[x_col], errors="coerce")
    reg_df[y_col] = pd.to_numeric(reg_df[y_col], errors="coerce")

    reg_df = reg_df.dropna()

    if reg_df.empty:
        return go.Figure()

    x = reg_df[x_col].values
    y = reg_df[y_col].values

    result = compute_regression(x, y)
    if result is None:
        return go.Figure()

    slope, intercept, r2, x, y, y_pred = result

    fig = go.Figure()

    # Scatter points
    fig.add_trace(go.Scatter(
        x=x,
        y=y,
        mode="markers",
        text=reg_df["county"],
        hovertemplate="<b>%{text}</b><br>X: %{x:.2f}<br>Y: %{y:.2f}<extra></extra>",
        marker=dict(size=8, color="#3498db")
    ))

    # Regression line
    fig.add_trace(go.Scatter(
        x=x,
        y=y_pred,
        mode="lines",
        line=dict(color="red", width=3),
        name="Regression Line"
    ))

    fig.update_layout(
        title=f"{COMORB_METRICS[metric]['label']} vs Cancer Rate",
        xaxis_title=COMORB_METRICS[metric]["label"],
        yaxis_title="Cancer Rate",
        height=400,
        annotations=[
            dict(
                x=0.05,
                y=0.95,
                xref="paper",
                yref="paper",
                text=f"y = {slope:.3f}x + {intercept:.3f}<br>R² = {r2:.3f}",
                showarrow=False,
                align="left",
                bgcolor="white"
            )
        ]
    )

    return fig

# Run the app
if __name__ == "__main__":
    print("Starting Dash server...")
    print("Make sure 'comorbidities.csv' and 'incd_2.csv' are in the same directory")
    print("Open http://127.0.0.1:8050 in your browser")
    app.run(debug=True, port=8050)