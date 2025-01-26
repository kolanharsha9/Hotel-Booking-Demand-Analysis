from dash import dcc, html, Input, Output
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import dash_daq as daq
import dash as dash
import dash_bootstrap_components as dbc
import plotly.express as px
import numpy as np
from dash.dependencies import Input, Output
import pandas as pd
from scipy.stats import normaltest, kstest, shapiro
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
import numpy as np
from dash import Dash, dcc, html, Input, Output, dash_table, State
import pandas as pd
import dash
from dash import dcc, html, Input, Output
import plotly.graph_objs as go
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
import dash_bootstrap_components as dbc






my_app = dash.Dash('My app')
server=my_app.server
my_app.layout = html.Div([html.H1(children='Hotel Booking Demand Analysis App',
                                  style={'textAlign':'center'}),

                          html.Br(),
                          dcc.Tabs(id = 'hw-questions',
                                   children=[

                                       dcc.Tab(label='Outlier Removal', value='q2'),
                                        dcc.Tab(label='Noramlity Tests', value='q3'),
                                       dcc.Tab(label='PCA Analysis', value='q4'),
                                       dcc.Tab(label='Analysis between variables using Plots',value='q5'),
dcc.Tab(label='Choropleth Map',value='q6'),

dcc.Tab(label='Other Plots',value='q7'),
dcc.Tab(label='Other plots',value='q8')


                                   ]
                                   ),
                          html.Div(id='layout')

]
)





df1=pd.read_csv('hotel_pre.csv')
df2=df1.copy(deep=True)
numerical_columns = ['special_requests_per_person', 'lead_time_per_night', 'adr_per_person', 'booking_to_arrival_ratio',
                     'adr']



tab2_layout = html.Div([
    html.H2("Outlier Detection and Removal", style={'textAlign': 'center'}),

    html.Div([
        html.Label("Select a numerical column:"),
        dcc.Dropdown(
            id='column-dropdown',
            options=[{'label': col, 'value': col} for col in numerical_columns],
            value=numerical_columns[0],  # Default value
            clearable=False
        ),
    ], style={'marginBottom': 30}),

    html.Div([
        dcc.RadioItems(
            id='plot-type',
            options=[
                {'label': 'Box Plot', 'value': 'boxplot'},
                {'label': 'Histogram', 'value': 'histogram'},
            ],
            value='boxplot',  # Default value
            labelStyle={'display': 'inline-block', 'marginRight': '10px'}
        ),
    ], style={'marginBottom': 30}),

    html.Div(id='plot-container')
])


@my_app.callback(
    Output('plot-container', 'children'),
    [Input('column-dropdown', 'value'),
     Input('plot-type', 'value')]
)
def update_plot(selected_column, plot_type):
    # Apply IQR outlier removal method to the selected column
    q1 = np.percentile(df2[selected_column], 25)
    q3 = np.percentile(df2[selected_column], 75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    # Filter out outliers
    cleaned_df = df2[(df1[selected_column] >= lower_bound) & (df2[selected_column] <= upper_bound)]

    # Create the appropriate plot based on the selected plot type
    if plot_type == 'boxplot':
        fig_before = px.box(df2, y=selected_column)
        fig_before.update_layout(title=dict(text=f'Box Plot - Before Outlier Removal', y=0.95, x=0.5))
        fig_after = px.box(cleaned_df, y=selected_column)
        fig_after.update_layout(title=dict(text=f'Box Plot - After Outlier Removal', y=0.95, x=0.5))
    elif plot_type == 'histogram':
        fig_before = px.histogram(df2, x=selected_column)
        fig_before.update_layout(title=dict(text=f'Histogram - Before Outlier Removal', y=0.95, x=0.5))
        fig_after = px.histogram(cleaned_df, x=selected_column)
        fig_after.update_layout(title=dict(text=f'Histogram - After Outlier Removal', y=0.95, x=0.5))

    return html.Div([
        dcc.Graph(figure=fig_before),
        dcc.Graph(figure=fig_after)
    ])
df = pd.read_csv('hotel_pre.csv')

# Select numerical columns
numerical_columns = ['lead_time_per_night', 'adr_per_person', 'booking_to_arrival_ratio',
                     'adr']
tab3_layout= html.Div([
    html.H2("Normality Tests and Plots to show Normality Distribution", style={'textAlign': 'center'}),

    html.Div([
        html.Label("Select a numerical column:"),
        dcc.Dropdown(
            id='column-dropdown',
            options=[{'label': col, 'value': col} for col in numerical_columns],
            value=numerical_columns[0],  # Default value
            clearable=False
        ),
    ], style={'marginBottom': 30}),

    html.Div([
        dcc.RadioItems(
            id='test-type',
            options=[
                {'label': 'D\'Agostino & Pearson Test', 'value': 'd_agostino'},
                {'label': 'Kolmogorov-Smirnov Test', 'value': 'kstest'},
                {'label': 'Shapiro-Wilk Test', 'value': 'shapiro'},
            ],
            value='d_agostino',  # Default value
            labelStyle={'display': 'block', 'marginBottom': '10px'}
        ),
    ], style={'marginBottom': 30}),
    html.Br(),

html.Div(id='output-container', children=[]),
html.Br(),
html.Br(),

    html.Div([
        dcc.RadioItems(
            id='plot-type',
            options=[
                {'label': 'QQ Plot', 'value': 'qqplot'},
                {'label': 'Histogram', 'value': 'histogram'},
            ],
            value='qqplot',  # Default value
            labelStyle={'display': 'block', 'marginBottom': '10px'}
        ),
    ], style={'marginBottom': 30}),

   dcc.Graph(id='graph-normality')
])


@my_app.callback(
    [Output('output-container', 'children'),
Output('graph-normality', 'figure')],
    [Input('column-dropdown', 'value'),
     Input('test-type', 'value'),
     Input('plot-type', 'value')]
)
def run_normality_test(selected_column, test_type, plot_type):
    # Apply the selected normality test
    if test_type == 'd_agostino':
        statistic, p_value = normaltest(df[selected_column])
        test_name = "D'Agostino & Pearson Test"
    elif test_type == 'kstest':
        statistic, p_value = kstest(df[selected_column], 'norm',
                                    args=(df[selected_column].mean(), df[selected_column].std()))
        test_name = "Kolmogorov-Smirnov Test"
    elif test_type == 'shapiro':
        statistic, p_value = shapiro(df[selected_column])
        test_name = "Shapiro-Wilk Test"

    # Determine normality based on the p-value
    normality = 'Normal' if p_value > 0.01 else 'Not Normal'

    # Generate plot based on user selection
    if plot_type == 'qqplot':
        qq = stats.probplot(df[selected_column], dist='norm', sparams=(1))

        # Create the Q-Q plot using Plotly
        x = np.array([qq[0][0][0], qq[0][0][-1]])
        fig = go.Figure()
        fig.add_scatter(x=qq[0][0], y=qq[0][1], mode='markers', name="Data Quantiles")
        fig.add_scatter(x=x, y=qq[1][1] + qq[1][0] * x, mode='lines', name="Standardized Line")
        fig.update_layout(title=dict(text=f'QQ Plot of {selected_column}', y=0.95, x=0.5),xaxis_title='Theoretical Quantiles',yaxis_title='Sample Quantiles'),


    elif plot_type == 'histogram':
        fig = px.histogram(df, x=selected_column)
        fig.update_layout(title=dict(text=f'Histogram Plot of {selected_column}', y=0.95, x=0.5))

    return ((f' For the {selected_column} variable: '
            f' Test Statistic: {round(statistic,2)},'
            f' p-value: {round(p_value,2)}'),
            fig)
# Read the CSV file
df = pd.read_csv('hotel_pre.csv')

# Select numerical columns
numerical_columns = ['special_requests_per_person', 'lead_time_per_night', 'adr_per_person', 'booking_to_arrival_ratio',
                     'adr']

# Extract numerical features
df_numerical = df[numerical_columns]
tab4_layout=html.Div([
    html.H2("PCA Analysis of the Numerical Features", style={'textAlign': 'center'}),



    html.Div([
        html.Label("Select explained variance threshold:"),
        dcc.Slider(
            id='explained-variance-slider',
            min=0,
            max=100,
            step=1,
            value=95,  # Default value
            marks={i: f'{i}%' for i in range(0, 101, 10)},
        ),
    ], style={'marginBottom': 30}),

    dcc.Graph(id='pca-graph'),

    html.Div(id='info-text')
])






@my_app.callback(
    Output('pca-graph', 'figure'),
    [Input('explained-variance-slider', 'value')]
)
def update_pca_graph(explained_variance):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_numerical)


    pca = PCA()
    pca.fit(X_scaled)


    var_ratio_cumsum = np.cumsum(pca.explained_variance_ratio_) * 100


    num_components = np.argmax(var_ratio_cumsum >= explained_variance) + 1


    pca_fig = go.Figure()
    pca_fig.add_trace(go.Scatter(x=np.arange(1, num_components + 1),
                                 y=var_ratio_cumsum[:num_components],
                                 mode='lines+markers',
                                 marker=dict(color='blue'),
                                 name='Cumulative Explained Variance Ratio'))

    pca_fig.update_layout(title=dict(text='Cumulative Explained Variance Ratio', y=0.95, x=0.5),
                          xaxis_title='Number of Components',
                          yaxis_title='Cumulative Explained Variance (%)',
                          margin=dict(l=0, r=0, t=50, b=0))

    return pca_fig


@my_app.callback(
    Output('info-text', 'children'),
    [Input('explained-variance-slider', 'value')]
)
def update_info_text(explained_variance):

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_numerical)


    pca = PCA()
    pca.fit(X_scaled)


    var_ratio_cumsum = np.cumsum(pca.explained_variance_ratio_) * 100


    num_components = np.argmax(var_ratio_cumsum >= explained_variance) + 1


    num_components_to_remove = pca.n_components_ - num_components


    reduced_X = pca.transform(X_scaled)[:, :-num_components_to_remove]

    
    condition_number = np.linalg.cond(reduced_X)

    return html.Div([
        html.Label(f"Number of components to remove: {num_components_to_remove}"),
        html.Br(),
        html.Label(f"Condition number of the reduced feature space: {round(condition_number,2)}")
    ])
df = pd.read_csv('hotel_pre.csv')

# Define variables
continuous = ['lead_time_per_night', 'adr_per_person', 'booking_to_arrival_ratio', 'adr']
discrete = ['stays_in_weekend_nights', 'stays_in_week_nights', 'adults', 'children', 'babies', 'days_in_waiting_list', 'required_car_parking_spaces', 'total_of_special_requests', 'deposit_type', 'reserved_room_type', 'assigned_room_type', 'market_segment', 'distribution_channel', 'meal']

tab5_layout=html.Div([
    html.H1('Plots to show relationship between different variables in the dataset', style={'textAlign': 'center', 'margin-bottom': '20px'}),
    html.Div([
        html.H3('Select the continuous variable:', style={'margin-bottom': '10px'}),
        dcc.Dropdown(id='continuous-dropdown',
                     options=[{'label': i, 'value': i} for i in continuous],
                     placeholder='Select the continuous variable',
                     style={'width': '50%', 'margin': 'auto'}),
        html.Br(),
        html.H3('Select the discrete or categorical variable:', style={'margin-bottom': '10px'}),
        dcc.Dropdown(id='discrete-dropdown',
                     options=[{'label': i, 'value': i} for i in discrete],
                     placeholder='Select the discrete variable',
                     style={'width': '50%', 'margin': 'auto'}),
        html.Br(),
        html.H3('Select the plot type:', style={'margin-bottom': '10px'}),
        dcc.RadioItems(
            id="plot-type",
            value="Box Plot",
            options=["Box Plot", 'Violin Plot', 'Scatter Plot'],
            labelStyle={'display': 'block', 'margin-bottom': '10px'}
        ),
        html.Div([
            html.Label('Include hue as "hotel"?', style={'margin-bottom': '10px'}),
            dcc.Checklist(
                id='hue-checkbox',
                options=[
                    {'label': 'Yes', 'value': 'include-hue'},
                ],
                value=[],
                style={'margin': 'auto'}
            ),
        ]),
        html.Br(),
        dcc.Graph(id='graph-3'),
    ], style={'width': '80%', 'margin': 'auto', 'border': '1px solid #ddd', 'padding': '20px', 'border-radius': '10px'}),
])

# Define callback to update the graph
@my_app.callback(
    Output('graph-3', 'figure'),
    [Input('continuous-dropdown', 'value'),
     Input('discrete-dropdown', 'value'),
     Input('plot-type', 'value'),
     Input('hue-checkbox', 'value')]
)
def update_bar_chart(continuous_var, discrete_var, plot_type, hue_checkbox):
    hue = None
    if 'include-hue' in hue_checkbox:
        hue = 'hotel'

    if plot_type == 'Box Plot':
        fig = px.box(df, x=discrete_var, y=continuous_var, color=hue)
    elif plot_type == 'Violin Plot':
        fig = px.violin(df, x=discrete_var, y=continuous_var, color=hue)
    elif plot_type == 'Scatter Plot':
        fig = px.scatter(df, x=discrete_var, y=continuous_var, color=hue)


    fig.update_layout(title=dict(text=f'{plot_type} between {continuous_var} and {discrete_var}', y=0.95, x=0.5))
    return fig
df=pd.read_csv('hotel_pre.csv')
country_counts = df[df['is_canceled'] == 0]['country'].value_counts()
country_data = pd.DataFrame(country_counts).reset_index()
country_data.columns = ['country', 'Number of Guests']

total_guests = country_data["Number of Guests"].sum()

country_data["Guests in %"] = round(country_data["Number of Guests"] / total_guests * 100, 2)
# Create the choropleth map figure
fig = px.choropleth(country_data,
                    locations=country_data['country'],
                    color=country_data["Guests in %"],
                    hover_name=country_data['country'],
                    color_continuous_scale=px.colors.sequential.Rainbow,
                    title="Home country of guests")
tab6_layout=html.Div([
    dcc.Graph(id='choropleth-map', figure=fig, style={'width':'100%', 'height': '600px'}),



    html.Label('Select color bar range:'),
    dcc.RangeSlider(
        id='color-range-slider',
        min=0,
        max=100,
        step=1,
        marks={i: str(i) for i in range(0, 101, 10)},
        value=[0, 100]
    ),
])

# Define callback to update the color bar range
@my_app.callback(
    Output('choropleth-map', 'figure'),
    [Input('color-range-slider', 'value')]
)
def update_color_range(selected_range):
    updated_fig = px.choropleth(country_data,
                                 locations=country_data['country'],
                                 color=country_data["Guests in %"],
                                 hover_name=country_data['country'],
                                 color_continuous_scale=px.colors.sequential.Rainbow,
                                 title="Home country of guests",
                                 range_color=selected_range)
    updated_fig.update_layout(title=dict(text="Home country of guests", y=0.99, x=0.5))
    return updated_fig
df = pd.read_csv('hotel_pre.csv')

# Define categorical and revenue analysis variables
categorical = ['deposit_type', 'reserved_room_type', 'assigned_room_type', 'market_segment', 'distribution_channel', 'meal']
adr_variable = ['adr', 'adr_per_person']

# Group by 'arrival_date_month' and 'country' and calculate averages
yearly_avg = df.groupby(['arrival_date_month', 'country']).agg({
    'adr': 'mean',
    'lead_time': 'mean',
    'booking_to_arrival_ratio': 'mean',
    'adr_per_person': 'mean',
}).reset_index()

# Define a dictionary mapping month names to numbers
month_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']

# Convert 'arrival_date_month' to categorical with specified order
yearly_avg['arrival_date_month'] = pd.Categorical(yearly_avg['arrival_date_month'], categories=month_order, ordered=True)

# Sort the DataFrame by 'arrival_date_month'
yearly_avg = yearly_avg.sort_values('arrival_date_month')

# Define continuous variables
continuous = ['adr_per_person', 'booking_to_arrival_ratio', 'adr', 'lead_time']

tab7_layout=html.Div([

    html.H3('Animated Bar Plot of Continuous Variables with respect to months:', style={'textAlign': 'center', 'margin-bottom': '10px'}),
    dcc.Dropdown(id='continuous-dropdown',
                 options=[{'label': i, 'value': i} for i in continuous],
                 placeholder='Select the continuous variable'),
    dcc.Loading(dcc.Graph(id="animated-graph"), type="cube"),
    html.H3('Pie Plot between Categorical and Revenue Variables:', style={'textAlign': 'center', 'margin-top': '40px', 'margin-bottom': '10px'}),
    dcc.Dropdown(id='categorical-dropdown',
                 options=[{'label': i, 'value': i} for i in categorical],
                 placeholder='Select the categorical variable'),
    dcc.Dropdown(id='adr-dropdown',
                 options=[{'label': i, 'value': i} for i in adr_variable],
                 placeholder='Select the revenue analysis variable'),
    dcc.Graph(id='pie-chart'),
], style={'width': '80%', 'margin': 'auto'})

# Define callbacks to update the graphs
@my_app.callback(
    Output("animated-graph", "figure"),
    Input("continuous-dropdown", "value")
)
def display_animated_graph(c):
    return px.bar(
        yearly_avg,
        x="country",
        y=c,
        color="country",
        animation_frame="arrival_date_month",
        animation_group="country",
        range_y=[0, yearly_avg[c].max() + yearly_avg[c].max() / 10],
        title='Animated Bar plot of Months and Continuous Variables'
    )

@my_app.callback(
    Output('pie-chart', 'figure'),
    [Input('categorical-dropdown', 'value'),
     Input('adr-dropdown', 'value')]
)
def update_pie_chart(categorical_choice, adr_choice):
    fig = px.pie(df, values=adr_choice, names=categorical_choice)
    fig.update_layout(title=dict(text=f'Pie plot between {categorical_choice} and {adr_choice}', y=0.95, x=0.5))
    return fig
df = pd.read_csv('hotel_pre.csv')
df1=df.copy(deep=True)
df1 = df1.drop_duplicates(subset=['country'])
country_list = df1['country'].unique()


# Country selection card
controls = dbc.Card(
    [
        html.Div(
            [
                dbc.Label("Select Country"),
                dcc.Dropdown(
                    id="country",
                    options=[{"label": i, "value": i} for i in country_list],
                    multi=False,
                ),
            ]
        )
    ]
)

# Graph for the first app
graph_country = dbc.Card(
    dcc.Graph(id="graph-country"),
    body=True,
    style={"margin-top": "20px"}
)

# Generate data for cancellation percentage per month
res_book_per_month = df.loc[df["hotel"] == "Resort Hotel"].groupby("arrival_date_month")["hotel"].count()
res_cancel_per_month = df.loc[df["hotel"] == "Resort Hotel"].groupby("arrival_date_month")["is_canceled"].sum()
cty_book_per_month = df.loc[df["hotel"] == "City Hotel"].groupby("arrival_date_month")["hotel"].count()
cty_cancel_per_month = df.loc[df["hotel"] == "City Hotel"].groupby("arrival_date_month")["is_canceled"].sum()

# DataFrame for cancellation data
full_cancel_data = pd.concat([
    pd.DataFrame({"Hotel": "Resort Hotel", "Month": list(res_book_per_month.index), "Bookings": list(res_book_per_month.values), "Cancelations": list(res_cancel_per_month.values)}),
    pd.DataFrame({"Hotel": "City Hotel", "Month": list(cty_book_per_month.index), "Bookings": list(cty_book_per_month.values), "Cancelations": list(cty_cancel_per_month.values)})
], ignore_index=True)
full_cancel_data["cancel_percent"] = full_cancel_data["Cancelations"] / full_cancel_data["Bookings"] * 100

# Number of arrivals per month
arrival_counts = df.groupby(["arrival_date_month", "hotel"]).size().unstack()
months_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
arrival_counts = arrival_counts.reindex(months_order)


tab8_layout=dbc.Container(
    [
        html.H3("Analysis of different variables by selecting specific country", style={"textAlign": "center"}),
        html.Hr(),
        controls,
        graph_country,
        html.H3("Arrivals and Cancellations trend through months", style={"textAlign": "center"}),
        html.P("Arrivals per month | Cancellations per month", style={"textAlign": "center"}),
        daq.BooleanSwitch(id="pb", on=True),
        html.Div(id="pb-result"),
    ]
)

# Callback for updating the first graph
@my_app.callback(Output("graph-country", "figure"), Input("country", "value"))
def make_country_graph(c):
    if c is None or len(c) == 0:
        return go.Figure()
    my_list = [c]
    dff = df1[df1.country.isin(my_list)]

    fig = go.Figure()

    for country in dff.country:
        dft = dff[dff.country == country].reset_index()
        dft.drop(columns=["special_requests_per_person"], inplace=True)
        dft = dft.iloc[:, 33:].T.astype(int)

        fig.add_trace(go.Bar(x=["total_people", "total_nights", "lead_time_per_night", "adr_per_person", "booking_to_arrival_ratio"], y=dft.iloc[:, 0], name=country))
    fig.update_layout(xaxis_tickfont_size=10,title=dict(text='Bar plot of variables per country',y=0.95,x=0.5),xaxis_title='variables',yaxis_title='values')

    return fig

# Callback for the boolean switch to update the second graph
@my_app.callback(
    Output("pb-result", "children"),
    Input("pb", "on"),
)
def update_graph(on):
    if on:
        fig = px.line(full_cancel_data, x='Month', y='cancel_percent', color='Hotel', title='Cancellation Percentage per Month by Hotel Type')
    else:
        fig = px.line(arrival_counts, title='Arrivals per Month by Hotel Type')
    fig.update(layout=dict(title=dict(x=0.5)))
    return dcc.Graph(figure=fig)
#callback for entire page
@my_app.callback(
    Output(component_id='layout',component_property='children'),
    Input(component_id='hw-questions',component_property='value')
)

def update_layout(ques):

    if ques=='q2':
        return tab2_layout
    elif ques == 'q3':
        return tab3_layout
    elif ques == 'q4':
        return tab4_layout
    elif ques == 'q5':
        return tab5_layout
    elif ques == 'q6':
        return tab6_layout
    elif ques=='q7':
        return tab7_layout
    elif ques == 'q8':
        return tab8_layout


my_app.run_server(
    port=8033,
    host='0.0.0.0'
)
