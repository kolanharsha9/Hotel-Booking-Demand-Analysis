#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.stats import shapiro
from scipy.stats import normaltest
from scipy.stats import kstest
from scipy.stats import norm
import plotly.express as px

df= pd.read_csv('hotel_bookings.csv')
unique_values_count = df.nunique()

# Identify categorical variables
categorical_variables = unique_values_count[unique_values_count <= 10]  # Assuming categorical variables have <= 10 unique values

# Get the count of categorical variables
num_categorical_variables = len(categorical_variables)
print(df.head().to_string())
print("Number of categorical variables:", num_categorical_variables)
print("Categorical variables:", categorical_variables.index.tolist())
#%%
print(df.columns)
print(df['reservation_status_date'].unique())
#%%
print(df.isna().sum())
#%%
print(df['agent'].value_counts())
#%%
#preprocessing
percent_missing = df.isna().sum() * 100 / len(df)
missing_value_df = pd.DataFrame({'column_name': df.columns,
                                 'percent_missing': percent_missing})
missing_value_df = missing_value_df.sort_values(by='percent_missing', ascending=False)

plt.figure(figsize=(25, 10))
sns.barplot(x=missing_value_df['percent_missing'], y=missing_value_df['column_name'], palette='Set1')
plt.xlabel('Percentage of Missing values', fontsize=15, fontname='serif', color='darkred')
plt.ylabel('Column', fontsize=15, fontname='serif', color='darkred')
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.grid(True, axis='x')
plt.title('Percentage of missing values in columns', fontsize=15, fontname='serif', color='blue')

plt.show()
#%%
replacements= {"children:": 0.0,"country": "Unknown", "agent": 0, "company": 0}
df = df.fillna(replacements)

df["meal"].replace("Undefined", "SC", inplace=True)

df.dropna(inplace=True)
#%%

print(df.isna().sum())
missing_values = df.isnull().any()

print("Missing values in each column:")
print(missing_values)
#%%
# Calculate Total Guests
df['total_people'] = df['adults'] + df['children'] + df['babies']

# Calculate Total Nights
df['total_nights'] = df['stays_in_weekend_nights'] + df['stays_in_week_nights']
# Calculate Total Special Requests per Person
df['special_requests_per_person'] = df['total_of_special_requests'] / df['total_people']

# Calculate Lead Time per Night
df['lead_time_per_night'] = df['lead_time'] / df['total_nights']

# Calculate Average Daily Rate per Person
df['adr_per_person'] = df['adr'] / df['total_people']

# Calculate Booking to Arrival Ratio
df['booking_to_arrival_ratio'] = df['lead_time'] / df['total_nights']
#%%
print(df.isna().sum())


# Replace missing values with the mean of each column
df['special_requests_per_person'].fillna(df['special_requests_per_person'].mean(), inplace=True)
df['lead_time_per_night'].fillna(df['lead_time_per_night'].mean(), inplace=True)
df['adr_per_person'].fillna(df['adr_per_person'].mean(), inplace=True)
df['booking_to_arrival_ratio'].fillna(df['booking_to_arrival_ratio'].mean(), inplace=True)

# Print updated DataFrame to verify changes
print(df.isna().sum())

#%%
#outlier detection
df_numerical = df[['special_requests_per_person', 'lead_time_per_night', 'adr_per_person', 'booking_to_arrival_ratio','adr']]

# Apply IQR outlier removal method to each numerical column
cleaned_df = df_numerical.copy()
for column in cleaned_df.columns:
    q1 = np.percentile(cleaned_df[column], 25)
    q3 = np.percentile(cleaned_df[column], 75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    plt.subplot(1, 2, 1)
    sns.boxplot(data=cleaned_df[column])
    plt.title('Before outlier removal', fontsize=15, fontname='serif', color='blue')
    plt.ylabel(column,fontsize=15, fontname='serif', color='darkred')


    # Filter out outliers
    cleaned_df = cleaned_df[(cleaned_df[column] >= lower_bound) & (cleaned_df[column] <= upper_bound)]
    df[column] = cleaned_df[column]

    plt.subplot(1, 2, 2)
    sns.boxplot(data=cleaned_df[column])
    plt.title('After outlier removal', fontsize=15, fontname='serif', color='blue')
    plt.ylabel(column,fontsize=15, fontname='serif', color='darkred')
    plt.tight_layout()
    plt.show()

df.dropna(inplace=True)
print(df.isna().sum())

#%%
#PCA
df_numerical = df[['special_requests_per_person', 'lead_time_per_night', 'adr_per_person', 'booking_to_arrival_ratio','adr']]


scaler = StandardScaler()
X=df_numerical
X_scaled = scaler.fit_transform(X)
print(X_scaled)
correlation_matrix = X.corr()


plt.figure(figsize=(12, 12))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)

plt.title('Correlation Coefficient Matrix between Features', fontsize=15, fontname='serif', color='blue')
plt.xticks(rotation=30)
plt.yticks(rotation=30)
plt.show()


condition_number = np.linalg.cond(X_scaled)
print("Condition number of the original feature space:", condition_number.round(2))

pca = PCA()
pca.fit(X_scaled)


var_ratio_cumsum = np.cumsum(pca.explained_variance_ratio_)
num_components = np.argmax(var_ratio_cumsum >= 0.90) + 1


pca_reduced = PCA(n_components=num_components)
X_reduced = pca_reduced.fit_transform(X_scaled)


num_features_removed = X_scaled.shape[1] - num_components
print("Number of features to be removed per PCA analysis with 90% threshold: ", num_features_removed)

print("Explained variance ratio of the original feature space:", pca.explained_variance_ratio_.round(2))
print("Explained variance ratio of the reduced feature space:", pca_reduced.explained_variance_ratio_.round(2))

condition_number = np.linalg.cond(X_reduced)
print("Condition number of the reduced feature space:", condition_number.round(2))

num_comp = np.arange(1, len(var_ratio_cumsum) + 1, 1)
cumulative_variance = np.cumsum(pca.explained_variance_ratio_) * 100


plt.plot(num_comp, cumulative_variance)


plt.xticks(num_comp)


plt.grid()


x_90 = num_comp[np.argmax(cumulative_variance >= 90)]
plt.axvline(x=x_90, color='black', linestyle='dashed')


plt.axhline(y=90, color='red', linestyle='dashed')


plt.xlabel('Number of Components', fontsize=15, fontname='serif', color='darkred')
plt.ylabel('Cumulative Explained Variance (%)', fontsize=15, fontname='serif', color='darkred')
plt.title('Explained Variance Ratio', fontsize=15, fontname='serif', color='blue')


plt.show()
#%%
#normality test
for column in df_numerical.columns:
    statistic, p_value = normaltest(df_numerical[column])
    normality = 'Normal' if p_value > 0.01 else 'Not Normal'
    print("For column {}: \n Da_k_squared test: statistics= {:.2f} p-value = {:.2f}".format(column,statistic, p_value))
    print(f"da_k_squared test: The data is {normality}")

    statistic, p_value = kstest(df_numerical[column], 'norm', args=(df_numerical[column].mean(), df_numerical[column].std()))
    normality = 'Normal' if p_value > 0.01 else 'Not Normal'

    print("For column {}: \n K_S test: statistics= {:.2f} p-value = {:.2f}".format( column,statistic, p_value))
    print(f"K_S test: The data is {normality}")

    statistic, p_value = shapiro(df_numerical[column])
    normality = 'Normal' if p_value > 0.01 else 'Not Normal'

    print("For column {}: \n Shapiro test: statistics= {:.2f} p-value = {:.2f}".format( column,statistic, p_value))
    print(f"Shapiro test: The data is {normality}")
#%%
from prettytable import PrettyTable

# Create a PrettyTable object
table = PrettyTable()

# Add columns to the table
table.field_names = ["Column", "Da_k_squared statistics", "Da_k_squared p-value",
                     "K_S statistics", "K_S p-value"
                     "Shapiro statistics", "Shapiro p-value", "Normality"]

# Add data to the table
for column in df_numerical.columns:
    # Calculate test statistics and p-values
    da_k_squared_statistic, da_k_squared_p_value = normaltest(df_numerical[column])
    k_s_statistic, k_s_p_value = kstest(df_numerical[column], 'norm',
                                        args=(df_numerical[column].mean(), df_numerical[column].std()))
    shapiro_statistic, shapiro_p_value = shapiro(df_numerical[column])

    # Determine normality
    da_k_squared_normality = 'Normal' if da_k_squared_p_value > 0.01 else 'Not Normal'
    k_s_normality = 'Normal' if k_s_p_value > 0.01 else 'Not Normal'
    shapiro_normality = 'Normal' if shapiro_p_value > 0.01 else 'Not Normal'

    # Add row to the table
    table.add_row([column,
                   f"{da_k_squared_statistic:.2f}", f"{da_k_squared_p_value:.2f}",
                   f"{k_s_statistic:.2f}", f"{k_s_p_value:.2f}"
                   f"{shapiro_statistic:.2f}", f"{shapiro_p_value:.2f}", shapiro_normality])

# Print the table
print(table)
#%%
correlation_matrix = df_numerical.corr()


# Plot scatter plot matrix
sns.pairplot(df_numerical)
plt.title('Scatter Plot Matrix', y=1.02)
plt.show()

#%%
df=pd.read_csv('hotel_pre.csv')

#line plot
arrival_counts = df.groupby(by=["arrival_date_month"]).size()

# Sorting the data by month
arrival_counts = arrival_counts.reindex(['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'])

# Plotting the data as a line plot
plt.figure(figsize=(15, 6))  # Adjust figure size if needed
arrival_counts.plot(marker='o', linestyle='-')

plt.title('Arrivals per month', fontsize=15, fontname='serif', color='blue')
plt.xlabel('Month', fontsize=15, fontname='serif', color='darkred')
plt.ylabel('Number of Arrivals', fontsize=15, fontname='serif', color='darkred')

# Adding data labels
for i, count in enumerate(arrival_counts):
    plt.text(i, count, str(count), ha='center', va='bottom')

plt.grid(True)
plt.xticks(range(len(arrival_counts)), arrival_counts.index,rotation=45)  # Rotate x-axis labels for better visibility if needed
plt.tight_layout()
plt.show()

#%%
#stacked bar plot
room_assignment_counts = df.groupby(['reserved_room_type', 'assigned_room_type']).size().reset_index(name='counts')

# Pivot the data for easier visualization
pivot_data = room_assignment_counts.pivot(index='reserved_room_type', columns='assigned_room_type', values='counts')

# Plotting
pivot_data.plot(kind='bar', stacked=True, figsize=(10, 6))
plt.title('Reserved Room Type vs Assigned Room Type', fontsize=15, fontname='serif', color='blue')
plt.xlabel('Reserved Room Type', fontsize=15, fontname='serif', color='darkred')
plt.xticks(rotation=360)
plt.ylabel('Count', fontsize=15, fontname='serif', color='darkred')
plt.legend(title='Assigned Room Type')
plt.tight_layout()
plt.show()
#%%
#grouped bar plot
count_data = df.groupby(['deposit_type', 'is_canceled']).size().reset_index(name='count')

# Plotting
plt.figure(figsize=(10, 6))
sns.barplot(data=count_data, x='deposit_type', y='count', hue='is_canceled', palette='Set2')



plt.title('Cancellation vs Deposit Type', fontsize=15, fontname='serif', color='blue')
plt.xlabel('Deposit Type', fontsize=15, fontname='serif', color='darkred')
plt.ylabel('Count', fontsize=15, fontname='serif', color='darkred')
plt.legend(title='Canceled')
plt.tight_layout()
plt.show()
#%%
#count plot
plt.figure(figsize=(10, 5))
sns.countplot(x='reservation_status', hue='is_canceled', data=df, palette=['darkturquoise', 'royalblue'])
plt.title('Count of Cancellations by Reservation Status', fontsize=15, fontname='serif', color='blue')
plt.xlabel('Reservation Status', fontsize=15, fontname='serif', color='darkred')
plt.ylabel('Count', fontsize=15, fontname='serif', color='darkred')
plt.legend(title='Is Canceled', labels=['Not Canceled', 'Canceled'])
plt.show()
#%%
#pie plot
# Separate the DataFrame into Resort Hotel and City Hotel
meal_counts = df['meal'].value_counts()


plt.figure(figsize=(10, 6))
plt.pie(meal_counts, labels=meal_counts.index, autopct='%1.1f%%', startangle=140)
plt.title('Distribution of Meals in the Hotel')
plt.axis('equal')
plt.show()
#%%
#Dist plot
plt.figure(figsize=(10, 6))
sns.distplot(df['adr'],kde=False)
plt.title('Distribution Plot of Average Daily Rate (ADR)', fontsize=15, fontname='serif', color='blue')
plt.xlabel('ADR', fontsize=15, fontname='serif', color='darkred')
plt.ylabel('Density', fontsize=15, fontname='serif', color='darkred')
plt.show()


#%%
#Hist plot with KDE
import seaborn as sns
import matplotlib.pyplot as plt


plt.figure(figsize=(10, 6))
sns.histplot(df['booking_to_arrival_ratio'], kde=True)
plt.title('Histogram Plot with KDE for Booking To Arrival Ratio', fontsize=15, fontname='serif', color='blue')
plt.xlabel('Booking To Arrival Ratio', fontsize=15, fontname='serif', color='darkred')
plt.ylabel('Density', fontsize=15, fontname='serif', color='darkred')
plt.show()
#%%
#qq plot
from statsmodels.graphics.gofplots import qqplot


plt.figure(figsize=(8, 6))
qqplot(df['adr'], line='s')
plt.title('QQ Plot for ADR', fontsize=15, fontname='serif', color='blue')
plt.xlabel('Theoretical Quantiles', fontsize=15, fontname='serif', color='darkred')
plt.ylabel('Sample Quantiles', fontsize=15, fontname='serif', color='darkred')
plt.grid(True)

#%%
#KDE plot
plt.figure(figsize=(10, 6))
sns.kdeplot(data=df['special_requests_per_person'], fill=True, alpha=0.6, palette='Set2', linewidth=2)
plt.title(f'KDE Plot for Special Requests Per Person', fontsize=15, fontname='serif', color='blue')
plt.xlabel('Special Requests Per Person', fontsize=15, fontname='serif', color='darkred')
plt.ylabel('Density', fontsize=15, fontname='serif', color='darkred')
plt.show()
#%%
#Regplot
plt.figure(figsize=(10, 6))
sns.regplot(data=df, x='lead_time_per_night', y='booking_to_arrival_ratio', scatter=True)
plt.title('Linear Regression Plot between Lead Time per Night and Booking to Arrival Ratio', fontsize=15, fontname='serif', color='blue')
plt.xlabel('Lead Time per Night', fontsize=15, fontname='serif', color='darkred')
plt.ylabel('Booking to Arrival Ratio', fontsize=15, fontname='serif', color='darkred')
plt.show()
#%%
#Boxen plot
variables = ['special_requests_per_person', 'lead_time_per_night',
                     'adr_per_person', 'booking_to_arrival_ratio']


plt.figure(figsize=(12, 8))
sns.boxenplot(data=df[variables])
plt.title('Multivariate Boxen Plot', fontsize=15, fontname='serif', color='blue')
plt.xlabel('Variables', fontsize=15, fontname='serif', color='darkred')
plt.ylabel('Values', fontsize=15, fontname='serif', color='darkred')
plt.show()
#%%
#Area plot
total_people_per_month = df.groupby('arrival_date_month')['total_people'].sum()

# Reindexing the series to ensure all months are present
total_people_per_month = total_people_per_month.reindex(['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'])


plt.figure(figsize=(10, 6))
total_people_per_month.plot.area()


plt.title('Total People Over Months', fontsize=15, fontname='serif', color='blue')
plt.xlabel('Month', fontsize=15, fontname='serif', color='darkred')
plt.ylabel('Total People', fontsize=15, fontname='serif', color='darkred')


plt.xticks(ticks=range(len(total_people_per_month)), labels=total_people_per_month.index, rotation=45)

plt.tight_layout()
plt.show()
#%%
#Violin plot
plt.figure(figsize=(10, 6))
sns.violinplot(x='market_segment', y='lead_time', data=df, palette='Set2')


plt.title('Lead Time by Market Segment', fontsize=15, fontname='serif', color='blue')
plt.xlabel('Market Segment', fontsize=15, fontname='serif', color='darkred')
plt.ylabel('Lead Time', fontsize=15, fontname='serif', color='darkred')


plt.xticks(rotation=45)


plt.tight_layout()
plt.show()


#%%
#Rug plot
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='adr', y='booking_to_arrival_ratio', s=5)
sns.rugplot(data=df, x='adr', y='booking_to_arrival_ratio', lw=1,alpha=0.3)
plt.title('Rug Plot for Booking to Arrival Ratio and ADR', fontsize=15, fontname='serif', color='blue')


plt.show()
#%%
# Create a 3D plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot
ax.scatter(df['lead_time'], df['adr'], df['booking_to_arrival_ratio'], c='b', marker='o')


ax.set_xlabel('Lead Time', fontsize=15, fontname='serif', color='darkred')
ax.set_ylabel('ADR', fontsize=15, fontname='serif', color='darkred')
ax.set_zlabel('Booking to Arrival Ratio', fontsize=15, fontname='serif', color='darkred')
ax.set_title('3D Scatter Plot', fontsize=15, fontname='serif', color='blue')


plt.show()

#%%
#Hexbin plot
plt.hexbin(df['adr_per_person'], df['booking_to_arrival_ratio'], gridsize=20, cmap='Blues', edgecolors='none')
plt.colorbar(label='count')
plt.xlabel('ADR Per person', fontsize=15, fontname='serif', color='darkred')
plt.ylabel('Booking to Arrival Ratio', fontsize=15, fontname='serif', color='darkred')
plt.title('Hexbin Plot of ADR/person & Booking:Arrival Ratio', fontsize=15, fontname='serif', color='blue')
plt.show()
#%%
# Create a strip plot
plt.figure(figsize=(10, 10))
sns.stripplot(data=df, x='arrival_date_month', y='lead_time', jitter=True)
plt.title('Strip Plot of Lead Time Across Arrival Months', fontsize=15, fontname='serif', color='blue')
plt.xlabel('Arrival Date Month', fontsize=15, fontname='serif', color='darkred')
plt.ylabel('Lead Time', fontsize=15, fontname='serif', color='darkred')
plt.xticks(rotation=30)
plt.grid(True)
plt.show()


#%%
# Filter the DataFrame and count occurrences of each country
country_counts = df[df['is_canceled'] == 0]['country'].value_counts()
country_data = pd.DataFrame(country_counts).reset_index()
country_data.columns = ['country', 'Number of Guests']
print(country_data.head())

total_guests = country_data["Number of Guests"].sum()

country_data["Guests in %"] = round(country_data["Number of Guests"] / total_guests * 100, 2)


fig = px.pie(country_data,
             values="Number of Guests",
             names="country",
             title="Home country of guests",
             template="ggplot2")
fig.update_traces(textposition="inside", textinfo="value+percent+label")
fig.show()
#%%
fig = px.choropleth(country_data,
                    locations=country_data['country'],
                    color=country_data["Guests in %"],
                    hover_name=country_data['country'],
                    color_continuous_scale=px.colors.sequential.Rainbow,
                    title="Home country of guests")
fig.update_layout(title=dict(text="Home country of guests", y=0.99, x=0.5))
fig.show()
#%%
print(df['company'].value_counts())







import plotly.express as px
resort_hotel_df=df[df['hotel']=='Resort Hotel']

# Group the DataFrame by company, count occurrences, and reset index
resort_company_counts = resort_hotel_df.groupby('company').size().reset_index(name='count')

# Sort the counts in descending order
resort_company_counts = resort_company_counts.sort_values(by='count', ascending=False)

# Select the top 10 companies
top_10_resort_companies = resort_company_counts.head(10)


fig_resort_company = px.pie(top_10_resort_companies,
                    values='count',
                    names='company',
                    title='Top 10 Companies for Resort Hotel',
                    template='ggplot2')


fig_resort_company.update_traces(textposition='inside', textinfo='value+percent+label')


fig_resort_company.show()
#%%
import plotly.express as px
city_hotel_df=df[df['hotel']=='City Hotel']

city_company_counts = city_hotel_df.groupby('company').size().reset_index(name='count')


city_company_counts = city_company_counts.sort_values(by='count', ascending=False)


top_10_city_companies = city_company_counts.head(10)


fig_city_company = px.pie(top_10_city_companies,
                  values='count',
                  names='company',
                  title='Top 10 Companies for City Hotel',
                  template='ggplot2')


fig_city_company.update_traces(textposition='inside', textinfo='value+percent+label')


fig_city_company.show()
#%%
resort_agent_counts = resort_hotel_df.groupby('agent').size().reset_index(name='count')

# Sort the counts in descending order
resort_agent_counts = resort_agent_counts.sort_values(by='count', ascending=False)


top_10_resort_agents = resort_agent_counts.head(10)


fig_resort_agent = px.pie(top_10_resort_agents,
                    values='count',
                    names='agent',
                    title='Top 10 agents for Resort Hotel',
                    template='ggplot2')


fig_resort_agent.update_traces(textposition='inside', textinfo='value+percent+label')


fig_resort_agent.show()
#%%
import plotly.express as px

# Group the DataFrame by company, count occurrences, and reset index
city_agent_counts = city_hotel_df.groupby('agent').size().reset_index(name='count')


city_agent_counts = city_agent_counts.sort_values(by='count', ascending=False)


top_10_city_agents = city_agent_counts.head(10)


fig_city_agent = px.pie(top_10_city_agents,
                  values='count',
                  names='agent',
                  title='Top 10 agents for City Hotel',
                  template='ggplot2')


fig_city_agent.update_traces(textposition='inside', textinfo='value+percent+label')


fig_city_agent.show()
#%%
from plotly.subplots import make_subplots
import plotly.graph_objects as go


specs = [[{'type':'domain'}, {'type':'domain'}], [{'type':'domain'}, {'type':'domain'}]]
fig = make_subplots(rows=2, cols=2, specs=specs)




fig.add_trace(go.Pie(
                     labels=top_10_resort_companies['company'],
                     values=top_10_resort_companies['count'],
                     title='Top 10 Companies for Resort Hotel'),

              row=1, col=1)

fig.add_trace(go.Pie(
                     labels=top_10_city_companies['company'],
                     values=top_10_city_companies['count'],
                     title='Top 10 Companies for City Hotel'),

              row=1, col=2)

fig.add_trace(go.Pie(
                     labels=top_10_resort_agents['agent'],
                     values=top_10_resort_agents['count'],
                     title='Top 10 Agents for Resort Hotel'),

              row=2, col=1)
fig.add_trace(go.Pie(
                     labels=top_10_city_agents['agent'],
                     values=top_10_city_agents['count'],
                     title='Top 10 Agents for Resort Hotel'),

              row=2, col=2)
fig.update_traces(hoverinfo='label+percent+name', textinfo='none')
fig.update_layout(
    title=dict(text='Pie Charts for Agents and Companies per Hotel Type', y=0.99, x=0.5),
    showlegend=False)
fig.show()

#%%
repeated_guests = df[df['is_repeated_guest'] == 1]

# Group by hotel type, country, and count repeated guests
repeated_guests_per_hotel_country = repeated_guests.groupby(['hotel', 'country'])['is_repeated_guest'].count().reset_index(name='repeated_guest_count')

# Sort the data by repeated guest count
repeated_guests_per_hotel_country = repeated_guests_per_hotel_country.sort_values(by='repeated_guest_count', ascending=False)


plt.figure(figsize=(14, 8))
sns.barplot(data=repeated_guests_per_hotel_country, x='country', y='repeated_guest_count', hue='hotel', palette='Set2')
plt.title('Number of Repeated Guests from Each Country by Hotel Type')
plt.xlabel('Country', fontsize=15, fontname='serif', color='darkred')
plt.ylabel('Number of Repeated Guests', fontsize=15, fontname='serif', color='darkred')
plt.xticks(rotation=90)
plt.legend(title='Hotel Type')
plt.tight_layout()
plt.show()


#%%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Assuming full_data_cln is your DataFrame containing the relevant data

# Create DataFrame with relevant data
res_book_per_month = df.loc[(df["hotel"] == "Resort Hotel")].groupby("arrival_date_month")["hotel"].count()
res_cancel_per_month = df.loc[(df["hotel"] == "Resort Hotel")].groupby("arrival_date_month")["is_canceled"].sum()

cty_book_per_month = df.loc[(df["hotel"] == "City Hotel")].groupby("arrival_date_month")["hotel"].count()
cty_cancel_per_month = df.loc[(df["hotel"] == "City Hotel")].groupby("arrival_date_month")["is_canceled"].sum()

res_cancel_data = pd.DataFrame({"Hotel": "Resort Hotel",
                                "Month": list(res_book_per_month.index),
                                "Bookings": list(res_book_per_month.values),
                                "Cancelations": list(res_cancel_per_month.values)})
cty_cancel_data = pd.DataFrame({"Hotel": "City Hotel",
                                "Month": list(cty_book_per_month.index),
                                "Bookings": list(cty_book_per_month.values),
                                "Cancelations": list(cty_cancel_per_month.values)})

full_cancel_data = pd.concat([res_cancel_data, cty_cancel_data], ignore_index=True)
full_cancel_data["cancel_percent"] = full_cancel_data["Cancelations"] / full_cancel_data["Bookings"] * 100

# Order by month
ordered_months = ["January", "February", "March", "April", "May", "June",
                  "July", "August", "September", "October", "November", "December"]
full_cancel_data["Month"] = pd.Categorical(full_cancel_data["Month"], categories=ordered_months, ordered=True)


plt.figure(figsize=(12, 8))
sns.lineplot(data=full_cancel_data, x='Month', y='cancel_percent', hue='Hotel', marker='o')
plt.title('Cancellation Percentage per Month by Hotel Type', fontsize=15, fontname='serif', color='blue')
plt.xlabel('Month', fontsize=15, fontname='serif', color='darkred')
plt.ylabel('Cancellation Percentage', fontsize=15, fontname='serif', color='darkred')
plt.xticks(rotation=45)
plt.legend(title='Hotel Type')
plt.tight_layout()
plt.grid()
plt.show()
#%%
print(df.head().to_string())
#%%
import seaborn as sns
import matplotlib.pyplot as plt

# Grouping by month and hotel type
arrival_counts = df.groupby(["arrival_date_month", "hotel"]).size().unstack()

# Reordering months
months_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
arrival_counts = arrival_counts.reindex(months_order)

plt.figure(figsize=(15, 6))


sns.lineplot(data=arrival_counts, marker='o', palette='Set1')

plt.title('Arrivals per month by Hotel Type', fontsize=15, fontname='serif', color='blue')
plt.xlabel('Month', fontsize=15, fontname='serif', color='darkred')
plt.ylabel('Number of Arrivals', fontsize=15, fontname='serif', color='darkred')
plt.grid(True)
plt.xticks(range(len(arrival_counts)), arrival_counts.index, rotation=45)
plt.tight_layout()
plt.show()

