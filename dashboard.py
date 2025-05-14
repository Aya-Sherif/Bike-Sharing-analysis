import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

# Load and preprocess data
@st.cache_data
def load_data():
    data = pd.read_csv('hour.csv')
    # Transform seasons
    seasons_mapping = {1: 'winter', 2: 'spring', 3: 'summer', 4: 'fall'}
    data['season'] = data['season'].apply(lambda x: seasons_mapping[x])
    # Transform yr
    yr_mapping = {0: 2011, 1: 2012}
    data['yr'] = data['yr'].apply(lambda x: yr_mapping[x])
    # Transform weekday
    weekday_mapping = {0: 'Sunday', 1: 'Monday', 2: 'Tuesday', 3: 'Wednesday', 4: 'Thursday', 5: 'Friday', 6: 'Saturday'}
    data['weekday'] = data['weekday'].apply(lambda x: weekday_mapping[x])
    # Transform weathersit
    weather_mapping = {1: 'clear', 2: 'cloudy', 3: 'light_rain_snow', 4: 'heavy_rain_snow'}
    data['weathersit'] = data['weathersit'].apply(lambda x: weather_mapping[x])
    # Transform hum and windspeed
    data['hum'] = data['hum'] * 100
    data['windspeed'] = data['windspeed'] * 67
    # Add day_type
    data['day_type'] = data['workingday'].map({0: 'Weekend', 1: 'Weekday'})
    return data

# Load the data
data = load_data()

# Dashboard Header
st.title("Bike Sharing Data Analysis")
st.markdown("""
    This dashboard provides insights on bike sharing trends based on user types, time, environmental factors, and statistical analyses.
    Use the filters to explore the data dynamically.
""")

# Sidebar for Global Filters
st.sidebar.header("Global Filters")
year_filter = st.sidebar.multiselect("Select Year", options=[2011, 2012], default=[2011, 2012])
season_filter = st.sidebar.multiselect("Select Season", options=['winter', 'spring', 'summer', 'fall'], default=['winter', 'spring', 'summer', 'fall'])
weather_filter = st.sidebar.multiselect("Select Weather", options=['clear', 'cloudy', 'light_rain_snow', 'heavy_rain_snow'], default=['clear', 'cloudy', 'light_rain_snow', 'heavy_rain_snow'])
day_type_filter = st.sidebar.multiselect("Select Day Type", options=['Weekday', 'Weekend'], default=['Weekday', 'Weekend'])

# Apply global filters
filtered_data = data.copy()
if year_filter:
    filtered_data = filtered_data[filtered_data['yr'].isin(year_filter)]
if season_filter:
    filtered_data = filtered_data[filtered_data['season'].isin(season_filter)]
if weather_filter:
    filtered_data = filtered_data[filtered_data['weathersit'].isin(weather_filter)]
if day_type_filter:
    filtered_data = filtered_data[filtered_data['day_type'].isin(day_type_filter)]

# Create tabs for categories
tab1, tab2, tab3, tab4 = st.tabs(["User Type Analysis", "Time-Based Analysis", "Environmental Analysis", "Statistical Analysis"])

# Tab 1: User Type Analysis
with tab1:
    st.header("User Type Analysis")
    st.markdown("Explore differences between casual and registered users across various dimensions.")

    # Filters for User Type Analysis
    user_type = st.multiselect("Select User Type", options=['Casual', 'Registered'], default=['Casual', 'Registered'], key='user_type')
    weekday_filter = st.multiselect("Select Weekday", options=['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'], default=['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'], key='weekday')

    # Apply weekday filter
    user_filtered_data = filtered_data.copy()
    if weekday_filter:
        user_filtered_data = user_filtered_data[user_filtered_data['weekday'].isin(weekday_filter)]

    # Plot 1: Casual vs Registered Users by Day
    st.subheader("Casual vs Registered Users by Day")
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(24, 6))
    colors = ["#72BCD4", "#D3D3D3", "#D3D3D3", "#D3D3D3", "#D3D3D3", "#D3D3D3", "#D3D3D3"]
    if 'Casual' in user_type:
        sum_casual_user = user_filtered_data.groupby("weekday").casual.sum().reset_index()
        sns.barplot(x="casual", y="weekday", data=sum_casual_user, palette=colors, ax=ax[0])
        ax[0].set_title("Casual Users")
    if 'Registered' in user_type:
        sum_registered_user = user_filtered_data.groupby("weekday").registered.sum().reset_index()
        sns.barplot(x="registered", y="weekday", data=sum_registered_user, palette=colors, ax=ax[1])
        ax[1].set_title("Registered Users")
    plt.suptitle("Casual and Registered Users by Day", fontsize=20)
    st.pyplot(fig)
    st.markdown("""
        **Insight:** 
        The chart shows the number of casual and registered users for each day of the week. 
        Casual users tend to have a relatively more spread-out usage across the week, 
        whereas registered users show a more consistent pattern, indicating a stronger engagement on weekdays.
    """)

    # Plot 4: Rentals by User Type on Weekdays vs Weekends
    st.subheader("Rentals by User Type on Weekdays vs Weekends")
    casual_weekday = user_filtered_data[user_filtered_data['day_type'] == 'Weekday']['casual'].mean()
    casual_weekend = user_filtered_data[user_filtered_data['day_type'] == 'Weekend']['casual'].mean()
    registered_weekday = user_filtered_data[user_filtered_data['day_type'] == 'Weekday']['registered'].mean()
    registered_weekend = user_filtered_data[user_filtered_data['day_type'] == 'Weekend']['registered'].mean()
    labels = ['Casual', 'Registered']
    weekday_values = [casual_weekday if 'Casual' in user_type else 0, registered_weekday if 'Registered' in user_type else 0]
    weekend_values = [casual_weekend if 'Casual' in user_type else 0, registered_weekend if 'Registered' in user_type else 0]
    x = range(len(labels))
    bar_width = 0.35
    plt.figure(figsize=(8, 5))
    plt.bar(x, weekday_values, width=bar_width, label='Weekday', color='skyblue')
    plt.bar([i + bar_width for i in x], weekend_values, width=bar_width, label='Weekend', color='salmon')
    plt.xticks([i + bar_width / 2 for i in x], labels)
    plt.ylabel('Average Rentals')
    plt.title('Rentals by User Type (Weekday vs Weekend)')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    st.pyplot(plt)
    st.markdown("""
        **Insight:** 
        This bar chart compares the total rentals by user type (Casual and Registered) on weekdays vs weekends. 
        Casual users tend to rent more bikes on weekends, while registered users show a higher number of rentals on weekdays, suggesting that registered users are more likely to use bikes for commuting during workdays.
    """)

    # Boxplot: Registered vs Casual Users by Weekday/Weekend
    st.subheader("Distribution of Rentals: Weekday vs Weekend")
    plt.figure(figsize=(12, 5))
    if 'Registered' in user_type:
        plt.subplot(1, 2, 1)
        sns.boxplot(x='day_type', y='registered', data=user_filtered_data, palette='pastel')
        plt.title('Registered Users: Weekday vs Weekend')
        plt.xlabel('')
        plt.ylabel('Registered Rentals')
    if 'Casual' in user_type:
        plt.subplot(1, 2, 2)
        sns.boxplot(x='day_type', y='casual', data=user_filtered_data, palette='pastel')
        plt.title('Casual Users: Weekday vs Weekend')
        plt.xlabel('')
        plt.ylabel('Casual Rentals')
    plt.tight_layout()
    st.pyplot(plt)
    st.markdown("""
        **Insight:** 
        The boxplots compare registered and casual user rentals between weekdays and weekends. 
        Registered users show relatively stable usage across both weekdays and weekends, while casual users show more fluctuation, with higher rentals on weekends.
    """)

    # Monthly Rentals
    st.subheader("Casual and Registered Riders by Month")
    plt.figure(figsize=(12, 6))
    if 'Casual' in user_type:
        sns.lineplot(data=user_filtered_data, x='mnth', y='casual', marker='o', label='Casual Users', color='blue')
    if 'Registered' in user_type:
        sns.lineplot(data=user_filtered_data, x='mnth', y='registered', marker='o', label='Registered Users', color='red')
    plt.xlabel('Month')
    plt.ylabel('Number of Riders')
    plt.title('Casual and Registered Riders by Month')
    st.pyplot(plt)
    st.markdown("""
        **Insight:** 
        This line plot shows the number of casual and registered riders for each month. 
        Casual riders show higher activity in the summer months, while registered riders have a more consistent usage pattern across the year.
    """)

    # Weekday Rentals
    st.subheader("Casual and Registered Riders by Week")
    plt.figure(figsize=(12, 6))
    if 'Casual' in user_type:
        sns.lineplot(data=user_filtered_data, x='weekday', y='casual', marker='o', label='Casual Users', color='blue')
    if 'Registered' in user_type:
        sns.lineplot(data=user_filtered_data, x='weekday', y='registered', marker='o', label='Registered Users', color='red')
    plt.xlabel('Day')
    plt.ylabel('Number of Riders')
    plt.title('Casual and Registered Riders by Week')
    st.pyplot(plt)
    st.markdown("""
        **Insight:** 
        This plot shows the number of casual and registered riders per day of the week. 
        Casual riders show higher activity on weekends, while registered riders show higher usage on weekdays, likely due to commuting.
    """)

    # Hourly Rentals
    st.subheader("Casual and Registered Riders by Hour")
    plt.figure(figsize=(12, 6))
    if 'Casual' in user_type:
        sns.lineplot(data=user_filtered_data, x='hr', y='casual', label='Casual Users', color='blue')
    if 'Registered' in user_type:
        sns.lineplot(data=user_filtered_data, x='hr', y='registered', label='Registered Users', color='red')
    plt.xlabel('Hour')
    plt.ylabel('Number of Riders')
    plt.title('Casual and Registered Riders by Hour')
    st.pyplot(plt)
    st.markdown("""
        **Insight:** 
        The graph compares casual and registered riders throughout the day. 
        Casual users have a peak in the afternoon and evening hours, while registered users have a more uniform usage pattern, peaking during typical work commute times.
    """)

# Tab 2: Time-Based Analysis
with tab2:
    st.header("Time-Based Analysis")
    st.markdown("Analyze bike sharing trends over time (hourly, daily, monthly).")

    # Filters for Time-Based Analysis
    hour_range = st.slider("Select Hour Range", min_value=0, max_value=23, value=(0, 23), key='hour')
    month_filter = st.multiselect("Select Month", options=list(range(1, 13)), default=list(range(1, 13)), key='month')

    # Apply time-based filters
    time_filtered_data = filtered_data.copy()
    time_filtered_data = time_filtered_data[(time_filtered_data['hr'] >= hour_range[0]) & (time_filtered_data['hr'] <= hour_range[1])]
    if month_filter:
        time_filtered_data = time_filtered_data[time_filtered_data['mnth'].isin(month_filter)]

    # Plot 2: Bike Sharing Productivity for Casual Users
    st.subheader("Bike Sharing Productivity for Casual Users")
    fig, ax = plt.subplots(figsize=(20, 5))
    sns.pointplot(data=time_filtered_data, x='hr', y='casual', hue='workingday', errorbar=None, ax=ax)
    ax.set(title='Casual User Trends by Hour of Day')
    ax.set_ylabel('Total Users')
    ax.set_xlabel('Hour of Day')
    st.pyplot(fig)
    st.markdown("""
        **Insight:** 
        The graph illustrates the trend of casual users throughout the day. Casual users tend to be more active during the daytime, with peaks during typical work hours. The workingday filter shows higher activity on weekdays, suggesting that casual users tend to rent bikes during their work hours.
    """)

    # Plot 3: Bike Sharing Productivity for Registered Users
    st.subheader("Bike Sharing Productivity for Registered Users")
    fig, ax = plt.subplots(figsize=(20, 5))
    sns.pointplot(data=time_filtered_data, x='hr', y='registered', hue='workingday', errorbar=None, ax=ax)
    ax.set(title='Registered User Trends by Hour of Day')
    ax.set_ylabel('Total Users')
    ax.set_xlabel('Hour of Day')
    st.pyplot(fig)
    st.markdown("""
        **Insight:** 
        Registered users show a more stable usage pattern throughout the day. The data suggests that registered users tend to have more predictable usage during the day, with an increase in activity during the morning and evening rush hours. Their peak activity appears during the workday hours.
    """)

    # Monthly Rentals
    st.subheader("Casual and Registered Riders by Month")
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=time_filtered_data, x='mnth', y='casual', marker='o', label='Casual Users', color='blue')
    sns.lineplot(data=time_filtered_data, x='mnth', y='registered', marker='o', label='Registered Users', color='red')
    plt.xlabel('Month')
    plt.ylabel('Number of Riders')
    plt.title('Casual and Registered Riders by Month')
    st.pyplot(plt)
    st.markdown("""
        **Insight:** 
        This line plot shows the number of casual and registered riders for each month. 
        Casual riders show higher activity in the summer months, while registered riders have a more consistent usage pattern across the year.
    """)

    # Weekday Rentals
    st.subheader("Casual and Registered Riders by Week")
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=time_filtered_data, x='weekday', y='casual', marker='o', label='Casual Users', color='blue')
    sns.lineplot(data=time_filtered_data, x='weekday', y='registered', marker='o', label='Registered Users', color='red')
    plt.xlabel('Day')
    plt.ylabel('Number of Riders')
    plt.title('Casual and Registered Riders by Week')
    st.pyplot(plt)
    st.markdown("""
        **Insight:** 
        This plot shows the number of casual and registered riders per day of the week. 
        Casual riders show higher activity on weekends, while registered riders show higher usage on weekdays, likely due to commuting.
    """)

    # Hourly Rentals
    st.subheader("Casual and Registered Riders by Hour")
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=time_filtered_data, x='hr', y='casual', label='Casual Users', color='blue')
    sns.lineplot(data=time_filtered_data, x='hr', y='registered', label='Registered Users', color='red')
    plt.xlabel('Hour')
    plt.ylabel('Number of Riders')
    plt.title('Casual and Registered Riders by Hour')
    st.pyplot(plt)
    st.markdown("""
        **Insight:** 
        The graph compares casual and registered riders throughout the day. 
        Casual users have a peak in the afternoon and evening hours, while registered users have a more uniform usage pattern, peaking during typical work commute times.
    """)

# Tab 3: Environmental Analysis
with tab3:
    st.header("Environmental Analysis")
    st.markdown("Examine the impact of environmental factors on bike rentals.")

    # Filters for Environmental Analysis
    temp_range = st.slider("Select Temperature Range", min_value=float(data['temp'].min()), max_value=float(data['temp'].max()), value=(float(data['temp'].min()), float(data['temp'].max())), key='temp')
    hum_range = st.slider("Select Humidity Range", min_value=float(data['hum'].min()), max_value=float(data['hum'].max()), value=(float(data['hum'].min()), float(data['hum'].max())), key='hum')
    wind_range = st.slider("Select Windspeed Range", min_value=float(data['windspeed'].min()), max_value=float(data['windspeed'].max()), value=(float(data['windspeed'].min()), float(data['windspeed'].max())), key='wind')

    # Apply environmental filters
    env_filtered_data = filtered_data.copy()
    env_filtered_data = env_filtered_data[(env_filtered_data['temp'] >= temp_range[0]) & (env_filtered_data['temp'] <= temp_range[1])]
    env_filtered_data = env_filtered_data[(env_filtered_data['hum'] >= hum_range[0]) & (env_filtered_data['hum'] <= hum_range[1])]
    env_filtered_data = env_filtered_data[(env_filtered_data['windspeed'] >= wind_range[0]) & (env_filtered_data['windspeed'] <= wind_range[1])]

    # Seasonal Rentals
    st.subheader("Total Users by Season")
    max_season = env_filtered_data.groupby('season')['cnt'].sum().idxmax()
    colors = ['red' if season == max_season else 'grey' for season in env_filtered_data['season'].unique()]
    plt.figure(figsize=(10, 6))
    sns.barplot(data=env_filtered_data, x='season', y='cnt', palette=colors)
    plt.title('Total Users by Season')
    plt.xlabel('Season')
    plt.ylabel('Total Users')
    st.pyplot(plt)
    st.markdown("""
        **Insight:** 
        The barplot highlights the season with the maximum number of rentals. 
        The summer season (`season=2`) shows the highest user engagement, likely due to better weather conditions for outdoor activities.
    """)

    # Weather Rentals
    st.subheader("Total Users by Weather Situation")
    max_weathersit = env_filtered_data.groupby('weathersit')['cnt'].sum().idxmax()
    colors = ['red' if weathersit == max_weathersit else 'grey' for weathersit in env_filtered_data['weathersit'].unique()]
    plt.figure(figsize=(10, 6))
    sns.barplot(data=env_filtered_data, x='weathersit', y='cnt', palette=colors)
    plt.title('Total Users by Weather Situation')
    plt.xlabel('Weather Situation')
    plt.ylabel('Total Users')
    st.pyplot(plt)
    st.markdown("""
        **Insight:** 
        The weather situation graph shows that clear weather (denoted by `weathersit=1`) is associated with the highest number of rentals, while rain or snow (weathersit=3) results in fewer rentals.
    """)

    # Correlation Matrix
    st.subheader("Correlation Matrix")
    numerical_features = ['temp', 'atemp', 'hum', 'windspeed', 'casual', 'registered']
    corr_matrix = env_filtered_data[numerical_features].corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Correlation Matrix')
    st.pyplot(plt)
    st.markdown("""
        **Insight:** 
        The correlation matrix reveals the relationships between numerical features. 
        For instance, temperature (`temp`) and apparent temperature (`atemp`) show a high positive correlation, suggesting that as the temperature increases, so does the apparent temperature. 
        `Casual` and `Registered` rentals also show moderate correlations with the weather variables.
    """)

# Tab 4: Statistical Analysis
with tab4:
    st.header("Statistical Analysis")
    st.markdown("View statistical tests and correlations for deeper insights.")

    # T-test Results
    st.subheader("T-test: Weekday vs Weekend Rentals")
    weekday_data = filtered_data[filtered_data['workingday'] == 1]
    weekend_data = filtered_data[filtered_data['workingday'] == 0]
    t_stat_reg, p_val_reg = ttest_ind(weekday_data['registered'], weekend_data['registered'], equal_var=False)
    t_stat_cas, p_val_cas = ttest_ind(weekday_data['casual'], weekend_data['casual'], equal_var=False)
    st.markdown(f"""
        **T-test Results:** 
        - Registered Users: T-statistic = {t_stat_reg:.2f}, P-value = {p_val_reg:.5f}
        - Casual Users: T-statistic = {t_stat_cas:.2f}, P-value = {p_val_cas:.5f}

        A lower p-value indicates that there is a significant difference between weekday and weekend rentals for both registered and casual users.
    """)

    # Correlation Matrix
    st.subheader("Correlation Matrix")
    numerical_features = ['temp', 'atemp', 'hum', 'windspeed', 'casual', 'registered']
    corr_matrix = filtered_data[numerical_features].corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Correlation Matrix')
    st.pyplot(plt)
    st.markdown("""
        **Insight:** 
        The correlation matrix reveals the relationships between numerical features. 
        For instance, temperature (`temp`) and apparent temperature (`atemp`) show a high positive correlation, suggesting that as the temperature increases, so does the apparent temperature. 
        `Casual` and `Registered` rentals also show moderate correlations with the weather variables.
    """)    