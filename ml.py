#import all libraries and modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import plotly
import plotly.express as px
import seaborn as sns
from matplotlib import cm
import plotly.graph_objects as go
import plotly.io as pio
#read csv file through pandas
df = pd.read_csv('covid_19_india.csv', parse_dates=['Date'], dayfirst=True)
df = df[['Date', 'State/UnionTerritory','Cured','Deaths','Confirmed']]
#renaming columns for simplification of dataframe
df.columns = ['date', 'state','cured','deaths','confirmed']

#creating new dataframe for extracting todays covid cases
today = df[df['date']=='2020-10-27']
confirmed = today.sort_values('confirmed', ascending=True)


#Statewise new confirmed
df['new_confirmed'] = df.groupby(['state'])['confirmed'].diff()
df['new_deaths'] = df.groupby(['state'])['deaths'].diff()
df['new_cured'] = df.groupby(['state'])['cured'].diff()

#plotting graphs throught plotly module 
# line graph for confirmed cases
fig = px.line(df, x="date", y="confirmed", color='state',template= "plotly_white")
fig.update_xaxes(tickfont=dict(family='Rockwell', color='black', size=14))
fig.update_yaxes(tickfont=dict(family='Rockwell', color='black', size=14))
fig.update_traces(mode='lines + markers')
fig.update_layout(legend_orientation="h",legend=dict(x= -.1, y=-.3),
                  autosize=False,
                  width= 750,
                  height= 850,
                  title_text='<b>Confirmed Cases of Covid-19 in India<b> ',
                  title_x=0.5,
                 paper_bgcolor='snow',
                 plot_bgcolor = "snow")
fig.show()

#bar graph for confirmed cases per state using px.bar function
fig = px.bar(confirmed, x="confirmed", y="state", orientation='h', text = 'confirmed')
fig.update_layout(
    title_text='<b>Confirmed cases of Covid-19 per State <b>',
    title_x=0.5,
    paper_bgcolor='aqua',
    plot_bgcolor = "aqua",
    autosize=False,
    width= 850,
    height=750)
fig.update_traces(marker_color='teal')
fig.update_xaxes(tickfont=dict(family='Rockwell', color='black', size=14))
fig.update_yaxes(tickfont=dict(family='Rockwell', color='black', size=14))


deaths = today.sort_values('deaths', ascending=True)
deaths = deaths[deaths.deaths > 0 ]
# px.bar function to plot deaths due to covid-19
fig = px.bar(deaths, x="deaths", y="state", orientation='h', text = 'deaths')
fig.update_layout(
    title_text='<b>Deaths due to covid 19<b>',
    title_x=0.5,
    paper_bgcolor='rgb(255,223,0)',
    plot_bgcolor = "rgb(255,223,0)",
    autosize=False,
    width=850,
    height= 850)
fig.update_traces(marker_color='red')
fig.update_xaxes(tickfont=dict(family='Rockwell', color='darkblue', size=14))
fig.update_yaxes(tickfont=dict(family='Rockwell', color='darkblue', size=14))
fig.show()

df2 = df.groupby(['date'])[['confirmed', 'deaths','cured',]].sum().reset_index()
df2['new_confirmed'] = df2.confirmed.diff()
df2['new_deaths'] = df2.deaths.diff()
df2['new_cured'] = df2.cured.diff()
#taking dates from 15th March
df2 = df2.iloc[44:]

fig = go.Figure(go.Bar(x= df2.date, y= df2.cured, name='Recovered'))
fig.add_trace(go.Bar(x=df2.date, y= df2.deaths, name='Deaths'))
fig.add_trace(go.Bar(x=df2.date, y= df2.confirmed, name='Confirmed'))

fig.update_layout(barmode='stack',legend_orientation="h",legend=dict(x= 0.3, y=1.1),
                  xaxis={'categoryorder':'total descending'},
                 title_text='<b>Covid 19 Total cases in India (since 15 March)<b>',
                  title_x=0.5,
                 paper_bgcolor='whitesmoke',
                 plot_bgcolor = "whitesmoke",)
fig.update_xaxes(tickfont=dict(family='Rockwell', color='black', size=14))
fig.update_yaxes(tickfont=dict(family='Rockwell', color='black', size=14))
fig.show()


#Forecasting
df3 = df2[['date' , 'confirmed']]
#Renaming column names according to fb prophet
df3.columns = ['ds' , 'y']

from fbprophet import Prophet

#model
m = Prophet()

#fitting the model
m.fit(df3)

#forecast
future = m.make_future_dataframe(periods= 20)
future.tail()
forecast = m.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(21)

from fbprophet.plot import plot_plotly
fig = plot_plotly(m, forecast)  # This returns a plotly Figure
fig.update_layout(
                  autosize=False,
                  width= 750,
                  height= 800,
    title_text='<b>Covid-19 Total cases Forecast<b>',
    title_x=0.5,
    paper_bgcolor='khaki',
    plot_bgcolor = "khaki",)
fig.show()


