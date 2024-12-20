import streamlit as st
import pandas as pd
import base64
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import yfinance as yf

st.set_page_config(page_title="S&P 500 Explorer", page_icon=":chart_with_upwards_trend:", layout="wide")
sns.set_style("whitegrid")
sns.set_palette("deep")

# Custom CSS for the sidebar
st.markdown(
    """
    <style>
    /* Style the sidebar */
    section[data-testid="stSidebar"] {
        background-color: #000000;
    }
    section[data-testid="stSidebar"] .stMarkdown h2, 
    section[data-testid="stSidebar"] .stMarkdown h3, 
    section[data-testid="stSidebar"] label {
        color: #FFFFFF;
        font-family: "Helvetica", "Arial", sans-serif;
    }
    section[data-testid="stSidebar"] .stMarkdown p {
        color: #FFFFFF;
    }
    /* Style the download link */
    a, a:hover {
        color: #0B3B8C;
        text-decoration: none;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True
)

st.title("S&P 500 Explorer")

st.markdown("""
This application retrieves the list of **S&P 500** companies and displays their year-to-date 
closing prices. Use the sidebar to filter sectors and select specific companies to visualize.
""")

st.sidebar.header("User Input Features")

@st.cache_data
def load_data():
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    html = pd.read_html(url, header=0)
    df = html[0]
    return df

with st.spinner("Loading S&P 500 data..."):
    df = load_data()

# Sector selection
sorted_sector_unique = sorted(df['GICS Sector'].unique())
selected_sector = st.sidebar.multiselect(
    'Select sectors:',
    sorted_sector_unique,
    default=sorted_sector_unique
)

df_selected_sector = df[df['GICS Sector'].isin(selected_sector)]

st.header("Companies in Selected Sectors")
st.write(f"**Data Dimension:** {df_selected_sector.shape[0]} rows, {df_selected_sector.shape[1]} columns.")
st.dataframe(df_selected_sector)

# User can now select specific companies
selected_companies = st.sidebar.multiselect(
    "Select Companies to Plot",
    df_selected_sector['Symbol'].tolist(),
    default=df_selected_sector['Symbol'].tolist()[:5]
)

def filedownload(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="SP500_filtered.csv">Download Filtered CSV</a>'
    return href

st.markdown(filedownload(df_selected_sector), unsafe_allow_html=True)

with st.spinner("Fetching stock price data..."):
    data = yf.download(
        tickers=selected_companies,
        period="ytd",
        interval="1d",
        group_by='ticker',
        auto_adjust=True,
        prepost=True,
        threads=True,
        proxy=None
    )


def price_plot(symbol):
    # Extract symbol data
    df_symbol = pd.DataFrame(data[symbol])
    df_symbol = df_symbol[['Close', 'Volume']].dropna()
    df_symbol['Date'] = df_symbol.index

    # Calculate a 50-day moving average on Close
    df_symbol['MA50'] = df_symbol['Close'].rolling(window=50).mean()

    # Identify min and max
    min_close = df_symbol['Close'].min()
    max_close = df_symbol['Close'].max()
    min_date = df_symbol['Date'][df_symbol['Close'].idxmin()]
    max_date = df_symbol['Date'][df_symbol['Close'].idxmax()]

    # Create subplots: top for price, bottom for volume
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True,
                                   gridspec_kw={'height_ratios': [3, 1]})

    # Price subplot
    ax1.fill_between(df_symbol['Date'], df_symbol['Close'], color='#66c2a5', alpha=0.1)
    ax1.plot(df_symbol['Date'], df_symbol['Close'], color='#3288bd', linewidth=2, label='Closing Price')
    if df_symbol['MA50'].notna().any():
        ax1.plot(df_symbol['Date'], df_symbol['MA50'], color='#d53e4f', linestyle='--', linewidth=2, label='50-day MA')

    # Annotate min and max
    ax1.scatter(min_date, min_close, color='red', zorder=5)
    ax1.annotate(f"Min: {min_close:.2f}", xy=(min_date, min_close), xytext=(15, -15),
                 textcoords="offset points", arrowprops=dict(arrowstyle='->', color='red'),
                 ha='left', va='top', fontsize=10, color='red')

    ax1.scatter(max_date, max_close, color='green', zorder=5)
    ax1.annotate(f"Max: {max_close:.2f}", xy=(max_date, max_close), xytext=(-15, 15),
                 textcoords="offset points", arrowprops=dict(arrowstyle='->', color='green'),
                 ha='right', va='bottom', fontsize=10, color='green')

    ax1.set_title(f"{symbol} Closing Price (YTD)", fontweight='bold', fontsize=14)
    ax1.set_ylabel('Price (USD)', fontweight='bold')
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend(loc='upper left')

    # Volume subplot
    ax2.bar(df_symbol['Date'], df_symbol['Volume'], color='#001D3D', width=0.9)
    ax2.set_ylabel('Volume', fontweight='bold')
    ax2.grid(True, linestyle='--', alpha=0.5)

    # Improve x-axis formatting
    plt.xticks(rotation=45)
    plt.tight_layout()

    st.pyplot(fig)

if st.sidebar.button('Show Stock Price Plots'):
    st.header('Selected Companies Stock Closing Price Trends')
    for symbol in selected_companies:
        price_plot(symbol)