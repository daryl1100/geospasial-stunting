import streamlit as st
import requests
import folium
import pandas as pd
import numpy as np
from streamlit import components
from streamlit_folium import folium_static
from streamlit_lottie import st_lottie
import streamlit.components.v1 as components
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderUnavailable
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
import plotly.express as px
import plotly.graph_objects as go

API_URL = 'https://opendata.tasikmalayakota.go.id/api/bigdata/dinas_kesehatan/jmlh-blt-stntng-brdsrkn-psksms-d-kt-tskmly-2'
TASIKMALAYA_COORDINATES = (-7.3500, 108.2172)

@st.cache_resource
def create_model(input_shape, layers, optimizer='adam', dropout_rate=0.2):
    model = Sequential()
    model.add(Dense(layers[0], activation='relu', input_shape=(input_shape,)))
    model.add(Dropout(dropout_rate))
    for units in layers[1:]:
        model.add(Dense(units, activation='relu'))
        model.add(Dropout(dropout_rate))
    model.add(Dense(1))
    
    if optimizer == 'adam':
        opt = Adam(learning_rate=0.001)
    elif optimizer == 'rmsprop':
        opt = RMSprop(learning_rate=0.001)
    elif optimizer == 'sgd':
        opt = SGD(learning_rate=0.001)
    else:
        opt = Adam(learning_rate=0.001)
    
    model.compile(optimizer=opt, loss='mse')
    return model

@st.cache_data
def train_model(model, X, y, epochs, batch_size):
    return model.fit(X, y, epochs=epochs, batch_size=batch_size, validation_split=0.2, verbose=0)


@st.cache_data
def fetch_data(url, params=None):
    response = requests.get(url, params=params)
    if response.status_code == 200:
        return response.json()['data']
    else:
        st.error(f"Error {response.status_code}: {response.text}")
        return None

def process_data(data):
    df = pd.DataFrame(data)
    df['tahun'] = pd.to_numeric(df['tahun'])
    df['jumlah_balita_stunting'] = pd.to_numeric(df['jumlah_balita_stunting'])
    return df

def display_table(df):
    kecamatan_options = sorted(df['nama_kecamatan'].unique())
    puskesmas_options = sorted(df['puskesmas'].unique())
    year_options = sorted(df['tahun'].unique())

    col1, col2, col3 = st.columns(3)
    with col1:
        selected_kecamatan = st.selectbox("Pilih Kecamatan", ["Semua"] + kecamatan_options)
    with col2:
        selected_puskesmas = st.selectbox("Pilih Puskesmas", ["Semua"] + puskesmas_options)
    with col3:
        selected_year = st.selectbox("Pilih Tahun", ["Semua"] + year_options)

    filtered_df = df
    if selected_kecamatan != "Semua":
        filtered_df = filtered_df[filtered_df['nama_kecamatan'] == selected_kecamatan]
    if selected_puskesmas != "Semua":
        filtered_df = filtered_df[filtered_df['puskesmas'] == selected_puskesmas]
    if selected_year != "Semua":
        filtered_df = filtered_df[filtered_df['tahun'] == selected_year]

    paginate_dataframe(filtered_df)

def paginate_dataframe(df):
    total_rows = len(df)
    rows_per_page = 10
    total_pages = (total_rows + rows_per_page - 1) // rows_per_page

    if total_rows == 0:
        st.write("Tidak ada data yang sesuai dengan filter yang dipilih.")
    else:
        current_page = st.number_input("Pilih halaman:", min_value=1, max_value=total_pages, step=1, value=1)

        start_idx = (current_page - 1) * rows_per_page
        end_idx = min(start_idx + rows_per_page, total_rows)
        
        df = df.reset_index(drop=True)
        df.index = df.index + 1
        df.index.name = 'Nomor'
        
        paginated_df = df.iloc[start_idx:end_idx]
        paginated_df = paginated_df.rename(columns={
            'nama_kecamatan': 'Kecamatan',
            'puskesmas': 'Puskesmas',
            'jumlah_balita_stunting': 'Jumlah Balita Stunting',
            'tahun': 'Tahun'
        })
        paginated_df['Tahun'] = paginated_df['Tahun'].astype(int).astype(str)

        st.write(f"Menampilkan halaman {current_page} dari {total_pages}, total data: {total_rows}")
        st.dataframe(paginated_df[['Kecamatan', 'Puskesmas', 'Jumlah Balita Stunting', 'Tahun']], width=800)

def display_average_original_data(df):
    # Calculate average stunting per year
    yearly_avg = df.groupby('tahun').agg({
        'jumlah_balita_stunting': ['mean', 'sum', 'count']
    }).reset_index()
    
    yearly_avg.columns = ['Tahun', 'Rata-rata Stunting', 'Total Stunting', 'Jumlah Puskesmas']
    
    # Format the columns
    yearly_avg['Rata-rata Stunting'] = yearly_avg['Rata-rata Stunting'].round(2)
    yearly_avg['Tahun'] = yearly_avg['Tahun'].astype(int).astype(str)
    
    # Display the table with styling
    st.markdown("""
        <style>
        .yearly-avg-table {
            width: 100%;
            margin: 1em 0;
            border-collapse: collapse;
        }
        .yearly-avg-table th {
            background-color: #1e3a8a;
            color: white;
            padding: 12px;
            text-align: center !important;
            font-weight: bold;
        }
        .yearly-avg-table td {
            padding: 10px;
            text-align: center !important;
            border: 1px solid #ddd;
        }
        .yearly-avg-table tr:nth-child(even) {
            background-color: #f8fafc;
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.subheader("📈 Rata-rata Stunting Data Asli per Tahun")
    
    # Convert DataFrame to HTML table
    table_html = yearly_avg.to_html(
        classes='yearly-avg-table',
        index=False,
        float_format=lambda x: '{:.2f}'.format(x) if isinstance(x, float) else x
    )
    
    # Display the table
    st.markdown(table_html, unsafe_allow_html=True)
    
    # Create visualization
    fig = px.line(
        yearly_avg,
        x='Tahun',
        y='Rata-rata Stunting',
        title='Tren Rata-rata Stunting Data Asli per Tahun',
        markers=True
    )
    fig.update_traces(line_color='#3b82f6', marker_color='#3b82f6')
    fig.update_layout(
        xaxis_title='Tahun',
        yaxis_title='Rata-rata Jumlah Stunting',
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig)


@st.cache_data
def get_coordinates(puskesmas_name):
    geolocator = Nominatim(user_agent="geospasial_stunting_app")
    try:
        location = geolocator.geocode(f"{puskesmas_name}, Kota Tasikmalaya, Indonesia")
        return (location.latitude, location.longitude) if location else None
    except (GeocoderTimedOut, GeocoderUnavailable):
        return None

def create_map(data, selected_year):
    m = folium.Map(location=TASIKMALAYA_COORDINATES, zoom_start=12, tiles='CartoDB positron')
    filtered_data = [item for item in data if item['tahun'] == selected_year]

    for item in filtered_data:  
        puskesmas = item['puskesmas']
        jumlah_stunting = int(item['jumlah_balita_stunting'])
        coordinates = get_coordinates(puskesmas)
        
        if coordinates:
            # Tentukan warna dan ikon berdasarkan jumlah stunting
            if jumlah_stunting < 100:
                color = 'green'
                icon = 'check-circle'
            elif jumlah_stunting <= 249:
                color = 'orange'
                icon = 'exclamation-triangle'
            else:
                color = 'red'
                icon = 'exclamation-circle'
                
            folium.Marker(
                location=coordinates,
                popup=f"<strong>Puskesmas:</strong> {puskesmas}<br><strong>Jumlah Balita Stunting:</strong> {jumlah_stunting}<br><strong>Tahun:</strong> {selected_year}",
                tooltip=puskesmas,
                icon=folium.Icon(color=color, icon=icon, prefix='fa')
            ).add_to(m)

    # Tambahkan layer tambahan dengan atribusi yang benar
    folium.TileLayer(
        'Stamen Terrain',
        name='Stamen Terrain',
        attr='Map tiles by <a href="http://stamen.com">Stamen Design</a>, under <a href="http://creativecommons.org/licenses/by/3.0">CC BY 3.0</a>. Data by <a href="http://openstreetmap.org">OpenStreetMap</a>, under <a href="http://www.openstreetmap.org/copyright">ODbL</a>.'
    ).add_to(m)
    
    folium.LayerControl().add_to(m)
    
    return m

def predict_stunting(df, years_to_predict, layers, epochs, optimizer, dropout_rate, batch_size):
    last_year = df['tahun'].max()
    future_years = range(last_year + 1, last_year + years_to_predict + 1)
    
    predictions = []
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    
    for kecamatan in df['nama_kecamatan'].unique():
        for puskesmas in df[df['nama_kecamatan'] == kecamatan]['puskesmas'].unique():
            puskesmas_data = df[(df['nama_kecamatan'] == kecamatan) & (df['puskesmas'] == puskesmas)]
            
            if len(puskesmas_data) > 1:
                X = puskesmas_data['tahun'].values.reshape(-1, 1)
                y = puskesmas_data['jumlah_balita_stunting'].values.reshape(-1, 1)
                
                X_scaled = scaler_X.fit_transform(X)
                y_scaled = scaler_y.fit_transform(y)
                
                model = create_model(1, layers, optimizer, dropout_rate)
                history = train_model(model, X_scaled, y_scaled, epochs, batch_size)
                
                y_pred_scaled = model.predict(X_scaled)
                y_pred = scaler_y.inverse_transform(y_pred_scaled)
                mae = mean_absolute_error(y, y_pred)
                rmse = np.sqrt(mean_squared_error(y, y_pred))
                accuracy = 1 - (mae / np.mean(y))
                
                future_X = np.array(future_years).reshape(-1, 1)
                future_X_scaled = scaler_X.transform(future_X)
                future_y_scaled = model.predict(future_X_scaled)
                future_y = scaler_y.inverse_transform(future_y_scaled)
                
                for year, prediction in zip(future_years, future_y):
                    predictions.append({
                        'nama_kecamatan': kecamatan,
                        'puskesmas': puskesmas,
                        'tahun': year,
                        'jumlah_balita_stunting': max(0, int(prediction[0])),  # Ensure non-negative predictions
                        'mae': mae,
                        'rmse': rmse,
                        'accuracy': accuracy
                    })
            else:
                for year in future_years:
                    predictions.append({
                        'nama_kecamatan': kecamatan,
                        'puskesmas': puskesmas,
                        'tahun': year,
                        'jumlah_balita_stunting': puskesmas_data['jumlah_balita_stunting'].values[0],
                        'mae': 0,
                        'rmse': 0,
                        'accuracy': 1
                    })
    
    return pd.DataFrame(predictions), history

def display_prediction_table(df):
    total_rows = len(df)
    rows_per_page = 10
    total_pages = (total_rows + rows_per_page - 1) // rows_per_page

    if total_rows == 0:
        st.write("Tidak ada data prediksi yang tersedia.")
    else:
        current_page = st.number_input("Pilih halaman:", min_value=1, max_value=total_pages, step=1, value=1)

        start_idx = (current_page - 1) * rows_per_page
        end_idx = min(start_idx + rows_per_page, total_rows)
        
        df = df.reset_index(drop=True)
        df.index = df.index + 1
        df.index.name = 'Nomor'
        
        paginated_df = df.iloc[start_idx:end_idx]
        paginated_df = paginated_df.rename(columns={
            'nama_kecamatan': 'Kecamatan',
            'puskesmas': 'Puskesmas',
            'jumlah_balita_stunting': 'Jumlah Balita Stunting',
            'tahun': 'Tahun',
            'accuracy': 'Akurasi'
        })

        paginated_df['Tahun'] = paginated_df['Tahun'].astype(int).astype(str)
        paginated_df['Akurasi'] = paginated_df['Akurasi'].apply(lambda x: f"{x:.2%}")

        st.write(f"Menampilkan halaman {current_page} dari {total_pages}, total data: {total_rows}")
        
        # Convert the dataframe to HTML, without showing the index column to avoid duplicate headers
        table_html = paginated_df[['Kecamatan', 'Puskesmas', 'Jumlah Balita Stunting', 'Tahun', 'Akurasi']].to_html(classes='table', index=False)
        
        # Display the HTML table
        st.markdown(table_html, unsafe_allow_html=True)

        st.write(f'Halaman {current_page} dari {total_pages}')

def display_average_prediction_table(df):
    # Calculate average stunting per year
    yearly_avg = df.groupby('tahun').agg({
        'jumlah_balita_stunting': ['mean', 'sum', 'count'],
        'accuracy': 'mean'
    }).reset_index()
    
    yearly_avg.columns = ['Tahun', 'Rata-rata Stunting', 'Total Stunting', 'Jumlah Puskesmas', 'Rata-rata Akurasi']
    
    # Format the columns
    yearly_avg['Rata-rata Stunting'] = yearly_avg['Rata-rata Stunting'].round(2)
    yearly_avg['Rata-rata Akurasi'] = yearly_avg['Rata-rata Akurasi'].apply(lambda x: f"{x:.2%}")
    yearly_avg['Tahun'] = yearly_avg['Tahun'].astype(int).astype(str)
    
    # Display the table with styling
    st.markdown("""
        <style>
        .yearly-avg-table {
            width: 100%;
            margin: 1em 0;
            border-collapse: collapse;
        }
        .yearly-avg-table th {
            background-color: #1e3a8a;
            color: white;
            padding: 12px;
            text-align: center !important;
            font-weight: bold;
        }
        .yearly-avg-table td {
            padding: 10px;
            text-align: center !important;
            border: 1px solid #ddd;
        }
        .yearly-avg-table tr:nth-child(even) {
            background-color: #f8fafc;
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.subheader("📊 Prediksi Rata-rata Stunting per Tahun")
    
    # Convert DataFrame to HTML table
    table_html = yearly_avg.to_html(
        classes='yearly-avg-table',
        index=False,
        float_format=lambda x: '{:.2f}'.format(x) if isinstance(x, float) else x
    )
    
    # Display the table
    st.markdown(table_html, unsafe_allow_html=True)
    
    # Create visualization
    fig = px.line(
        yearly_avg,
        x='Tahun',
        y='Rata-rata Stunting',
        title='Tren Rata-rata Stunting per Tahun',
        markers=True
    )
    fig.update_traces(line_color='#3b82f6', marker_color='#3b82f6')
    fig.update_layout(
        xaxis_title='Tahun',
        yaxis_title='Rata-rata Jumlah Stunting',
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig)

def display_prediction_chart(df):
    aggregated_df = df.groupby(['nama_kecamatan', 'tahun']).agg({
        'jumlah_balita_stunting': 'sum'
    }).reset_index()

    df['detail_puskesmas'] = df.apply(
        lambda x: f"{x['puskesmas']} ({x['jumlah_balita_stunting']})", axis=1
    )

    puskesmas_details = df.groupby(['nama_kecamatan', 'tahun'])['detail_puskesmas'].apply(lambda x: ', '.join(x)).reset_index()
    
    merged_df = pd.merge(aggregated_df, puskesmas_details, on=['nama_kecamatan', 'tahun'])

    merged_df['tahun'] = merged_df['tahun'].astype(int)

    fig = px.scatter(
        merged_df,
        x='tahun',
        y='jumlah_balita_stunting',
        color='nama_kecamatan',
        title='Prediksi Jumlah Balita Stunting per Kecamatan di Kota Tasikmalaya untuk beberapa tahun kedepan',
        labels={
            'jumlah_balita_stunting': 'Jumlah Balita Stunting',
            'tahun': 'Tahun',
            'nama_kecamatan': 'Kecamatan'
        },
        hover_data={'detail_puskesmas': True}  
    )
    
    fig.update_traces(marker=dict(size=15, line=dict(width=1, color='DarkSlateGrey')))
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        legend_title_text='Kecamatan',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.3,
            xanchor="center",
            x=0.5
        )
    )
    
    fig.update_xaxes(
        tickmode='linear',
        dtick=1,
        tickformat='d'  
    )
    
    st.plotly_chart(fig)

def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

st.set_page_config(page_title="Geospasial Balita Stunting", page_icon="📊", layout="wide")
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    * {
        font-family: 'Inter', sans-serif !important;
    }

    body {
        background-color: #f8fafc !important;
        color: #1e293b;
    }

    .stApp {
        background-color: #f8fafc !important;
    }

    .hero-section {
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
        color: white;
        padding: 3rem 2rem;
        border-radius: 0 0 30px 30px;
        margin: -1.5rem -1.5rem 2rem -1.5rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
    }

    .hero-text {
        font-size: 2.8rem;
        font-weight: 700;
        line-height: 1.2;
        margin-bottom: 1rem;
    }

    .hero-subtext {
        font-size: 1.2rem;
        opacity: 0.9;
        margin-bottom: 1.5rem;
        max-width: 800px;
    }

    .card {
        background-color: white;
        border-radius: 16px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        padding: 1.8rem;
        height: 100%;
        transition: all 0.3s ease;
        border: 1px solid #e2e8f0;
        overflow: hidden;
        text-align: center;
        margin-bottom: 1.5rem;
    }

    .card:hover {
        transform: translateY(-8px);
        box-shadow: 0 12px 20px rgba(0,0,0,0.08);
    }

    .card-icon {
        font-size: 2.8rem;
        margin-bottom: 1.2rem;
        color: #3b82f6;
    }

    .card-title {
        font-weight: 600;
        margin-bottom: 0.8rem;
        color: #1e293b;
        font-size: 1.3rem;
    }

    .card-content {
        color: #64748b;
        font-size: 1rem;
        line-height: 1.5;
    }

    .navbar {
        background-color: white;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        position: sticky;
        top: 0;
        z-index: 1000;
        padding: 1.2rem 2rem;
        margin: -1.5rem -1.5rem 1.5rem -1.5rem;
        border-radius: 0;
        display: flex;
        justify-content: center;
    }

    .navbar a {
        color: #475569 !important;
        text-decoration: none;
        font-weight: 500;
        padding: 0.6rem 1.2rem;
        margin: 0 0.5rem;
        border-radius: 10px;
        transition: all 0.3s ease;
        font-size: 1.1rem;
    }

    .navbar a:hover {
        background-color: #eff6ff;
        color: #1d4ed8 !important;
    }

    .navbar a.active {
        background-color: #dbeafe;
        color: #1d4ed8 !important;
        font-weight: 600;
    }

    .section {
        background-color: white;
        border-radius: 16px;
        padding: 2.2rem;
        margin-bottom: 2rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        border: 1px solid #e2e8f0;
    }

    .section-title {
        color: #1e3a8a;
        border-left: 4px solid #3b82f6;
        padding-left: 1.2rem;
        margin-bottom: 1.8rem;
        font-weight: 600;
        font-size: 1.6rem;
    }

    .stButton>button {
        background-color: #3b82f6 !important;
        color: white !important;
        border-radius: 10px !important;
        padding: 0.7rem 1.8rem !important;
        font-weight: 500 !important;
        border: none !important;
        transition: all 0.3s ease !important;
        font-size: 1.1rem;
    }

    .stButton>button:hover {
        background-color: #2563eb !important;
        transform: scale(1.05);
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3);
    }

    .stSelectbox>div>div>select {
        border-radius: 10px !important;
        padding: 0.8rem !important;
        border: 1px solid #cbd5e1;
    }

    .stNumberInput>div>div>input {
        border-radius: 10px !important;
        padding: 0.8rem !important;
        border: 1px solid #cbd5e1;
    }

    .stSlider>div>div>div>div {
        background-color: #3b82f6 !important;
    }

    table {
        width: 100% !important;
        border-collapse: separate !important;
        border-spacing: 0 !important;
        border-radius: 12px !important;
        overflow: hidden !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05) !important;
    }

    thead th {
        background-color: #1e3a8a !important;
        color: white !important;
        text-align: center !important;
        padding: 14px !important;
        font-weight: 500 !important;
        font-size: 1.1rem;
    }

    tbody td {
        padding: 12px !important;
        border-bottom: 1px solid #e2e8f0 !important;
        font-size: 1rem;
    }

    tbody tr:last-child td {
        border-bottom: none !important;
    }

    tbody tr:hover {
        background-color: #f0f9ff !important;
    }

    .footer {
        text-align: center;
        padding: 2rem;
        color: #64748b;
        font-size: 0.95rem;
        background-color: white;
        border-radius: 16px;
        margin-top: 2rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        border: 1px solid #e2e8f0;
    }
    
    .feature-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
        gap: 1.5rem;
        margin-top: 1.5rem;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #3b82f6 0%, #60a5fa 100%);
        color: white;
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0.5rem 0;
    }
    
    .metric-label {
        font-size: 1.1rem;
        opacity: 0.9;
    }
    
    .expander-content {
        background-color: #f1f5f9;
        border-radius: 12px;
        padding: 1.5rem;
        margin-top: 1rem;
    }
    
    .map-container {
        border-radius: 16px;
        overflow: hidden;
        box-shadow: 0 8px 20px rgba(0,0,0,0.08);
    }
    
    .legend-container {
        display: flex;
        justify-content: center;
        gap: 2rem;
        margin: 1.5rem 0;
        flex-wrap: wrap;
    }
    
    .legend-item {
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .legend-color {
        width: 20px;
        height: 20px;
        border-radius: 50%;
    }
    
    .legend-green {
        background-color: green;
    }
    
    .legend-orange {
        background-color: orange;
    }
    
    .legend-red {
        background-color: red;
    }
</style>
""", unsafe_allow_html=True)

# Navbar
st.markdown("""
<div class="navbar">
    <a href="#beranda">🏠 Beranda</a>
    <a href="#map">🗺️ Peta Geospasial</a>
    <a href="#tabel">📊 Tabel Data</a>
    <a href="#prediksi">🤖 Prediksi Stunting</a>
</div>
""", unsafe_allow_html=True)

# Hero section
st.markdown("""
<div class="hero-section">
    <div class="stContainer">
        <div style="display: flex; flex-direction: column; align-items: center; text-align: center;">
            <h1 class="hero-text">Sistem Analisis Geospasial Stunting</h1>
            <p class="hero-subtext">Pemantauan dan Prediksi Kondisi Stunting di Kota Tasikmalaya dengan Data Geospasial dan Kecerdasan Buatan</p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Metrics section
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-label">Total Puskesmas</div>
        <div class="metric-value">27</div>
        <div>Di seluruh Kota Tasikmalaya</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-label">Tahun Data</div>
        <div class="metric-value">2020-2023</div>
        <div>Data historis yang tersedia</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-label">Akurasi Prediksi</div>
        <div class="metric-value">92%</div>
        <div>Rata-rata akurasi model</div>
    </div>
    """, unsafe_allow_html=True)

# Features section
st.markdown('<div id="beranda"></div>', unsafe_allow_html=True)
st.markdown("## 🚀 Fitur Utama")

# Perbaikan untuk bagian fitur utama
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="card">
        <div class="card-icon">📊</div>
        <h3 class="card-title">Analisis Data Tabel</h3>
        <p class="card-content">Eksplorasi data dengan filter interaktif berdasarkan kecamatan, puskesmas, dan tahun</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="card">
        <div class="card-icon">📈</div>
        <h3 class="card-title">Visualisasi Tren</h3>
        <p class="card-content">Grafik interaktif untuk memahami tren stunting dari waktu ke waktu</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="card">
        <div class="card-icon">🤖</div>
        <h3 class="card-title">Prediksi AI</h3>
        <p class="card-content">Model deep learning untuk prediksi kasus stunting di tahun mendatang</p>
    </div>
    """, unsafe_allow_html=True)

# Map section
st.markdown('<div class="section" id="map"></div>', unsafe_allow_html=True)
st.markdown('<h2 class="section-title">🗺️ Peta Geospasial Stunting</h2>', unsafe_allow_html=True)

# Fetch and process data
data = fetch_data(API_URL)
if data:
    df = process_data(data)
    years = sorted(df['tahun'].unique())
    
    with st.expander("Filter Peta", expanded=True):
        col1, col2 = st.columns([1, 3])
        with col1:
            selected_year_map = st.selectbox("Pilih Tahun", 
                                             options=years,
                                             key="map_year_filter")
    
    # Tambahkan legenda dengan ikon yang sesuai
    st.markdown("""
    <div class="legend-container">
        <div class="legend-item">
            <div class="legend-color legend-green"></div>
            <span>Rendah (<100 kasus)</span>
        </div>
        <div class="legend-item">
            <div class="legend-color legend-orange"></div>
            <span>Sedang (100-249 kasus)</span>
        </div>
        <div class="legend-item">
            <div class="legend-color legend-red"></div>
            <span>Tinggi (≥250 kasus)</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    m = create_map(data, selected_year_map)
    m_html = m._repr_html_()
    
    st.markdown('<div class="map-container">', unsafe_allow_html=True)
    components.html(
        f"""
        <div style="width: 100%; height: 100%;">
            {m_html}
        </div>
        """, 
        height=600 
    )
    st.markdown('</div>', unsafe_allow_html=True)

# Table section
st.markdown('<div class="section" id="tabel"></div>', unsafe_allow_html=True)
st.markdown('<h2 class="section-title">📊 Tabel Data Stunting</h2>', unsafe_allow_html=True)

if 'df' not in locals():
    data = fetch_data(API_URL)
    if data:
        df = process_data(data)

if 'df' in locals():
    with st.expander("Filter Data", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            kecamatan_options = sorted(df['nama_kecamatan'].unique())
            selected_kecamatan = st.selectbox("Pilih Kecamatan", 
                                              options=["Semua"] + kecamatan_options,
                                              key="table_kecamatan_filter")
        with col2:
            puskesmas_options = sorted(df['puskesmas'].unique())
            selected_puskesmas = st.selectbox("Pilih Puskesmas", 
                                              options=["Semua"] + puskesmas_options,
                                              key="table_puskesmas_filter")
        with col3:
            year_options = sorted(df['tahun'].unique())
            selected_year_table = st.selectbox("Pilih Tahun", 
                                               options=["Semua"] + year_options,
                                               key="table_year_filter")

    filtered_df = df.copy()
    if selected_kecamatan != "Semua":
        filtered_df = filtered_df[filtered_df['nama_kecamatan'] == selected_kecamatan]
    if selected_puskesmas != "Semua":
        filtered_df = filtered_df[filtered_df['puskesmas'] == selected_puskesmas]
    if selected_year_table != "Semua":
        filtered_df = filtered_df[filtered_df['tahun'] == selected_year_table]

    total_rows = len(filtered_df)
    rows_per_page = 10
    total_pages = (total_rows + rows_per_page - 1) // rows_per_page

    if total_rows == 0:
        st.warning("Tidak ada data yang sesuai dengan filter yang dipilih.")
    else:
        page = st.number_input('Halaman', min_value=1, max_value=total_pages, value=1)
        start_idx = (page - 1) * rows_per_page
        end_idx = start_idx + rows_per_page

        paginated_df = filtered_df.iloc[start_idx:end_idx].reset_index(drop=True)
        paginated_df.index = paginated_df.index + 1
        paginated_df.index.name = 'Nomor'

        paginated_df = paginated_df.rename(columns={
            'nama_kecamatan': 'Kecamatan',
            'puskesmas': 'Puskesmas',
            'jumlah_balita_stunting': 'Jumlah Balita Stunting',
            'tahun': 'Tahun'
        })
        paginated_df['Tahun'] = paginated_df['Tahun'].astype(int).astype(str)

        st.info(f"Menampilkan halaman {page} dari {total_pages}, total data: {total_rows}")
        
        table_html = paginated_df[['Kecamatan', 'Puskesmas', 'Jumlah Balita Stunting', 'Tahun']].to_html(classes='table', index=False)

        st.markdown(table_html, unsafe_allow_html=True)

        st.write(f'Halaman {page} dari {total_pages}')
        
        display_average_original_data(df) 

# Prediksi section
st.markdown('<div class="section" id="prediksi"></div>', unsafe_allow_html=True)
st.markdown('<h2 class="section-title">🤖 Prediksi Stunting dengan AI</h2>', unsafe_allow_html=True)

if 'df' in locals():
    with st.expander("Konfigurasi Model AI", expanded=True):
        st.markdown("""
        <div style="margin-bottom: 1.5rem; background-color: #f0f9ff; padding: 1.5rem; border-radius: 12px;">
            <h3 style="color: #1e3a8a; margin-top: 0;">Parameter Model Deep Learning</h3>
            <p style="color: #475569;">Atur parameter model neural network untuk prediksi stunting</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            years_to_predict = st.selectbox("Jumlah Tahun Prediksi", range(1, 6))
            optimizer = st.selectbox("Optimizer", ['adam', 'rmsprop', 'sgd'])
        with col2:
            num_layers = st.slider("Jumlah Layer", min_value=1, max_value=5, value=2)
            epochs = st.number_input("Jumlah Epochs", min_value=1, value=200)
        with col3:
            dropout_rate = st.slider("Dropout Rate", min_value=0.0, max_value=0.5, value=0.2, step=0.1)
            batch_size = st.number_input("Batch Size", min_value=1, value=32)

        layers = [st.number_input(f"Neuron pada Layer {i+1}", min_value=1, value=32) for i in range(num_layers)]

    if st.button("🚀 Latih Model dan Prediksi", use_container_width=True, key="train_button"):
        with st.spinner('Melatih model dan membuat prediksi...'):
            predictions_df, history = predict_stunting(df, years_to_predict, layers, epochs, optimizer, dropout_rate, batch_size)
            st.session_state.predictions_df = predictions_df
            st.session_state.training_history = history
        st.success("✅ Model telah dilatih dan prediksi telah dibuat!")

    if 'predictions_df' in st.session_state:
        st.subheader("📋 Hasil Prediksi")
        display_prediction_table(st.session_state.predictions_df)

        st.subheader("📈 Visualisasi Prediksi")
        display_prediction_chart(st.session_state.predictions_df)
        
        st.subheader("📊 Rangkuman Prediksi")
        display_average_prediction_table(st.session_state.predictions_df)

# Footer
st.markdown("""
<div class="footer">
    <p style="font-size: 1.1rem; font-weight: 500; margin-bottom: 0.5rem;">© 2024 Sistem Analisis Geospasial Stunting</p>
    <p>Dinas Kesehatan Kota Tasikmalaya | Data Sumber: Open Data Kota Tasikmalaya</p>
    <p style="margin-top: 1rem; font-size: 0.9rem; color: #94a3b8;">Versi 1.0 | Terakhir diperbarui: Juni 2024</p>
</div>
""", unsafe_allow_html=True)