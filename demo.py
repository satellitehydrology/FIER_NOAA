import folium
import folium.plugins as plugins
import streamlit as st
from streamlit_folium import folium_static
from PIL import Image
import xarray as xr
from syn_noaa import *
from pynwm_upd import *
import numpy.ma as ma
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import branca
import branca.colormap as cm

today_date = datetime.date.today()
today_datestr = today_date.strftime('%Y-%m-%d')
exp_mid_fct = MediumRange(station_id=7469342)
exp_mid_fct_indata = exp_mid_fct.data["mean"][0]["data"]
exp_mid_fct_data = pd.DataFrame(exp_mid_fct_indata)["forecast-time"]
exp_mid_fct_time = pd.to_datetime(exp_mid_fct_data)

last_date = exp_mid_fct_time[len(exp_mid_fct_time)-1]
last_datestr = last_date.strftime('%Y-%m-%d')

# Page Configuration
st.set_page_config(layout="wide")

# Title and Description
st.title("Forecasting Inundation Extents using REOF analysis (FIER) – VIIRS/ABI Water Fraction")


row1_col1, row1_col2 = st.columns([2, 1])
# Set up Geemap
with row1_col1:
    m = folium.Map(
        zoom_start=4,
        location =(36.52, -89.55),
        control_scale=True,
    )

    plugins.Fullscreen(position='topright').add_to(m)
    folium.TileLayer('Stamen Terrain').add_to(m)
    m.add_child(folium.LatLngPopup())
    folium.LayerControl().add_to(m)





with row1_col2:
    st.subheader('Determine Region of Interest')
    with st.form('Select Region'):

        region = st.selectbox(
     'Determine region:',
     ('Mississippi River', ),
     )

        date = st.date_input(
             "Select Date ("+today_datestr+" to "+last_datestr+"):",
             value = today_date,
             min_value = today_date,
             max_value = last_date,
             )

        submitted = st.form_submit_button("Submit")
        if submitted:
            AOI_str = region.replace(" ", "")
            st.write('Region:', region)
            st.write('Date:', date)

            bounds = run_fier(AOI_str, str(date))


            location = [36.62, -89.15] # NEED FIX!!!!!!!!!!!
            m = folium.Map(
                zoom_start = 8,
                location = location,
                control_scale=True,
            )

            folium.raster_layers.ImageOverlay(
                image= 'Output/water_fraction.png',
                # image = sar_image,
                bounds = bounds,
                opacity = 0.5,
                name = 'Water Fraction Map',
                show = True,
            ).add_to(m)

            colormap = cm.LinearColormap(colors=['blue','green','red'],
                                      vmin=0, vmax=100,
                                     caption='Water Fraction (%)')
            m.add_child(colormap)
              
            plugins.Fullscreen(position='topright').add_to(m)
            folium.TileLayer('Stamen Terrain').add_to(m)
            m.add_child(folium.LatLngPopup())
            folium.LayerControl().add_to(m)

    try:
        with open('Output/output.nc', 'rb') as f:
            st.download_button('Download Latest Run Output',
            f,
            file_name='water_fraction_%s_%s.nc'%(AOI_str, date),
            mime= "application/netcdf")
    except:
        pass



with row1_col1:
    folium_static(m, height = 600, width = 900)
    st.write('Disclaimer: This is a test version of FIER using VIIRS-derived water fraction maps over selected regions in US.')
    url = "https://www.sciencedirect.com/science/article/pii/S0034425720301024?casa_token=kOYlVMMWkBUAAAAA:fiFM4l6BUzJ8xTCksYUe7X4CcojddbO8ybzOSMe36f2cFWEXDa_aFHaGeEFlN8SuPGnDy7Ir8w"
    st.write("Reference: [Chang, C. H., Lee, H., Kim, D., Hwang, E., Hossain, F., Chishtie, F., ... & Basnayake, S. (2020). Hindcast and forecast of daily inundation extents using satellite SAR and altimetry data with rotated empirical orthogonal function analysis: Case study in Tonle Sap Lake Floodplain. Remote Sensing of Environment, 241, 111732.](%s)" % url)
    url = "https://uofh-my.sharepoint.com/:b:/g/personal/cchang37_cougarnet_uh_edu/EZ70ySxmR3RDhJL5-uAHlAEBf0xI4c-BMsXQnUKT009kFA?e=tynflq"
    st.write("See here for the [Data and Procedure](%s)" % url)
    st.write("This app has been developed by Chi-Hung Chang  & Son Do at University of Houston with supports from NOAA JPSS program.")
    st.write("Kel Markert at the Brigham Young University is also acknowledged for the development of this App.")

    uh = Image.open("logo/uh_logo_2.PNG")
    byu = Image.open("logo/BYU_Logo.png")
    jpss = Image.open("logo/JPSS_Logo.png")
    st.image([uh, byu, jpss], width=150)
