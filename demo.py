import folium
import folium.plugins as plugins
import streamlit as st
from streamlit_folium import folium_static
from PIL import Image
import xarray as xr
from syn_noaa import *
import numpy.ma as ma
import pandas as pd
import matplotlib.pyplot as plt
import datetime


def colorize(data, cmap='viridis'):
    array = ma.masked_invalid(data)
    normed_data = (array - array.min()) / (array.max() - array.min())
    cm = plt.cm.get_cmap(cmap)
    return cm(array)

# Page Configuration
st.set_page_config(layout="wide")

# Title and Description
st.title("FIER – Water Fraction")


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
             "Select Date:",
             value = datetime.date(2019, 2, 25),
             min_value = datetime.date(2019, 2, 1),
             max_value = datetime.date(2021, 6, 30),
             )

        submitted = st.form_submit_button("Submit")
        if submitted:
            AOI_str = region.replace(" ", "")
            st.write('Region:', region)
            st.write('Date:', date)

            bounds = run_fier(AOI_str, str(date))


            location = [36.62, -89.15] # NEED FIX!!!!!!!!!!!
            m = folium.Map(
                zoom_start = 7,
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

            plugins.Fullscreen(position='topright').add_to(m)
            folium.TileLayer('Stamen Terrain').add_to(m)
            m.add_child(folium.LatLngPopup())
            folium.LayerControl().add_to(m)


with row1_col1:
    folium_static(m, height = 600, width = 900)
    st.write('Disclaimer: This is a test version of FIER using VIIRS-derived water fraction maps over selected regions in US.')
    url = "https://www.sciencedirect.com/science/article/pii/S0034425720301024?casa_token=kOYlVMMWkBUAAAAA:fiFM4l6BUzJ8xTCksYUe7X4CcojddbO8ybzOSMe36f2cFWEXDa_aFHaGeEFlN8SuPGnDy7Ir8w"
    st.write("Reference: [Chang, C. H., Lee, H., Kim, D., Hwang, E., Hossain, F., Chishtie, F., ... & Basnayake, S. (2020). Hindcast and forecast of daily inundation extents using satellite SAR and altimetry data with rotated empirical orthogonal function analysis: Case study in Tonle Sap Lake Floodplain. Remote Sensing of Environment, 241, 111732.](%s)" % url)
    url = "https://uofh-my.sharepoint.com/:b:/g/personal/cchang37_cougarnet_uh_edu/EZ70ySxmR3RDhJL5-uAHlAEBf0xI4c-BMsXQnUKT009kFA?e=tynflq"
    st.write("See here for the [Data and Procedure](%s)" % url)    
    st.write("This app has been developed by Chi-Hung Chang  & Son Do at University of Houston with supports from NOAA JPSS program.")
    st.write("Kel Markert at SERVIR Coordination Office is also acknowledged for the development of this App.")
    
    
    jpss = Image.open("https://github.com/skd862/logo/JPSS_Logo.png")
    st.image(jpss,width=50)
