import streamlit as st
import pandas as pd
import pickle


@st.cache_resource
def load_data():
    with open('rf_classifier.pkl', 'rb') as model_file:
        model = pickle.load(model_file)

    # load the df
    df = pd.read_csv('./cleaned_data.csv')
    return model, df

# Load the model, and dataframe
model, df = load_data()


encoded_regions = {
    'DAKAR' : 0,
  'DIOURBEL' : 1,
  'FATICK' : 2,
  'KAFFRINE' : 3,
  'KAOLACK' : 4,
  'KEDOUGOU' : 5,
  'KOLDA' : 6,
  'LOUGA' : 7,
  'MATAM' : 8,
  'SAINT-LOUIS' : 9,
  'SEDHIOU' : 10,
  'TAMBACOUNDA' : 11,
  'THIES' : 12,
  'ZIGUINCHOR' : 13
}

encoded_mrg = {
    "NO": 0
}


# Create a Streamlit app
st.title("Churn Prediction App")
st.write("This app predicts whether a customer will churn or not based on their data.")


# Create input fields for the user to enter data
region = encoded_regions[st.selectbox("Region", options=df["REGION"].unique())]
montant = st.number_input("Montant", min_value=df["MONTANT"].min(), max_value=df["MONTANT"].max())
frequency_rech = st.slider("Frequency of Recharge", min_value=df["FREQUENCE_RECH"].min(), max_value=df["FREQUENCE_RECH"].max())
revenue = st.slider("Revenue", min_value=df["REVENUE"].min(), max_value=df["REVENUE"].max())
arpu_segment = st.selectbox("ARPU Segment", options=df["ARPU_SEGMENT"].unique())
frequency = st.slider("Frequency", min_value=df["FREQUENCE"].min(), max_value=df["FREQUENCE"].max())
data_volume = st.slider("Data Volume", min_value=df["DATA_VOLUME"].min(), max_value=df["DATA_VOLUME"].max())
on_net = st.slider("On Net", min_value=df["ON_NET"].min(), max_value=df["ON_NET"].max())
orange = st.slider("Orange", min_value=df["ORANGE"].min(), max_value=df["ORANGE"].max())
tigo = st.slider("Tigo", min_value=df["TIGO"].min(), max_value=df["TIGO"].max())
zone1 = st.slider("Zone 1", min_value=df["ZONE1"].min(), max_value=df["ZONE1"].max())
zone2 = st.slider("Zone 2", min_value=df["ZONE2"].min(), max_value=df["ZONE2"].max())
mrg = encoded_mrg[st.selectbox("MRG", options=df["MRG"].unique())]
regularity = st.slider("Regularity", min_value=df["REGULARITY"].min(), max_value=df["REGULARITY"].max())
freq_top_pack = st.slider("Frequency of Top Pack", min_value=df["FREQ_TOP_PACK"].min(), max_value=df["FREQ_TOP_PACK"].max())


# Create a DataFrame with the input data
input_data = pd.DataFrame({
    "REGION": [region],
    "MONTANT": [montant],
    "FREQUENCE_RECH": [frequency_rech],
    "REVENUE": [revenue],
    "ARPU_SEGMENT": [arpu_segment],
    "FREQUENCE": [frequency],
    "DATA_VOLUME": [data_volume],
    "ON_NET": [on_net],
    "ORANGE": [orange],
    "TIGO": [tigo],
    "ZONE1": [zone1],
    "ZONE2": [zone2],
    "MRG": [mrg],
    "REGULARITY": [regularity],
    "FREQ_TOP_PACK": [freq_top_pack]
})

#use button to make prediction
if st.button("Predict"):
    # Make a prediction
    prediction = model.predict(input_data)[0]
    # Display the prediction
    if prediction == 1:
        st.write("The model predicts that the customer will churn.")
    else:
        st.write("The model predicts that the customer will not churn.")