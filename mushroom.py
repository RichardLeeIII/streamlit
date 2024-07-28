import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.ensemble import GradientBoostingClassifier
import io


COLS = ['class', 'odor', 'gill-size', 'gill-color', 'stalk-surface-above-ring',
       'stalk-surface-below-ring', 'stalk-color-above-ring',
       'stalk-color-below-ring', 'ring-type', 'spore-print-color']

#Function to read the data
@st.cache_data(show_spinner="Fetching data...")
def read_data(file,cols):
    df= pd.read_csv(file)
    df = df[cols]

    return df

#Function to fit the LabelEncoder
@st.cache_resource
def get_target_encoder(data):
    le = LabelEncoder()
    le.fit(data['class'])

    return le

#Function to fit the OrinalEncoder
@st.cache_resource
def get_features_encoder(data):
    oe = OrdinalEncoder()
    X_cols = data.columns[1:]
    oe.fit(data[X_cols])

    return oe

#Function to encode data
@st.cache_data(show_spinner="Encoding data...")
def encode_data(data, _X_encoder, _y_encoder):
    data['class']=_y_encoder.transform(data['class'])

    X_cols = data.columns[1:]
    data[X_cols] = _X_encoder.transform(data[X_cols])

    return data

#Function to train the model
@st.cache_resource(show_spinner="Training the model...")
def train_model(data):
    X = data.drop(['class'], axis=1)
    y = data['class']

    gbc = GradientBoostingClassifier(max_depth=5, random_state=42)

    gbc.fit(X,y)

    return gbc

#Function to make a prediction
@st.cache_data(show_spinner="Making the prediction...")
def make_prediction(_model, _X_encoder, X_pred,):

    features = [each[0] for each in X_pred]
    features = np.array(features).reshape(1,-1)
    encoded_features = _X_encoder.transform(features)

    pred = _model.predict(encoded_features)

def process_uploaded_file(uploaded_file, model, feature_encoder, target_encoder):
    if uploaded_file is not None:
        # Read the uploaded file
        if uploaded_file.name.endswith('.csv'):
            data = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith('.xlsx'):
            data = pd.read_excel(uploaded_file)
        else:
            st.error("Unsupported file format")
            return None

        if set(COLS[1:]).issubset(data.columns):
            # Encode the features
            data_encoded = feature_encoder.transform(data[COLS[1:]])

            # Make predictions
            predictions = model.predict(data_encoded)

            # Decode the target to original labels
            decoded_predictions = target_encoder.inverse_transform(predictions)

            # Append predictions to the original data
            data['prediction'] = decoded_predictions

            return data
        else:
            st.error("The uploaded file does not have the required columns.")
            return None
    return None

#Function to train the model

if __name__ == "__main__":
    st.title("Mushroom classifier 🍄")
    
    #Read the data
    df = read_data('mushroom.csv', COLS)
    
    st.subheader("Step 1: Select the values for prediction")

    col1, col2, col3 = st.columns(3)

    with col1:
        odor = st.selectbox('Odor', ('a - almond', 'l - anisel', 'c - creosote', 'y - fishy', 'f - foul', 'm - musty', 'n - none', 'p - pungent', 's - spicy'))
        stalk_surface_above_ring = st.selectbox('Stalk surface above ring', ('f - fibrous', 'y - scaly', 'k - silky', 's - smooth'))
        stalk_color_below_ring = st.selectbox('Stalk color below ring', ('n - brown', 'b - buff', 'c - cinnamon', 'g - gray', 'o - orange', 'p - pink', 'e - red', 'w - white', 'y - yellow'))
    with col2:
        gill_size = st.selectbox('Gill size', ('b - broad', 'n - narrow'))
        stalk_surface_below_ring = st.selectbox('Stalk surface below ring', ('f - fibrous', 'y - scaly', 'k - silky', 's - smooth'))
        ring_type = st.selectbox('Ring type', ('e - evanescente', 'f - flaring', 'l - large', 'n - none', 'p - pendant', 's - sheathing', 'z - zone'))
    with col3:
        gill_color = st.selectbox('Gill color', ('k - black', 'n - brown', 'b - buff', 'h - chocolate', 'g - gray', 'r - green', 'o - orange', 'p - pink', 'u - purple', 'e - red', 'w - white', 'y - yellow'))
        stalk_color_above_ring = st.selectbox('Stalk color above ring', ('n - brown', 'b - buff', 'c - cinnamon', 'g - gray', 'o - orange', 'p - pink', 'e - red', 'w - white', 'y - yellow'))
        spore_print_color = st.selectbox('Spore print color', ('k - black', 'n - brown', 'b - buff', 'h - chocolate', 'r - green', 'o - orange', 'u - purple', 'w - white', 'y - yellow'))

    st.subheader("Step 2: Ask the model for a prediction")

    pred_btn = st.button("Predict", type="primary")

    #if the button is clicked
    if pred_btn:
        #fit the labelencoder
        le = get_target_encoder(df)
        #fit the ordinalencoder
        oe = get_features_encoder(df)
        #encode the data
        encoded_df = encode_data(df, oe, le)
        #train model
        gbc = train_model(encoded_df)

        x_pred = [odor, 
                  gill_size, 
                  gill_color, 
                  stalk_surface_above_ring, 
                  stalk_surface_below_ring, 
                  stalk_color_above_ring, 
                  stalk_color_below_ring, 
                  ring_type, 
                  spore_print_color]
        
        pred = make_prediction(gbc, oe, x_pred)

        nice_pred = "The mushroom is poisonous 🤢" if pred == 1 else "The mushroom is edible 🍴"

        st.write(nice_pred)


st.subheader("Step 3: Upload a CSV or Excel file for batch prediction")

uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx"])

if uploaded_file is not None:
    le = get_target_encoder(df)
    oe = get_features_encoder(df)
    encoded_df = encode_data(df, oe, le)
    gbc = train_model(encoded_df)

    result_df = process_uploaded_file(uploaded_file, gbc, oe, le)

    if result_df is not None:
        st.dataframe(result_df)
        
        # Convert the dataframe to CSV
        csv = result_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download data as CSV",
            data=csv,
            file_name='predicted_mushrooms.csv',
            mime='text/csv',
        )

        # Convert the dataframe to Excel
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            result_df.to_excel(writer, index=False, sheet_name='Sheet1')
            writer.close()
        st.download_button(
            label="Download data as Excel",
            data=output.getvalue(),
            file_name='predicted_mushrooms.xlsx',
            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        )  



