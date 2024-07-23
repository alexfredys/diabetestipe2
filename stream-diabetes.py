import pickle
import streamlit as st
import numpy as np

# Load model and scaler
diabetes_model = pickle.load(open('diabetes_model.sav', 'rb'))
scaler = pickle.load(open('scaler.sav', 'rb'))

st.markdown(
    """
    <style>
    /* Center-align the title */
    div.stButton > button {
        display: block;
        margin: 0 auto;
    }
    
    </style>
    """,
    unsafe_allow_html=True
)

# Title
st.title('Prediksi Diabetes')
st.warning('Silakan jawab setiap pertanyaan dengan "1" untuk IYA dan "0" untuk TIDAK.')

# Input features

# Create two columns
col1, col2 = st.columns(2)



# Input features in the first column
with col1:
    age = st.number_input('Berapa umur Anda?', min_value=0, max_value=120, value=0)
    gender = st.radio("Jenis kelamin Anda:", ("Male", "Female"))
    polyuria = st.radio("Apakah Anda memiliki gejala terlalu sering buang air kecil?", (1, 0))
    polydipsia = st.radio("Apakah Anda memiliki gejala terlalu sering merasa haus?", (1, 0))
    sudden_weight_loss = st.radio("Apakah Anda mengalami penurunan berat badan mendadak?", (1, 0))
    weakness = st.radio("Apakah Anda sering merasa lemah?", (1, 0))
    polyphagia = st.radio("Apakah Anda sering merasa sangat lapar?", (1, 0))
    genital_thrush = st.radio("Apakah Anda mengalami infeksi jamur genital?", (1, 0))
    
# Input features in the second column
with col2:
    visual_blurring = st.radio("Apakah Anda mengalami penglihatan kabur?", (1, 0))
    itching = st.radio("Apakah Anda sering merasa gatal?", (1, 0))
    irritability = st.radio("Apakah Anda sering merasa mudah marah?", (1, 0))
    delayed_healing = st.radio("Apakah Anda mengalami penyembuhan luka yang tertunda?", (1, 0))
    partial_paresis = st.radio("Apakah Anda mengalami kelemahan sebagian pada otot?", (1, 0))
    muscle_stiffness = st.radio("Apakah Anda mengalami kekakuan otot?", (1, 0))
    alopecia = st.radio("Apakah Anda mengalami kebotakan?", (1, 0))
    obesity = st.radio("Apakah Anda mengalami obesitas?", (1, 0))
    
# Mapping gender to one-hot encoded values
gender_female = 1 if gender == 'Female' else 0
gender_male = 1 if gender == 'Male' else 0

# Validate and predict
if st.button('Prediksi'):
    # Ensure age is not empty and is a number
    try:
        # Convert input to appropriate data types
        age = float(age)
        polyuria = int(polyuria)
        polydipsia = int(polydipsia)
        sudden_weight_loss = int(sudden_weight_loss)
        weakness = int(weakness)
        polyphagia = int(polyphagia)
        genital_thrush = int(genital_thrush)
        visual_blurring = int(visual_blurring)
        itching = int(itching)
        irritability = int(irritability)
        delayed_healing = int(delayed_healing)
        partial_paresis = int(partial_paresis)
        muscle_stiffness = int(muscle_stiffness)
        alopecia = int(alopecia)
        obesity = int(obesity)
        
        # Input data as numpy array
        input_data = np.array([age, polyuria, polydipsia, sudden_weight_loss, weakness, polyphagia, 
                               genital_thrush, visual_blurring, itching, irritability, delayed_healing, 
                               partial_paresis, muscle_stiffness, alopecia, obesity, gender_female, gender_male], 
                              dtype=np.float32)

        # Reshape input data for model prediction
        input_data_reshape = input_data.reshape(1, -1)

        # Standardize the input data
        std_data = scaler.transform(input_data_reshape)

        # Prediction
        prediction = diabetes_model.predict(std_data)
        if prediction[0] == 0:
            st.success('Pasien tidak terkena diabetes')
        else:
            st.error('Pasien terkena diabetes')

    except ValueError:
        st.error('Pastikan semua input valid dan sesuai format yang diminta.')
