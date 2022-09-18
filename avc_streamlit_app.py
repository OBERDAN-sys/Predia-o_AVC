import streamlit as st
import numpy as np
import pickle
from PIL import Image
from sklearn.ensemble import GradientBoostingClassifier

@st.cache
def get_image(path: str) -> Image:
    image = Image.open(path)
    return image

image = get_image("AVC_DCV.jpg")
st.image(image, use_column_width=True)
st.write('Os dados para o exemplo são do site kaggle.com. Eles incluem adultos com idade de 18 a 82 anos. Este App'
         ' é apenas para fins educacionais  não pode ser usado como substituto de um conselho médico real.')

# Criação dos dicionarios
genero_dict = {"Male": 0, "Female": 1, 'Other': 2}
feature_dict = {"No": 0, "Yes": 1}
feature_dict1 = {"Rural": 0, "Urban": 1}
feature_dict2 = {'Private': 0, 'Self-employed': 1, 'children': 2, 'Govt_job': 3, 'Never_worked': 4}
feature_dict3 = {'never smoked': 0, 'Unknown': 1, 'formerly smoked': 2, 'smokes': 3}

def get_value(val, my_dict):
	for key, value in my_dict.items():
		if val == key:
			return value

def get_fvalue(val):
	feature_dict = {"No": 0, "Yes": 1}
	for key, value in feature_dict.items():
		if val == key:
			return value

def get_rvalue(val):
	feature_dict1 = {"Rural": 0, "Urban": 1}
	for key, value in feature_dict1.items():
		if val == key:
			return value

def get_hvalue(val):
	feature2_dict = {'Private': 0, 'Self-employed': 1, 'children': 2, 'Govt_job': 3, 'Never_worked': 4}
	for key, value in feature2_dict.items():
		if val == key:
			return value

def get_cvalue(val):
	feature3_dict = {'never smoked': 0, 'Unknown': 1, 'formerly smoked': 2, 'smokes': 3}
	for key, value in feature3_dict.items():
		if val == key:
			return value

# Desenvolvimento da função principal
def main():
    with st.form(key='pred-avc-form', clear_on_submit=True):
        col1, col2 = st.columns(2)
        genero = col1.selectbox('Genero', tuple(genero_dict.keys()))
        idade = col1.slider('Qual é sua idade ?', 18, 82)
        hipertensao = col1.radio('Hipertenso ?', tuple(feature_dict.keys()))
        st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>',
                 unsafe_allow_html=True)
        doença_do_coracao = col1.radio('Doença do Coração ?', tuple(feature_dict.keys()))
        st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>',
                 unsafe_allow_html=True)
        ja_se_casou = col1.radio('Já se casou?', tuple(feature_dict.keys()))
        st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>',
                 unsafe_allow_html=True)
        tipo_trabalho = col2.selectbox('Tipo de Trabalho ?', tuple(feature_dict2.keys()))
        tipo_residencia = col2.radio('Reside em área ?', tuple(feature_dict1.keys()))
        nivel_medio_glicose = col2.number_input(label='Glicose:')
        imc = col2.number_input(label='Indice Masa Corporal:')
        status_tabagismo = col2.selectbox('Status do Tabagismo', tuple(feature_dict3.keys()))

        feature_list = [get_value(genero, genero_dict), idade, get_fvalue(hipertensao), get_fvalue(doença_do_coracao),
                        get_fvalue(ja_se_casou), get_hvalue(tipo_trabalho), get_rvalue(tipo_residencia),
                        nivel_medio_glicose, imc, get_cvalue(status_tabagismo)]
        resultado = {'genero': genero, 'idade': idade, 'hipertensao': hipertensao,
                     'doença_do_coracao': doença_do_coracao, 'ja_se_casou': ja_se_casou,
                     'tipo_trabalho': tipo_trabalho, 'tipo_residencia': tipo_residencia,
                     'nivel_medio-glicose': nivel_medio_glicose, 'imc': imc, 'status_tabagismo': status_tabagismo}

        # st.json(resultado)
        single_sample = np.array(feature_list).reshape(1, -1)

        submit = st.form_submit_button(label='Faça Predição')

        if submit:
            # Carregar o classificador
            filename = 'pred_gbc.pkl'
            gbc_classifier = pickle.load(open(filename, 'rb'))

            #numpy_data = np.asarray(resultado)
            # input_reshaped = numpy_data.reshape(1, -1)
            # prediction = load_classifier.predict(input_reshaped)
            prediction_proba = gbc_classifier.predict_proba(single_sample)[0][1]

            # Escrevendo a saida
            st.subheader('Probabilidade de Acidente Vascular Cerebral(AVC)')
            st.write('Você tem uma probabilidade de {:.2f} % para risco de AVC, consulte um médico especialista!'
                     .format((prediction_proba) * 100))

if __name__ == '__main__':
    main()

