import streamlit as st
import pandas as pd
from xgboost import XGBClassifier
from sklearn.preprocessing import OneHotEncoder, RobustScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer, make_column_transformer, make_column_selector
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
from matplotlib import pyplot as plt

header = st.container()
competition = st.container()
features = st.container()
model_training = st.container()
plot = st.container()

@st.cache
def merged_data():
    path = ('/Users/caobai/code/chantalwuer/earthquake_damage/processed_data/df_imputed.csv')
    merged = pd.read_csv(path)
    return merged

@st.cache
def load_data():
    data_labels = pd.read_csv('/Users/caobai/code/chantalwuer/earthquake_damage/raw_data/challenge_data/train_labels.csv')
    data_values = pd.read_csv('/Users/caobai/code/chantalwuer/earthquake_damage/raw_data/challenge_data/train_values.csv')
    return data_labels, data_values


@st.cache
def processed():

    y, X = load_data()

    # Make Preprocessing Pipeline
    num_transformer = make_pipeline(RobustScaler())
    num_col = make_column_selector(dtype_include=['int64', 'float64'])

    cat_transformer = make_pipeline(OneHotEncoder(handle_unknown='ignore'))
    cat_col = make_column_selector(dtype_include=['object'])

    preprocessor = make_column_transformer((num_transformer, num_col),
                                            (cat_transformer, cat_col),
                                            remainder='passthrough')

    X_processed = pd.DataFrame(preprocessor.fit_transform(X))
    y = y['damage_grade']
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    return X_processed, y_encoded

@st.cache
def smote():
    X_processed, y_encoded = processed()
    X_train, X_test, y_train, y_test = train_test_split(X_processed, y_encoded, test_size=0.3, random_state=42)
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    return X_train_smote, y_train_smote, X_test, y_test

with header:
    st.title('Earthquake Damage')
    st.text('Predicting the level of damage to buildings caused by the 2015 Gorkha earthquake in Nepal')



with competition:
    st.header('Competition Dataset Description')
    st.text('The dataset contains 260,601 observations of building characteristics and their damage levels.\nThe data was collected through surveys by the Central Bureau of Statistics that works under the National Planning Commission Secretariat of Nepal.\nThe labels are categorized into 3 classes: 1 - Low Damage, 2 - Medium Damage, and 3 - Almost Complete Destruction.')
    st.subheader('Damage Level Distribution')
    data_labels, data_values  = load_data()

    label_distribution = data_labels.damage_grade.value_counts()[data_labels.damage_grade.unique()]
    st.bar_chart(label_distribution)


with features:
    st.header('Features Preprocessing')
    st.markdown('* **Categorical Features:** We used One-Hot Encoding to encode the categorical features.\
                \n* **Numerical Features:** We used RobustScaler to scale the numerical features.\
                \n* **Missing Values:** We used SimpleImputer to fill the missing values with the most frequent values of the columns.')

# with model_training:
#     st.header('Model Training with XGBoost Classifier')

#     sel_col, disp_col = st.columns(2)

#     learning_rate = sel_col.slider('Select the learning rate(eta) for the model', min_value=0.0, max_value=1.0, step=0.1, value = 0.6)
#     max_depth = sel_col.slider('Select the maximum depth of the model', min_value=0, max_value=10, value=5, step=1)
#     subsample = sel_col.slider('Select the subsample ratio of the model', min_value=0.0, max_value=1.0, value=0.5, step=0.1)
#     gamma = sel_col.slider('Select the gamma value of the model', min_value=0.0, max_value=1.0, value=0.5, step=0.25)

#     clf = XGBClassifier(learning_rate=learning_rate, max_depth=max_depth, subsample=subsample, gamma=gamma)

#     X_train_smote, y_train_smote, X_test, y_test = smote()

#     clf.fit(X_train_smote, y_train_smote)

#     y_pred = clf.predict(X_test)

#     accuracy = clf.score(X_test, y_test)
#     precision = precision_score(y_test, y_pred, average='weighted')
#     recall = recall_score(y_test, y_pred, average='weighted')
#     f1_micro = f1_score(y_test, y_pred, average='micro')
#     confusion = confusion_matrix(y_test, y_pred)

#     disp_col.subheader('Model Performance -Accuracy')
#     disp_col.write(accuracy)
#     disp_col.subheader('Model Performance -Precision')
#     disp_col.write(precision)
#     disp_col.subheader('Model Performance -Recall')
#     disp_col.write(recall)
#     disp_col.subheader('Model Performance -F1 Score')
#     disp_col.write(f1_micro)

# with plot:
#     st.header('Confusion Matrix')
#     sel_col, disp_col = st.columns(2)
#     width = sel_col.slider('Select the width of the plot', min_value=0, max_value=10, value=5, step=1)
#     height = sel_col.slider('Select the height of the plot', min_value=0, max_value=10, value=5, step=1)
#     confusion = confusion_matrix(y_test, y_pred)
#     fig = plt.figure(figsize=(width, height))
#     sns.heatmap(confusion, annot=True, fmt='d',xticklabels=['damage_grade_1','damage_grade_2','damage_grade_3'], yticklabels=['damage_grade_1','damage_grade_2','damage_grade_3'])
#     disp_col.pyplot(fig)

############################# With Merged Dataset #####################################

house_header = st.container()

house_model_training = st.container()

@st.cache
def feature_display():
    merged = merged_data()
    a = pd.DataFrame(merged.columns.to_list()[0:11])
    b = pd.DataFrame(merged.columns.to_list()[11:22])
    c = pd.DataFrame(merged.columns.to_list()[22:33])
    d = pd.DataFrame(merged.columns.to_list()[33:44])
    return a, b, c, d

with house_header:
    st.title('Merged Dataset')
    st.text('Our team has merged the dataset of household characteristics with the building characteristics dataset.')
    st.subheader('The features in the merged dataset are:')
    merged = merged_data()
    col_1, col_2, col_3, col_4 = st.columns(4)
    a , b, c, d = feature_display()
    col_1.table(a)
    col_2.table(b)
    col_3.table(c)
    col_4.table(d)
with house_model_training:
    ## District
    # dist_name =['Okhaldhunga','Sindhuli','Ramechhap','Dolakha','Sindhupalchok','Kabhrepalanchok','Nuwakot','Rasuwa','Dhading','Makawanpur','Gorkha']
    # name = st.selectbox('select the district name', dist_name)
    # dist_id = merged.district_id.unique().tolist()
    # distict_dict = dict(zip(dist_name, dist_id))
    # input_id = distict_dict[name]
    # st.write('The district id is', input_id)

    ## Municipality
    municipality_codes = {
        1201: 'Champadevi',
        1202: 'Chisankhugadhi',
        1203: 'Khijidemba',
        1204: 'Likhu',
        1205: 'Manebhanjyang',
        1206: 'Molung',
        1207: 'Siddhicharan',
        1208: 'Sunkoshi',
        2001: 'Dudhouli',
        2002: 'Ghyanglekha',
        2003: 'Golanjor',
        2004: 'Hariharpurgadhi',
        2005: 'Kamalamai',
        2006: 'Marin',
        2007: 'Phikkal',
        2008: 'Sunkoshi',
        2009: 'Tinpatan',
        2101: 'Doramba',
        2102: 'Gokulganga',
        2103: 'Khadadevi',
        2104: 'Likhu Tamakoshi',
        2105: 'Manthali',
        2106: 'Ramechhap',
        2107: 'Sunapati',
        2108: 'Umakunda',
        2201: 'baiteshwor',
        2202: 'Bhimeshwor',
        2203: 'Bigu',
        2204: 'Gaurishankar',
        2205: 'Jiri',
        2206: 'Kalinchok',
        2207: 'Melung',
        2208: 'Sailung',
        2209: 'Tamakoshi',
        2301: 'Barhabise',
        2302: 'Balefi',
        2303: 'Bhotekoshi',
        2304: 'Chautara sangachok gadhi',
        2305: 'Helambu',
        2306: 'Indrawati',
        2307: 'Jugal',
        2308: 'Langtang National Park',
        2309: 'lisangkhu pakhar',
        2310: 'Melamchi',
        2311: 'Panchpokhari Thangpal',
        2312: 'Sunkoshi',
        2313: 'Tripurasundari',
        2401: 'Banepa',
        2402: 'Bethanchowk',
        2403: 'Bhumlu',
        2404: 'Chaurideurali',
        2405: 'Dhulikhel',
        2406: 'Khanikhola',
        2407: 'Mahabharat',
        2408: 'Mandandeupur',
        2409: 'Namobuddha',
        2410: 'Panauti',
        2411: 'Panchkhal',
        2412: 'Roshi',
        2413: 'Temal',
        2801: 'Belkotgadhi',
        2802: 'Bidur',
        2803: 'Dupcheshwar',
        2804: 'Kakani',
        2805: 'Kispang',
        2807: 'Likhu',
        2808: 'Myagang',
        2809: 'Panchakanya',
        2810: 'Shivapuri',
        2811: 'Suryagadhi',
        2812: 'Tadi',
        2813: 'Tarakeshwor',
        2901: 'Aamachhodingmo',
        2902: 'Gosaikunda',
        2903: 'Kalika',
        2904: 'Naukunda',
        2905: 'Uttargaya',
        3001: 'Benighat Rorang',
        3002: 'Dhunibesi',
        3003: 'Gajuri',
        3004: 'Galchi',
        3005: 'Gangajamuna',
        3006: 'Jwalamukhi',
        3007: 'Khaniyabash',
        3008: 'Netrawati',
        3009: 'Nilakantha',
        3010: 'Rubi Valley',
        3011: 'Siddhalek',
        3012: 'Thakre',
        3013: 'Tripurasundari',
        3101: 'Bagmati',
        3102: 'Bakaiya',
        3103: 'Bhimphedi',
        3104: 'Hetauda',
        3105: 'Indrasarowar',
        3106: 'Kailash',
        3107: 'Makawanpurgadhi',
        3108: 'Manahari',
        3109: 'Parsa Wildlife Reserve',
        3110: 'Raksirang',
        3111: 'Thaha',
        3601: 'Aarughat',
        3602: 'Ajirkot',
        3603: 'Sulikot',
        3604: 'Bhimsen',
        3605: 'Chum nubri',
        3606: 'Dharche',
        3607: 'Gandaki',
        3608: 'Gorkha',
        3609: 'Palungtar',
        3610: 'Sahid lakhan',
        3611: 'Siranchok'
}
    municipality_codes_input = {v: k for k, v in municipality_codes.items()}
    muni = st.selectbox('select the municipality name', list(municipality_codes_input.keys()))
    muni_id = municipality_codes_input[muni]
    dis =  municipality_codes_input[muni]
    st.write('The district id is', dis)
    st.write('The municipality id is', muni_id)
