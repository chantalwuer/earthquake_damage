import streamlit as st
import pandas as pd
import pickle
import os

from earthquake_damage.data.model_input import get_model_input
import seaborn as sns
import matplotlib.ticker as mtick

header = st.container()
user_input = st.container()

plot = st.container()

with header:
    st.title('Earthquake Damage Grade Prediction')
    st.subheader('Please fill in the following information to predict the damage grade of the building')

with user_input:

    ## Municipality
    st.markdown('##### Where is the building located? Please select the municipality name')
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
    muni = st.selectbox(' ', list(municipality_codes_input.keys()))
    muni_id = municipality_codes_input[muni]

    district_codes = {
        12:'Okhaldhunga',
        20:'Sindhuli',
        21:'Ramechhap',
        22:'Dolakha',
        23:'Sindhupalchok',
        24:'Kabhrepalanchok',
        28:'Nuwakot',
        29:'Rasuwa',
        30:'Dhading',
        31:'Makawanpur',
        36:'Gorkha'
}
    user_district_id = int(str(municipality_codes_input[muni])[:2])
    district_name = district_codes[user_district_id]


    all_ward ={
        'Ward number 01':'01' ,
        'Ward number 02':'02' ,
        'Ward number 03':'03' ,
        'Ward number 04':'04' ,
        'Ward number 05':'05' ,
        'Ward number 06':'06' ,
        'Ward number 07':'07' ,
        'Ward number 08':'08',
        'Ward number 09':'09' ,
        'Ward number 10':'10',
        'Ward number 11':'11',
        'Ward number 12':'12',
        'Ward number 13':'13',
        'Ward number 14':'14',
        'Ward number 15':'15',
        'Ward number 16':'16',
        'Ward number 17':'17',
        'Ward number 18':'18',
        'Ward number 19':'19',
        'Ward number 20':'20',
        }
    ward = st.selectbox(' ', list(all_ward.keys()))
    ward_num = all_ward[ward]
    ward_input = int(ward_num)
    ward_id = int(str(municipality_codes_input[muni])+ all_ward[ward])
    st.write('The ward ID is', ward_id,'.')
    st.write('The municipality ID is', muni_id,'.', 'The district ID is', user_district_id,'.','The building is located in', district_name, muni,'.')
    st.markdown('---')

    ## Building age
    st.markdown('##### How old is the building in years?')
    age_building = st.slider(' ', min_value = 0, max_value = 150, step =1, value = 50)
    st.markdown('---')

    ##Superstructure type
    st.markdown('##### What is the superstructure type of the building? Please select.')
    superstructures = {
        0: 'has_superstructure_adobe_mud',
        1: 'has_superstructure_mud_mortar_stone',
        2: 'has_superstructure_stone_flag',
        3: 'has_superstructure_cement_mortar_stone',
        4: 'has_superstructure_mud_mortar_brick',
        5: 'has_superstructure_cement_mortar_brick',
        6: 'has_superstructure_timber',
        7: 'has_superstructure_bamboo',
        8: 'has_superstructure_rc_non_engineered',
        9: 'has_superstructure_rc_engineered',
        10: 'has_superstructure_other'
}
    super_input = {v: k for k, v in superstructures.items()}
    superstructure = st.selectbox('', list(super_input.keys()))
    supernum = super_input[superstructure]
    st.write('The superstructure type ID is', supernum,'.')
    st.markdown('---')

    ## Number of floors building
    st.markdown('##### How many floors in the building?')
    count_floors = st.slider(' ', min_value = 1, max_value = 9, step =1, value = 2)
    st.markdown('---')

    ##Foundation type
    st.markdown('##### What is the material for building foundation?')
    foundation_list = ['Mud mortar-Stone/Brick', 'Cement-Stone/Brick','Bamboo/Timber', 'RC','Other']
    foundation_user = st.selectbox(' ', foundation_list)

    ##Ground floor type
    st.markdown('##### What is the material for building ground floor?')
    floor_list = ['Mud', 'Brick/Stone', 'RC', 'Timber', 'Other']
    floor_user = st.selectbox(' ', floor_list)

    ##Roof type
    st.markdown('##### What is the material for building roof?')
    roof_list = ['Bamboo/Timber-Light roof', 'Bamboo/Timber-Heavy roof','RCC/RB/RBC']
    roof_user = st.selectbox(' ', roof_list)


    user_pd = get_model_input(district_id = user_district_id,
                                municipality_id = muni_id,
                                ward = ward_input,
                                age = age_building,
                                floors = count_floors,
                                superstructure = supernum,
                                foundation = foundation_user,
                                floor = floor_user,
                                roof = roof_user)


    ## Call the model and make prediction
    file = os.path.join(os.path.dirname(__file__), 'fit_best_model.pkl')
    pickled_model = pickle.load(open(file, 'rb'))

    prediction = pd.DataFrame(pickled_model.predict_proba(user_pd))
    prediction.columns = ['Damage Grade 1', 'Damage Grade 2', 'Damage Grade 3', 'Damage Grade 4', 'Damage Grade 5']
    prediction.index = ['Probability']

    st.markdown('### Final Prediction')
    st.dataframe(prediction.style.format("{:.2%}"))


with plot:

    #Visualize the prediction
    st.markdown('### Visualize the prediction')
    visual = prediction.copy()
    ax = sns.lineplot(data=visual.T*100, markers=True, dashes=False)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.set_xticklabels(['1', '2', '3', '4', '5'])
    ax.set_xlabel('Damage Grade')
    ax.set_ylabel('Probability')
    ax.set_title('Probability of Damage Grade')
    st.pyplot(ax.get_figure())
