import streamlit as st
import pandas as pd
import pickle
import os

from earthquake_damage.data.model_input import get_model_input
import seaborn as sns
import matplotlib.ticker as mtick
import pydeck as pdk

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

    col_1, col_2 = st.columns([2,4])

    muni = col_1.selectbox(' ', list(municipality_codes_input.keys()))
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
    ward = col_1.selectbox(' ', list(all_ward.keys()))
    ward_num = all_ward[ward]
    ward_input = int(ward_num)
    ward_id = int(str(municipality_codes_input[muni])+ all_ward[ward])



with col_1:
    st.write('The ward ID is', ward_id,'.')
    st.write('The municipality ID is', muni_id,'.')
    st.write('The district ID is', user_district_id,'.')
    st.write('The building is located in', district_name, muni,'.')

with col_2:

    city_name = municipality_codes.values()
    lat = [27.3083143,
 27.3188786,
 27.4474543,
 27.3734482,
 27.2119455,
 27.3663236,
 27.3389751,
 27.0894567,
 27.0249538,
 27.3205422,
 27.2620588,
 27.2573611,
 27.213335,
 27.2484338,
 27.1851123,
 27.0894567,
 27.1330641,
 27.5465665,
 27.5842,
 27.4858578,
 27.4492192,
 27.409272,
 27.53695,
 27.5065283,
 27.6431344,
 27.6343759,
 27.6792304,
 27.9548695,
 27.8104018,
 27.6674115,
 27.7486799,
 27.552817,
 27.6143154,
 27.5430555,
 27.8398962,
 27.7866333,
 27.932976,
 27.7513265,
 28.0059782,
 27.7856078,
 28.0033471,
 28.25,
 27.6771251,
 27.8203846,
 28.0083949,
 27.0894567,
 27.7482014,
 27.647359,
 27.5160012,
 27.627306,
 28.0,
 27.6176507,
 27.4284386,
 27.3978715,
 27.7078995,
 27.5562733,
 27.585045,
 27.6421329,
 27.4687803,
 27.5111928,
 27.8261946,
 27.9197994,
 27.9010003,
 27.8156252,
 28.0354143,
 27.3734482,
 27.9759296,
 27.8909078,
 27.8499755,
 27.9465535,
 27.9653081,
 27.7827033,
 28.0,
 27.6299765,
 27.715056,
 28.0192553,
 28.07071,
 27.747423,
 27.723534,
 27.7500422,
 27.7791686,
 28.0889647,
 27.9197712,
 28.0836189,
 27.989134,
 27.9193008,
 28.2259489,
 27.8468985,
 27.7369344,
 27.7482014,
 27.2952743,
 27.3421101,
 27.5332335,
 27.4187156,
 27.601665,
 27.6098398,
 27.4362136,
 27.5407519,
 27.348073,
 27.591167,
 27.6440592,
 28.1193887,
 28.2357608,
 28.206495,
 28.0016586,
 28.5410375,
 28.3131299,
 27.8626621,
 28.2708368,
 28.0121232,
 27.9280154,
 28.0853412]
    lng = [86.308931,
 86.6176882,
 86.3542525,
 86.2622963,
 86.4655248,
 86.4091885,
 86.528925,
 86.6473318,
 86.2145256,
 85.811476,
 86.0557226,
 85.5424065,
 85.915497,
 85.697352,
 86.2976906,
 86.6473318,
 86.1333699,
 85.9207189,
 86.277741,
 85.9402797,
 86.183151,
 86.0528813,
 86.2567909,
 85.8594835,
 86.445547,
 86.1319243,
 86.0197114,
 86.2023852,
 86.322059,
 86.27282,
 86.0937364,
 86.0342601,
 85.9771473,
 86.1212899,
 85.8734025,
 85.8024941,
 85.9058163,
 85.7267833,
 85.5313208,
 85.624202,
 85.8060427,
 85.5,
 85.8782594,
 85.5247319,
 85.6493846,
 86.6473318,
 85.9494987,
 85.5175084,
 85.4864738,
 85.7404919,
 84.0,
 85.5635152,
 85.4884677,
 85.6285562,
 85.5957623,
 85.6345507,
 85.516022,
 85.6368745,
 85.6550411,
 85.7530579,
 85.108931,
 85.1308028,
 85.2099401,
 85.2465766,
 85.1033007,
 86.2622963,
 85.096226,
 85.3012618,
 85.3652283,
 85.2362028,
 85.3379856,
 85.3010622,
 84.0,
 85.5423784,
 84.5957685,
 85.3005894,
 85.1826395,
 84.7578817,
 85.1698116,
 84.89745,
 84.9872768,
 84.9201382,
 84.7926663,
 85.06034,
 84.9722483,
 84.9298873,
 85.0821968,
 84.8367433,
 85.0635714,
 85.9494987,
 85.3649145,
 85.2090156,
 85.1135695,
 85.0100934,
 85.1871048,
 84.9529718,
 85.1151574,
 84.8055604,
 84.8333666,
 84.8306576,
 85.098317,
 84.8141466,
 84.6854822,
 84.7501842,
 84.7412557,
 84.8728096,
 84.8956092,
 84.6952753,
 84.8407583,
 84.5188437,
 84.6394667,
 84.5853382]
    map_df = pd.DataFrame({'lat':lat, 'lon':lng})
    map_df.index = city_name
    map_df = map_df.drop_duplicates()
    location = map_df.loc[[muni]]
    st.map(location, zoom = 6)


with user_input:
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
