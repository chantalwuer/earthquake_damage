'''
Methods to create a dataframe with the user input and the average
values for the other buildings in the ward
'''

import os
import numpy as np
import pandas as pd

from earthquake_damage.ml_logic.preprocessor import fit_preprocessor

my_name = os.environ.get('MY_NAME')

def get_model_input(district_id=12, municipality_id=1201, ward=5, age=5, floors=2, superstructure=5,
                    foundation= 'Mud mortar-Stone/Brick',
                    floor= 'Mud', roof= 'Bamboo/Timber-Light roof'):
    '''
    Takes user input, creates a dataframe and processes the features
    Output is a dataframe with the processed features ready for the model
    '''
    household_comp = pd.read_csv(f'/Users/{my_name}/code/chantalwuer/earthquake_damage/processed_data/comp_data_household.csv')
    household_comp.drop(columns=['damage_grade', 'building_id'], inplace=True)

    # Determine superstructure kind
    has_superstructure_adobe_mud = 0
    has_superstructure_mud_mortar_stone = 0
    has_superstructure_stone_flag = 0
    has_superstructure_cement_mortar_stone = 0
    has_superstructure_mud_mortar_brick = 0
    has_superstructure_cement_mortar_brick = 0
    has_superstructure_timber = 0
    has_superstructure_bamboo = 0
    has_superstructure_rc_non_engineered = 0
    has_superstructure_rc_engineered = 0
    has_superstructure_other = 0

    if superstructure == 0:
        has_superstructure_adobe_mud = 1
    elif superstructure == 1:
        has_superstructure_mud_mortar_stone = 1
    elif superstructure == 2:
        has_superstructure_stone_flag = 1
    elif superstructure == 3:
        has_superstructure_cement_mortar_stone = 1
    elif superstructure == 4:
        has_superstructure_mud_mortar_brick = 1
    elif superstructure == 5:
        has_superstructure_cement_mortar_brick = 1
    elif superstructure == 6:
        has_superstructure_timber = 1
    elif superstructure == 7:
        has_superstructure_bamboo = 1
    elif superstructure == 8:
        has_superstructure_rc_non_engineered = 1
    elif superstructure == 9:
        has_superstructure_rc_engineered = 1
    elif superstructure == 10:
        has_superstructure_other = 1

    # Create ward_id
    if ward < 10:
        ward_id = str(municipality_id) + '0' + str(ward)
    else:
        ward_id = str(municipality_id) + str(ward)

    ward_id = int(ward_id)

    # Create user input
    user_input = {'district_id': district_id,
                'vdcmun_id' : municipality_id,
                'ward_id': ward_id,
                'count_floors_pre_eq': floors,
                'age_building': age,
                'has_superstructure_adobe_mud' : has_superstructure_adobe_mud,
                'has_superstructure_mud_mortar_stone': has_superstructure_mud_mortar_stone,
                'has_superstructure_stone_flag': has_superstructure_stone_flag,
                'has_superstructure_cement_mortar_stone': has_superstructure_cement_mortar_stone,
                'has_superstructure_mud_mortar_brick': has_superstructure_mud_mortar_brick,
                'has_superstructure_cement_mortar_brick': has_superstructure_cement_mortar_brick,
                'has_superstructure_timber': has_superstructure_timber,
                'has_superstructure_bamboo': has_superstructure_bamboo,
                'has_superstructure_rc_non_engineered': has_superstructure_rc_non_engineered,
                'has_superstructure_rc_engineered': has_superstructure_rc_engineered,
                'has_superstructure_other': has_superstructure_other,
                'foundation_type': foundation,
                'ground_floor_type': floor,
                'roof_type': roof,
                }

    user_input_df = pd.DataFrame(user_input, index=[ward_id])

    # Create dataframe with buildings in ward
    ward_values = household_comp.loc[household_comp['ward_id'] == ward_id]

    # Within ward, filter for age, number of floors, roof, foundation, ground floor type
    smaller_df_wards = ward_values.where(ward_values['count_floors_pre_eq'] == floors).dropna()
    age_mask = ward_values['age_building'].isin(np.arange(age-3, age+3))
    smaller_df_wards = smaller_df_wards[age_mask]
    if foundation in smaller_df_wards.foundation_type.unique():
        smaller_df_wards = smaller_df_wards.where(smaller_df_wards['foundation_type'] == foundation).dropna()

    if floor in smaller_df_wards.ground_floor_type.unique():
        smaller_df_wards = smaller_df_wards.where(smaller_df_wards['ground_floor_type'] == floor).dropna()

    if roof in smaller_df_wards.roof_type.unique():
        smaller_df_wards = smaller_df_wards.where(smaller_df_wards['roof_type'] == roof).dropna()

    # Get average data for all wards in municipality
    agg_method = {'float64': 'mean', 'int64': 'mean', 'object':  pd.Series.mode}
    data_grouped_wards = smaller_df_wards.groupby('ward_id').agg({k: agg_method[str(v)] for k, v in smaller_df_wards.dtypes.items()}).round(0)
    data_grouped_wards.drop(user_input.keys(), axis=1, inplace=True)
    # data_grouped_wards.drop(['damage_grade'], axis=1, inplace=True)

    # Get average data for chosen ward_id
    remaining_data = data_grouped_wards.loc[ward_id]
    remaining_data_df = pd.DataFrame(remaining_data).T

    # Create model input
    model_input = pd.concat([user_input_df, remaining_data_df], axis=1)

    # Preprocess model input
    preprocessor = fit_preprocessor()
    model_input_proc = preprocessor.transform(model_input)

    return pd.DataFrame(model_input_proc)
