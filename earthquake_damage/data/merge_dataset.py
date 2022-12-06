import pandas as pd
import os

my_name = os.environ.get('MY_NAME')

# Methods to drop columns we do not want and merge remaining columns

def drop_columns(b_structure, b_owner_use):
    """
    This function drops the columns that are
    not needed for the analysis.
    """

    drop_structure = ['count_floors_post_eq',
            'plinth_area_sq_ft',
            'height_ft_pre_eq',
            'height_ft_post_eq',
            'condition_post_eq',
            'technical_solution_proposed']

    b_structure = b_structure.drop(drop_structure, axis=1)

    drop_owner = ['district_id', 'vdcmun_id', 'ward_id']

    b_owner_use = b_owner_use.drop(drop_owner, axis=1)

    return b_structure, b_owner_use


def merge_dataset(b_structure, b_owner_use):
    """
    This function merges the two smaller datasets into one dataset.
    """
    merged = pd.merge(b_structure, b_owner_use, on='building_id', how='inner')
    return merged


def drop_and_merge():
    """
    This function drops the columns that are not needed
    for the analysis and merges the datasets.
    """

    b_structure = pd.read_csv(f'/Users/{my_name}/code/chantalwuer/earthquake_damage/raw_data/building/csv_building_structure.csv')
    b_owner_use = pd.read_csv(f'/Users/{my_name}/code/chantalwuer/earthquake_damage/raw_data/building/csv_building_ownership_and_use.csv')

    b_structure, b_owner_use = drop_columns(b_structure, b_owner_use)
    merged = merge_dataset(b_structure, b_owner_use)

    path = f'/Users/{my_name}/code/chantalwuer/earthquake_damage/processed_data/comp_data_full.csv'
    merged.to_csv(path, index=False)

    print(f"✅ comp_data_full saved to /processed_data/comp_data_full.csv")

    return None

def refine_demographics():

    '''
    Method that loads csv_building_demographics.csv and drops columns that are not needed for the analysis
    Transforms some of the categorical variables into numerical variables
    And adds the building_id
    Saves as csv
    '''

    h_demographics = pd.read_csv(f'/Users/{my_name}/code/chantalwuer/earthquake_damage/raw_data/household/csv_household_demographics.csv')

    # Drop columns
    h_dem_drop = ['district_id', 'ward_id',
                  'vdcmun_id', 'is_bank_account_present_in_household',
                  'caste_household']

    h_demographics = h_demographics.drop(h_dem_drop, axis=1)

    # print(f"Adapting income level...")

    # # Income Level
    # h_demographics.income_level_household = h_demographics.income_level_household.replace({
    #                 'Rs. 10 thousand': 0, 'Rs. 10-20 thousand': 1,
    #                 'Rs. 20-30 thousand': 2, 'Rs. 30-50 thousand': 3,
    #                 'Rs. 50 thousand or more': 4})

    print(f"Adapting education level...")

    # Education
    in_education = ['Class 5', 'Class 4',
        'Class 10', 'Class 9',
        'Class 7', 'Class 2', 'Class 1',
        'Class 8', 'Class 3', 'Class 6', 'Nursery/K.G./Kindergarten'
       ]

    university = ['Bachelors or equivalent',
                  'Masters or equivalent',
                  'Ph.D. or equivalent']

    h_demographics.education_level_household_head = h_demographics.education_level_household_head.\
        map(lambda x: 'In education' if x in in_education else x)
    h_demographics.education_level_household_head = h_demographics.education_level_household_head.\
        map(lambda x: 'University' if x in university else x)
    h_demographics.education_level_household_head = h_demographics.education_level_household_head.\
        map(lambda x: 'High School' if x == 'SLC or equivalent' else x)


    # Add building_id
    print(f"Adding building_id...")

    mapping = pd.read_csv(f'/Users/{my_name}/code/chantalwuer/earthquake_damage/raw_data/geographical/mapping.csv')
    mapping = mapping.drop(columns=['individual_id'], axis=1)
    mapping = mapping.drop_duplicates()

    h_demographics_ids = pd.merge(h_demographics, mapping, on='household_id', how='inner')

    print(f"Saving the file to csv...")

    path = f'/Users/{my_name}/code/chantalwuer/earthquake_damage/processed_data/h_demographics_ids.csv'
    h_demographics_ids.to_csv(path, index=False)

    print(f"✅ h_demographics_ids with {h_demographics_ids.shape} saved to /processed_data/h_demographics_ids.csv")

    return None


def merge_household_building():
    '''
    Takes comp_data_full (building data) and h_demographics_ids (household data)
    Merges on building_id
    Saves as csv
    '''

    print(f"Loading household_demographics_ids.csv...")
    household = pd.read_csv(f'/Users/{my_name}/code/chantalwuer/earthquake_damage/processed_data/h_demographics_ids.csv')

    print(f"Loading comp_data_full.csv...")
    building = pd.read_csv(f'/Users/{my_name}/code/chantalwuer/earthquake_damage/processed_data/comp_data_full.csv')

    print(f"Merging datasets...")
    comp_data_household = building.merge(household, on='building_id', how='inner')

    print(f"Saving the file to csv...")
    comp_data_household.to_csv(f'/Users/{my_name}/code/chantalwuer/earthquake_damage/processed_data/comp_data_household.csv', index=False)

    print(f"✅ comp_data_household with {comp_data_household.shape} saved to /processed_data/comp_data_household.csv")

    return None
