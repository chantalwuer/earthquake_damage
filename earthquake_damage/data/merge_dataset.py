import pandas as pd

b_structure = pd.read_csv('/Users/chantalwuerschinger/code/chantalwuer/earthquake_damage/raw_data/building/csv_building_structure.csv')
b_owner_use = pd.read_csv('/Users/chantalwuerschinger/code/chantalwuer/earthquake_damage/raw_data/building/csv_building_ownership_and_use.csv')


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


def drop_and_merge(b_structure, b_owner_use):
    """
    This function drops the columns that are not needed
    for the analysis and merges the datasets.
    """
    b_structure, b_owner_use = drop_columns(b_structure, b_owner_use)
    merged = merge_dataset(b_structure, b_owner_use)
    return merged
