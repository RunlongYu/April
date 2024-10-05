from split_and_combine_twoCsv import combine_and_split_csv
from preprocess import save_npy_data, pre_geometry, grouping
from lakePreprocess import USE_FEATURES_COLUMNS

if __name__ == "__main__":
    ################
    # combine two csv file and
    combine_and_split_csv()

    ################
    # Grouping
    pre_geometry()
    grouping()

    ################
    COLUMNS_USE = USE_FEATURES_COLUMNS
    save_npy_data(COLUMNS_USE)
