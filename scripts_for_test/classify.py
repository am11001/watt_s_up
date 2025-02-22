from tsfresh import extract_features
from tsfresh.feature_extraction import EfficientFCParameters
import pandas as pd


def get_features(df, label, params = EfficientFCParameters()):
    df_features = extract_features(
        df.reset_index().melt(id_vars = 'timestamp', var_name = 'id', value_name = 'value'),
        column_id = 'id',
        column_sort = 'timestamp',
        column_value = 'value',
        default_fc_parameters = params
    )
    df_features['label'] = label
    return df_features


def create_df_features(df_real, df_synth):
    df_real_features = get_features(df_real, 1)
    df_synth_features = get_features(df_synth, 0)
    df_features = pd.concat([df_real_features, df_synth_features])
    df_features = df_features.loc[:, df_features.isna().sum() == 0]
    return df_features