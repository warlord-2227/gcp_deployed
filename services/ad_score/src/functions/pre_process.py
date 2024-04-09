import numpy as np
import pandas as pd
import json
import sklearn
from google.cloud import storage, pubsub_v1
import google.cloud.logging
import logging

def preprocess_raw_data(file_path, sheet_name='ditty', sort_columns=['client', 'ad_name', 'audience', 'ad_format', 'campaign_objective', 'promo'], columns_to_remove=['ad_name', 'client', 'promo', 'ad_format']):
    # Load data, sort and remove unnecessary cols
    data = pd.read_excel(file_path, sheet_name=sheet_name)
    data = data.sort_values(by=sort_columns)
    data = data.drop(columns=columns_to_remove)
    return data

def pre_process_json(data):
    project_name = data['project']
    model_name = data['model']
    environment_name = data['environment']
    data_state = data['data_state']
    platform_name = data['platform']
    account_from = list(map(str, data['account_details'].keys()))[0]
    account_name = data['account_details'][account_from]['account_name']
    account_id = data['account_details'][account_from]['account_id']
    account_platform_name = data['account_details'][account_from]['account_platform_name']
    account_platform_id = data['account_details'][account_from]['account_platform_id']
    df = pd.DataFrame(data['payload']['data'])
    df['roas'] = df['purchase_roas'].apply(lambda x: x[0]['value'] if isinstance(x, list) and x else None)
    df['conversions'] = df['actions'].apply(lambda x: x[0]['value'] if isinstance(x, list) and x else None)
    df.drop(columns=['purchase_roas', 'actions'], inplace=True)
    object_columns = ['campaign_name','ad_name','objective']
    float_columns = ['spend', 'ctr','roas']
    int_columns = ['ad_id','impressions','inline_link_clicks','conversions']
    df[int_columns] = df[int_columns].fillna(0).astype('int')
    df[float_columns] = df[float_columns].astype('float')
    df['date_start'] = pd.to_datetime(df['date_start'],format='%Y-%m-%d')
    print('Min:- {}, Max:- {}, Diff:- {} Days'.format(min(df['date_start']),max(df['date_start']),(max(df['date_start'])-min(df['date_start'])).days))
    df['date_stop'] = pd.to_datetime(df['date_stop'],format='%Y-%m-%d')
    df['revenue'] = df['roas']*df['spend']
    df['days_running'] = 1
    result_df = df.groupby(['campaign_name', 'ad_name', 'objective', 'ad_id']).agg({
        'spend': 'sum',
        'revenue': 'sum',
        'impressions': 'sum',
        'inline_link_clicks': 'sum',
        'conversions': 'sum',
        'days_running': 'sum'
    }).reset_index().rename(columns={'inline_link_clicks':'clicks','objective':'campaign_objective'})
    result_df = result_df.sort_values('days_running',ascending=False)
    prospecting_regex = ['prospecting','awareness','consideration','topfunnel','acquisition','traffic','reach','pro','awa','traf','acq']
    retargeting_regex = ['retargeting','reengagement','retargeting','bottomfunnel','fullfunnel','rmk','rtg','rem','rmkt','asc']
    def tag(name):
        if any(i in name for i in prospecting_regex):
            return 'prospecting'
        elif any(i in name for i in retargeting_regex):
            return 'retargeting'
        else:
            return np.nan
    result_df['audience_campaign_name'] = result_df['campaign_name'].str.lower().apply(tag)
    result_df['audience_ad_name'] = result_df['ad_name'].str.lower().apply(tag)
    result_df['audience'] = result_df['audience_campaign_name'].combine_first(result_df['audience_ad_name']).fillna('prospecting')
    # result_df = result_df.loc[~(result_df['audience']=='others')]
    result_df['Conversions'] = result_df['conversions'].fillna(0)
    result_df['ROAS'] = result_df['revenue']/result_df['spend']
    result_df['CTR'] = result_df['clicks']/result_df['impressions']
    result_df['AOV'] = result_df['revenue']/result_df['Conversions']
    result_df['CVR'] = result_df['Conversions']/result_df['clicks']
    result_df['CPA'] = np.where(result_df['Conversions'] == 0, result_df['spend'] / (result_df['Conversions'] + 1), result_df['spend'] / result_df['Conversions'])
    result_df['AOV'] = result_df['AOV'].fillna(0)
    result_df['CVR'] = result_df['CVR'].fillna(0)
    result_df['spend_by_total_audience_spend'] = result_df['spend'] / result_df.groupby('audience')['spend'].transform('sum')
    result_df['spend_by_total_account_spend'] = result_df['spend'] / result_df.groupby('campaign_objective')['spend'].transform('sum')
    result_df['CTR_audience'] = result_df['clicks'] / result_df.groupby('audience')['impressions'].transform('sum')
    result_df['CVR_audience'] = result_df['Conversions'] / result_df.groupby('audience')['clicks'].transform('sum')
    result_df['ROAS_audience'] = result_df['revenue'] / result_df.groupby('audience')['spend'].transform('sum')
    result_df['CPA_audience'] = result_df['spend'] / result_df.groupby('audience')['Conversions'].transform('sum')
    result_df['AOV_audience'] = result_df['revenue'] / result_df.groupby('audience')['Conversions'].transform('sum')
    result_df['CTR_by_CTR_audience'] = result_df['CTR'] / result_df['CTR_audience']
    result_df['CVR_by_CVR_audience'] = result_df['CVR'] / result_df['CVR_audience']
    result_df['ROAS_by_ROAS_audience'] = result_df['revenue'] / result_df['ROAS_audience']
    result_df['CPA_by_CPA_audience'] = result_df['spend'] / result_df['CPA_audience']
    result_df['AOV_by_AOV_audience'] = result_df['revenue'] / result_df['AOV_audience']
    result_df = result_df[['ad_name','days_running','audience','campaign_objective',
        'spend','spend_by_total_audience_spend','spend_by_total_account_spend',
        'CTR','CTR_by_CTR_audience',
        'CVR','CVR_by_CVR_audience',
        'ROAS','ROAS_by_ROAS_audience',
        'CPA','CPA_by_CPA_audience',
        'Conversions',
        'AOV','AOV_by_AOV_audience']]
    result_df.replace([np.inf, -np.inf, np.nan], 0, inplace=True)
    temp_df_validate = result_df
    result_df['campaign_objective'] = result_df['campaign_objective'].str.lower()
    result_df['audience'] = result_df['audience'].str.lower()
    conditions_campaign_objective = [
        (result_df['campaign_objective'] == 'conversions'),
        (result_df['campaign_objective'] == 'outcome_sales'),
        (result_df['campaign_objective'] == 'product_catalog_sales'),
        (result_df['campaign_objective'] == 'others')
    ]
    choices_campaign_objective = [1, 2, 3, 0]
    result_df['campaign_objective'] = np.select(conditions_campaign_objective, choices_campaign_objective, default=0)
    conditions_audience = [
        (result_df['audience'] == 'prospecting'),
        (result_df['audience'] == 'retargeting'),
        (result_df['audience'] == 'others')
    ]
    choices_audience = [1, 2, 0]
    result_df['audience'] = np.select(conditions_audience, choices_audience, default=0)
    result_df[result_df['days_running']>=30].reset_index(drop=True)
    print(result_df.columns)
    
    return result_df

def transform_campaign_objective(data):
    data['campaign_objective'] = data['campaign_objective'].str.lower()
    conditions_campaign_objective = [
        (data['campaign_objective'] == 'conversions'),
        (data['campaign_objective'] == 'outcome_sales'),
        (data['campaign_objective'] == 'product_catalog_sales'),
        (data['campaign_objective'] == 'others')
    ]
    choices_campaign_objective = [1, 2, 3, 0]
    data['campaign_objective'] = np.select(conditions_campaign_objective, choices_campaign_objective, default=0)
    return data

def tag_audience(data):
    prospecting_regex = ['prospecting','awareness','consideration','topfunnel','acquisition','traffic','reach','pro','awa','traf','acq']
    retargeting_regex = ['retargeting','reengagement','retargeting','bottomfunnel','fullfunnel','rmk','rtg','rem','rmkt']
    def tag(name):
        if any(i in name for i in prospecting_regex):
            return 'prospecting'
        elif any(i in name for i in retargeting_regex):
            return 'retargeting'
        else:
            return 'other'
    data['audience'] = data['audience'].str.lower().apply(tag)
    conditions_audience = [
        (data['audience'] == 'prospecting'),
        (data['audience'] == 'retargeting'),
        (data['audience'] == 'other')
    ]
    choices_audience = [1, 2, 0]
    data['audience'] = np.select(conditions_audience, choices_audience, default=0)
    return data

def encode_target_labels(data):
    conditions_label = [
        (data['Label'] == 'Good'),
        (data['Label'] == 'OK'),
        (data['Label'] == 'Bad')
    ]
    choices_label = [0, 1, 2]
    data['Label'] = np.select(conditions_label, choices_label, default=-1)
    return data