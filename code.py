import numpy as np
import torch
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import os
from sklearn import linear_model
from sklearn.metrics import r2_score, mean_squared_error
import sys
module_path = os.path.abspath(os.path.join(".."))
if module_path not in sys.path:
    sys.path.append(module_path)
import numpy as np
from utils.logger import PrettyLogger
from utils.date import str2date, int2date_delta, date2str
from utils.io_func import save_to_npy, load_from_tiff
from config import (
    START_V_I, START_H_I, SIDE_LEN, INTRPL_START_DATE_STR, INTRPL_END_DATE_STR
)
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches


logger = PrettyLogger()

SITE = "Site_A"
YEAR = "2015"
DATA_DIR = "../data/{}/ARD/{}/".format(SITE, YEAR)
OUTPUT_DIR = "./out/{}/ARD/cropped_interpolated/{}/".format(SITE, YEAR)
AVAI_PATH = os.path.join(OUTPUT_DIR, "availability.npy")
FILTER_BAND_PATH = os.path.join(OUTPUT_DIR, "filter_band.npy")
INTERPOLATED_PATH = os.path.join(OUTPUT_DIR, "interpolated.npy")
FINAL_OUTOUT_FILEPATH = "./out/{}/x-{}.npy".format(SITE, YEAR)
# link the filenames to date
date_filename_dict = {}
for filename in sorted(os.listdir(DATA_DIR)):
    date = str2date(filename[15:23])
    if (
        date >= str2date("{}{}".format(YEAR, INTRPL_START_DATE_STR))
        and date <= str2date("{}{}".format(YEAR, INTRPL_END_DATE_STR))
    ):
        if date not in date_filename_dict.keys():
            date_filename_dict[date] = []
        date_filename_dict[date].append(filename)
        

 #read ARD images, crop ARD images and detect invalid values
raw_dates = sorted(date_filename_dict.keys())
availability = np.zeros((SIDE_LEN, SIDE_LEN, len(raw_dates)))
valid = np.zeros((SIDE_LEN, SIDE_LEN, len(raw_dates), 6))

to_fill = np.vectorize(lambda x: int("{:011b}".format(x)[-1], 2))
to_clear = np.vectorize(lambda x: int("{:011b}".format(x)[-2], 2))
to_cloud_shadow = np.vectorize(lambda x: int("{:011b}".format(x)[-4], 2))
to_cloud = np.vectorize(lambda x: int("{:011b}".format(x)[-6], 2))
for i, date in enumerate(raw_dates):
    logger.info("Loading: {}/{}".format(i+1, len(raw_dates)), date2str(date))
    sr_bands = []
    for filename in date_filename_dict[date]:
        band = load_from_tiff(os.path.join(DATA_DIR, filename))[
            START_V_I:START_V_I+SIDE_LEN, START_H_I:START_H_I+SIDE_LEN
        ]
        if filename[-11:-4] != "PIXELQA":
            sr_bands.append(band)
        else:
            qa_band = band
    sr_bands = np.array(sr_bands).transpose((1, 2, 0))

    flag_sr_range = ((sr_bands >= 0) & (sr_bands <= 10000)).all(axis=2)
    fill_band = to_fill(qa_band)
    flag_fill = (fill_band == 0)
    clear_band = to_clear(qa_band)
    flag_clear = (clear_band == 1)
    cloud_shadow_band = to_cloud_shadow(qa_band)
    flag_cloud_shadow = (cloud_shadow_band == 0)
    cloud_band = to_cloud(qa_band)
    
for i, date in enumerate(raw_dates):
    logger.info("Loading: {}/{}".format(i+1, len(raw_dates)), date2str(date))
    sr_bands = []
    for filename in date_filename_dict[date]:
        band = load_from_tiff(os.path.join(DATA_DIR, filename))[
            START_V_I:START_V_I+SIDE_LEN, START_H_I:START_H_I+SIDE_LEN
        ]
        if filename[-11:-4] != "PIXELQA":
            sr_bands.append(band)
        else:
            qa_band = band
    sr_bands = np.array(sr_bands).transpose((1, 2, 0))

    flag_sr_range = ((sr_bands >= 0) & (sr_bands <= 10000)).all(axis=2)
    fill_band = to_fill(qa_band)
    flag_fill = (fill_band == 0)
    clear_band = to_clear(qa_band)
    flag_clear = (clear_band == 1)
    cloud_shadow_band = to_cloud_shadow(qa_band)
    flag_cloud_shadow = (cloud_shadow_band == 0)
    cloud_band = to_cloud(qa_band)
    flag_cloud = (cloud_band == 0)
    flag = flag_sr_range*flag_fill*flag_clear*flag_cloud_shadow*flag_cloud

    availability[:, :, i] = flag

    # make invalid observations zero, only for the convenience of debugging
    valid[:, :, i, :] = sr_bands
    valid[:, :, i, :] = valid[:, :, i, :]*(flag.reshape(*flag.shape, 1))

save_to_npy(availability, AVAI_PATH)
"""
========== PIXEL FILTER METHOD BY AVAILABILITY ==========
If the number of valid observations after May 15 >= 7,
the pixel will be included in the dataset, otherwise it will be excluded.
"""

index4filter = raw_dates.index(list(filter(
    lambda x: x > str2date("{}0515".format(YEAR)), raw_dates
))[0])
filter_band = availability[:, :, index4filter:].sum(axis=2) >= 7
logger.info("Validity percentage ({} {}): {:.4f}".format(
    SITE, YEAR,
    filter_band.sum()/(filter_band.shape[0]*filter_band.shape[1])
))
save_to_npy(filter_band, FILTER_BAND_PATH)
# prepare target dates for interpolation
intrpl_start_date = str2date("{}{}".format(YEAR, INTRPL_START_DATE_STR))
intrpl_end_date = str2date("{}{}".format(YEAR, INTRPL_END_DATE_STR))
intrpl_delta_days = list(range(
    0, (intrpl_end_date - intrpl_start_date).days + 1, 7
))

'''
========== INTERPOLATION METHOD ==========
situation I (normal): d_1, d_2*, target, d_3*, d_4, ...
situation II (close to the start date): target, d_1*, d_2*, d_3, ...
situation III (close to the end date): d_1, d_2, ..., d_(-2), d_(-1), target
'''
# prepare target dates for interpolation
intrpl_start_date = str2date("{}{}".format(YEAR, INTRPL_START_DATE_STR))
intrpl_end_date = str2date("{}{}".format(YEAR, INTRPL_END_DATE_STR))
intrpl_delta_days = list(range(
    0, (intrpl_end_date - intrpl_start_date).days + 1, 7
))
intrpl_dates = [
    int2date_delta(intrpl_delta_day) + intrpl_start_date
    for intrpl_delta_day in intrpl_delta_days
]
interpolated = np.zeros((SIDE_LEN, SIDE_LEN, len(intrpl_dates), 6))
for intrpl_date_index, intrpl_date in enumerate(intrpl_dates):
    logger.info("Interpolating: {}/{} {} ".format(
        intrpl_date_index + 1, len(intrpl_dates), date2str(intrpl_date))
    )
    # descending/ascending order for searching the nearest day before/after
    before_dates = list(filter(lambda x: x <= intrpl_date, raw_dates))[::-1]
    after_dates = list(filter(lambda x: x >= intrpl_date, raw_dates))

    for i in range(SIDE_LEN):
        for j in range(SIDE_LEN):

            # filter invalid pixel
            if not filter_band[i, j]:
                continue

            # situation I
            d_1 = None
            for nearest_before_index, before_date in enumerate(before_dates):
                before_date_raw_index = raw_dates.index(before_date)
                if availability[i, j][before_date_raw_index]:
                    d_1 = before_date
                    date_raw_index_1 = before_date_raw_index
                    break
            d_2 = None
            for nearest_after_index, after_date in enumerate(after_dates):
                after_date_raw_index = raw_dates.index(after_date)
                if availability[i, j][after_date_raw_index]:
                    d_2 = after_date
                    date_raw_index_2 = after_date_raw_index
                    break

            # situation II: search the second nearest after date
            if not d_1:
                for after_date in after_dates[nearest_after_index+1:]:
                    after_date_raw_index = raw_dates.index(after_date)
                    if availability[i, j][after_date_raw_index]:
                        d_1 = after_date
                        date_raw_index_1 = after_date_raw_index
                        break

            # situation III: search the second nearest before date
            if not d_2:
                for before_date in before_dates[nearest_before_index+1:]:
                    before_date_raw_index = raw_dates.index(before_date)
                    if availability[i, j][before_date_raw_index]:
                        d_2 = before_date
                        date_raw_index_2 = before_date_raw_index
                        break

            interpolated[i][j][intrpl_date_index] = [np.interp(
                (intrpl_date-d_1).days,
                [0, (d_2-d_1).days],
                [valid[i, j, date_raw_index_1, band_index],
                    valid[i, j, date_raw_index_2, band_index]]
            ) for band_index in range(6)]

save_to_npy(interpolated, INTERPOLATED_PATH)
x = interpolated[filter_band]
save_to_npy(x, FINAL_OUTOUT_FILEPATH)
intrpl_dates = [
    int2date_delta(intrpl_delta_day) + intrpl_start_date
    for intrpl_delta_day in intrpl_delta_days
]

class DCM(nn.Module):
    def __init__(
        self, seed, input_feature_size, hidden_size, num_layers,
        bidirectional, dropout, num_classes
    ):
        super().__init__()
        self._set_reproducible(seed)

        self.lstm = nn.LSTM(
            input_size=input_feature_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            batch_first=True,
            dropout=dropout,
        )  # i/o: (batch, seq_len, num_directions*input_/hidden_size)
        num_directions = 2 if bidirectional else 1
        self.attention = nn.Linear(
            in_features=num_directions * hidden_size,
            out_features=1,
        )
        self.fc = nn.Linear(
            in_features=num_directions * hidden_size,
            out_features=num_classes,
        )

    def _set_reproducible(self, seed, cudnn=False):
        np.random.seed(seed)
        torch.manual_seed(seed)
        if cudnn:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def forward(self, x):
        self.lstm.flatten_parameters()
        # lstm_out: (batch, seq_len, num_directions*hidden_size)
        lstm_out, _ = self.lstm(x)
        # softmax along seq_len axis
        attn_weights = F.softmax(F.relu(self.attention(lstm_out)), dim=1)
        # attn (after permutation): (batch, 1, seq_len)
        fc_in = attn_weights.permute(0, 2, 1).bmm(lstm_out)
        fc_out = self.fc(fc_in)
        return fc_out.squeeze(), attn_weights.squeeze()
    
## CSDL Layer Code
# if needed, change these to the directories containing your data and the name of your data file
nass_dir = '../data/nass_counties/'
cdl_dir = '../data/cdl_counties/'
csdl_dir = '../data/csdl_counties/'

nass_corn = pd.read_csv(os.path.join(nass_dir, 'NASS_cropAreaCorn_1999to2018_raw.csv'))
nass_soy = pd.read_csv(os.path.join(nass_dir, 'NASS_cropAreaSoy_1999to2018_raw.csv'))

cols = ['state_fips_code', 'county_code', 'year', 'state_alpha', 
        'class_desc', 'short_desc','statisticcat_desc', 'commodity_desc',
        'util_practice_desc', 'Value']
nass_soy = nass_soy[cols]
nass_corn = nass_corn[cols]
nass = pd.concat([nass_corn, nass_soy])
print(nass.shape)
nass.head()
# Add the unique county FIPS code: stateFIPS+countyFIPS
nass['county_code'] = nass['county_code'].map(int).apply(lambda x: '{0:0>3}'.format(x))
nass['state_fips_code'] = nass['state_fips_code'].apply(lambda x: '{0:0>2}'.format(x))
nass['fips'] = (nass['state_fips_code'].map(str)+nass['county_code'].map(str)).map(int)

nass['commodity_desc'] = nass['commodity_desc'].str.title()
nass = nass.rename(columns={"commodity_desc":"crop", "state_alpha":"state", "Value":"Nass_Area_acres"})
nass['Nass_Area_acres'] = nass['Nass_Area_acres'].str.replace(',', '').astype('float')
nass["Nass_Area_m2"] = nass["Nass_Area_acres"] * 4046.86
nass = nass[['fips', 'year', 'state', 'crop', 'Nass_Area_m2']]
nass.head()
cdl = pd.DataFrame()
for filename in sorted(os.listdir(cdl_dir)):
    if (filename.endswith('.csv')):
        temp = pd.read_csv(os.path.join(cdl_dir, filename)).drop(['.geo','system:index'],axis=1)
        cdl = pd.concat([temp, cdl], sort=True)
cdl = cdl[cdl['masterid'] != 0] # drop dummy feature
print(cdl.shape)
cdl.head()
# compute CDL coverage by county
classes = list(set(cdl.columns.tolist()) - set(['Year', 'area_m2', 'masterid', 'COUNTYFP', 'STATEFP']))
other = list(set(classes) - set(['1', '5']))
cdl['cdl_coverage'] = cdl[classes].sum(axis=1)
cdl['Other'] = cdl[other].sum(axis=1)
cdl = cdl.drop(other, axis=1)
cdl.head()
maxcoverage = cdl[cdl['Year'] == 2018][['masterid', 'cdl_coverage']].rename(
    {'cdl_coverage': '2018cdl_coverage'}, axis=1)
cdl = cdl.merge(maxcoverage, on='masterid')
cdl['CDL_perccov'] = cdl['cdl_coverage'] / cdl['2018cdl_coverage'] * 100
cdl_key = {'1': "Corn", '5': "Soybeans", 'Year': "year"}  
cdl = cdl.rename(cdl_key, axis=1)
cdl.head()
crops = ["Corn", "Soybeans", "Other"]
cdl = pd.melt(cdl, id_vars=['masterid', 'year', 'COUNTYFP', 'STATEFP', 'CDL_perccov'], value_vars=crops, value_name='CDL_Area_m2')
cdl = cdl.rename(columns={"variable": "crop", "masterid": "fips"})
cdl.head()
csdl = pd.DataFrame()
for filename in sorted(os.listdir(csdl_dir)):
    if (filename.endswith('.csv')):
        temp = pd.read_csv(os.path.join(csdl_dir, filename))
        csdl = pd.concat([temp, csdl], sort=True)
csdl = csdl.dropna(subset=['masterid'])
csdl[['COUNTYFP', 'STATEFP', 'masterid', 'year']] = csdl[['COUNTYFP', 'STATEFP', 'masterid', 'year']].astype(int)
csdl = csdl[['0', '1', '5', 'COUNTYFP', 'STATEFP', 'masterid', 'year']]
print(csdl.shape)
csdl.head()
csdl = csdl.merge(maxcoverage, on='masterid')
csdl['CSDL_coverage'] = csdl[['0','1','5']].sum(axis=1)
csdl['CSDL_perccov'] = csdl['CSDL_coverage'] / csdl['2018cdl_coverage']*100
# note that CDL covers lakes, rivers; CSDL does not
csdl_key = {'0':"Other",'1': "Corn", '5': "Soybeans"} 
csdl = csdl.rename(csdl_key,axis=1)
csdl.head()
crops = ["Corn", "Soybeans", "Other"]
csdl = pd.melt(csdl, id_vars=['masterid', 'year', 'COUNTYFP', 'STATEFP', 'CSDL_perccov'], value_vars=crops, value_name='CSDL_Area_m2')
csdl = csdl.rename(columns={"variable": "crop", "masterid": "fips"})
csdl.head()
df = nass.merge(cdl, on=['year', 'fips', 'crop'], how='left').merge(
    csdl, on=['year', 'fips', 'crop', 'COUNTYFP', 'STATEFP'], how='left')

# convert m2 to km2: 1e6
df['Nass_Area_km2'] = df['Nass_Area_m2'] / 1e6
df['CDL_Area_km2'] = df['CDL_Area_m2'] / 1e6
df['CSDL_Area_km2'] = df['CSDL_Area_m2'] / 1e6

print(df.shape)
df.head()
recent_years = np.arange(2008, 2019)
past_years = np.arange(1999, 2008)
df['year_group'] = '1999-2007'
df.loc[df['year'].isin(recent_years),'year_group'] = '2008-2018'

df_sub = df[df['CDL_perccov'] > 95]
print(df.shape)
print(df_sub.shape)

color_mapCSDL = { "Corn": 'orange',  "Soybeans": 'green'}
color_mapCDL = { "Corn": 'brown',  "Soybeans": 'darkseagreen'}
fsize = 15

lm = linear_model.LinearRegression()
bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.0)

fig, ax = plt.subplots(nrows=2, ncols=4,figsize=(16,8))

for i,(id,group) in enumerate(df_sub[df_sub['year_group']=='1999-2007'].groupby(['crop'])):
            
    ax[i,0].scatter(group['CDL_Area_km2'],group['Nass_Area_km2'], color = color_mapCDL[id], alpha=0.3, label=id)
    ax[i,1].scatter(group['CSDL_Area_km2'],group['Nass_Area_km2'], color = color_mapCSDL[id], alpha=0.3, label=id)

    group1= group.dropna(subset=['CDL_Area_km2','Nass_Area_km2'])
    R2 = r2_score(group1['CDL_Area_km2'].values,group1['Nass_Area_km2'].values)
    ax[i,0].text(0.05, 0.9, '$R^2$={0:.3f}'.format(R2), ha="left", va="center", size=fsize, bbox=bbox_props, transform=ax[i,0].transAxes)

    group2= group.dropna(subset=['CSDL_Area_km2','Nass_Area_km2'])
    R2 = r2_score(group2['CSDL_Area_km2'].values,group2['Nass_Area_km2'].values)
    ax[i,1].text(0.05, 0.9, '$R^2$={0:.3f}'.format(R2), ha="left", va="center", size=fsize, bbox=bbox_props, transform=ax[i,1].transAxes)

    lims = [0,2.6e+9/1e6] # km2
    # now plot both limits against each other
    ax[i,0].plot(lims, lims, 'k--', alpha=0.75, zorder=0)
    ax[i,1].plot(lims, lims, 'k--', alpha=0.75, zorder=0)
    ax[i,0].set_xlim(lims)
    ax[i,1].set_xlim(lims)
    ax[i,0].set_ylim(lims)
    ax[i,1].set_ylim(lims)  
    ax[i,0].set_aspect('equal')
    ax[i,1].set_aspect('equal')
    ax[0,0].set_xticklabels('', rotation=90)
    ax[0,1].set_xticklabels('', rotation=90)
    ax[i,1].set_yticklabels('', rotation=90)

recent_years = np.arange(2008, 2019)
past_years = np.arange(1999, 2008)
df['year_group'] = '1999-2007'
df.loc[df['year'].isin(recent_years),'year_group'] = '2008-2018'

df_sub = df[df['CDL_perccov'] > 95]
print(df.shape)
print(df_sub.shape)
for i,(id,group) in enumerate(df_sub[df_sub['year_group']=='2008-2018'].groupby(['crop'])):
       
    ax[i,2].scatter(group['CDL_Area_km2'],group['Nass_Area_km2'], color = color_mapCDL[id], alpha=0.3, label=id)
    ax[i,3].scatter(group['CSDL_Area_km2'],group['Nass_Area_km2'], color = color_mapCSDL[id], alpha=0.3, label=id)
    
    # set background to gray and alpha=0.2
    ax[i,2].set_facecolor((0.5019607843137255, 0.5019607843137255, 0.5019607843137255, 0.2))
    ax[i,3].set_facecolor((0.5019607843137255, 0.5019607843137255, 0.5019607843137255, 0.2))
    
    group1= group.dropna(subset=['CDL_Area_km2','Nass_Area_km2'])
    R2 = r2_score(group1['CDL_Area_km2'].values, group1['Nass_Area_km2'].values)
    ax[i,2].text(0.05, 0.9, '$R^2$={0:.3f}'.format(R2), ha="left", va="center", size=fsize, bbox=bbox_props, transform=ax[i,2].transAxes)

    group2= group.dropna(subset=['CSDL_Area_km2','Nass_Area_km2'])
    R2 = r2_score(group2['CSDL_Area_km2'].values, group2['Nass_Area_km2'].values)
    ax[i,3].text(0.05, 0.9, '$R^2$={0:.3f}'.format(R2), ha="left", va="center", size=fsize, bbox=bbox_props, transform=ax[i,3].transAxes)

    lims = [0,2.6e+9/1e6] # area in km2
    # now plot both limits against each other
    ax[i,2].plot(lims, lims, 'k--', alpha=0.75, zorder=0)
    ax[i,3].plot(lims, lims, 'k--', alpha=0.75, zorder=0)
    ax[i,2].set_xlim(lims)
    ax[i,3].set_xlim(lims)
    ax[i,2].set_ylim(lims)
    ax[i,3].set_ylim(lims)    
    ax[i,2].set_aspect('equal')
    ax[i,3].set_aspect('equal')
    ax[0,2].set_xticklabels('', rotation=90)
    ax[0,3].set_xticklabels('', rotation=90)
    ax[i,2].set_yticklabels('', rotation=90)
    ax[i,3].set_yticklabels('', rotation=90)

ax[0,0].set_ylabel('NASS county area [$km^2$]',fontsize=fsize)    
ax[1,0].set_ylabel('NASS county area [$km^2$]',fontsize=fsize)   
ax[1,0].set_xlabel('CDL county area [$km^2$]',fontsize=fsize)
ax[1,1].set_xlabel('CSDL county area [$km^2$]',fontsize=fsize)
ax[1,2].set_xlabel('CDL county area [$km^2$]',fontsize=fsize)
ax[1,3].set_xlabel('CSDL county area [$km^2$]',fontsize=fsize)
ax[0,0].set_title('1999-2007',fontsize=fsize) 
ax[0,1].set_title('1999-2007',fontsize=fsize) 
ax[0,2].set_title('2008-2018',fontsize=fsize) 
ax[0,3].set_title('2008-2018',fontsize=fsize) 

# Create the legend
legend_elements = [ Line2D([0], [0], color='k', linewidth=1, linestyle='--', label='1:1'),
                    Line2D([0], [0], marker='o', color='w', markerfacecolor='brown', markersize=15, label='Corn CDL'),
                    Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', markersize=15, label='Corn CSDL'),
                    Line2D([0], [0], marker='o', color='w', markerfacecolor='darkseagreen', markersize=15, label='Soybeans CDL'),
                    Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=15, label='Soybeans CSDL'),
                    mpatches.Patch(facecolor='gray', edgecolor='gray', alpha = 0.2, label='NASS informed by CDL')]
fig.legend(handles=legend_elements,
           loc='lower center',  
           bbox_to_anchor=(0.5,-0.01),
           fontsize = 'x-large',
           ncol=7)

fig.tight_layout(rect=[0,0.05,1,1]) # legend on the bottom

# if needed, change these to the directories containing your data and the name of your data file
nass_dir = '../data/nass_counties/'
cdl_dir = '../data/cdl_counties/'
csdl_dir = '../data/csdl_counties/'
nass_corn = pd.read_csv(os.path.join(nass_dir, 'NASS_cropAreaCorn_1999to2018_raw.csv'))
nass_soy = pd.read_csv(os.path.join(nass_dir, 'NASS_cropAreaSoy_1999to2018_raw.csv'))

cols = ['state_fips_code', 'county_code', 'year', 'state_alpha', 
        'class_desc', 'short_desc','statisticcat_desc', 'commodity_desc',
        'util_practice_desc', 'Value']
nass_soy = nass_soy[cols]
nass_corn = nass_corn[cols]
nass = pd.concat([nass_corn, nass_soy])
print(nass.shape)
nass.head()
# Add the unique county FIPS code: stateFIPS+countyFIPS
nass['county_code'] = nass['county_code'].map(int).apply(lambda x: '{0:0>3}'.format(x))
nass['state_fips_code'] = nass['state_fips_code'].apply(lambda x: '{0:0>2}'.format(x))
nass['fips'] = (nass['state_fips_code'].map(str)+nass['county_code'].map(str)).map(int)

nass['commodity_desc'] = nass['commodity_desc'].str.title()
nass = nass.rename(columns={"commodity_desc":"crop", "state_alpha":"state", "Value":"Nass_Area_acres"})
nass['Nass_Area_acres'] = nass['Nass_Area_acres'].str.replace(',', '').astype('float')
nass["Nass_Area_m2"] = nass["Nass_Area_acres"] * 4046.86
nass = nass[['fips', 'year', 'state', 'crop', 'Nass_Area_m2']]
nass.head()
cdl = pd.DataFrame()
for filename in sorted(os.listdir(cdl_dir)):
    if (filename.endswith('.csv')):
        temp = pd.read_csv(os.path.join(cdl_dir, filename)).drop(['.geo','system:index'],axis=1)
        cdl = pd.concat([temp, cdl], sort=True)
cdl = cdl[cdl['masterid'] != 0] # drop dummy feature
print(cdl.shape)
cdl.head()
# compute CDL coverage by county
classes = list(set(cdl.columns.tolist()) - set(['Year', 'area_m2', 'masterid', 'COUNTYFP', 'STATEFP']))
other = list(set(classes) - set(['1', '5']))
cdl['cdl_coverage'] = cdl[classes].sum(axis=1)
cdl['Other'] = cdl[other].sum(axis=1)
cdl = cdl.drop(other, axis=1)
cdl.head()
maxcoverage = cdl[cdl['Year'] == 2018][['masterid', 'cdl_coverage']].rename(
    {'cdl_coverage': '2018cdl_coverage'}, axis=1)
cdl = cdl.merge(maxcoverage, on='masterid')
cdl['CDL_perccov'] = cdl['cdl_coverage'] / cdl['2018cdl_coverage'] * 100
cdl_key = {'1': "Corn", '5': "Soybeans", 'Year': "year"}  
cdl = cdl.rename(cdl_key, axis=1)
cdl.head()
crops = ["Corn", "Soybeans", "Other"]
cdl = pd.melt(cdl, id_vars=['masterid', 'year', 'COUNTYFP', 'STATEFP', 'CDL_perccov'], value_vars=crops, value_name='CDL_Area_m2')
cdl = cdl.rename(columns={"variable": "crop", "masterid": "fips"})
cdl.head()
csdl = pd.DataFrame()
for filename in sorted(os.listdir(csdl_dir)):
    if (filename.endswith('.csv')):
        temp = pd.read_csv(os.path.join(csdl_dir, filename))
        csdl = pd.concat([temp, csdl], sort=True)
csdl = csdl.dropna(subset=['masterid'])
csdl[['COUNTYFP', 'STATEFP', 'masterid', 'year']] = csdl[['COUNTYFP', 'STATEFP', 'masterid', 'year']].astype(int)
csdl = csdl[['0', '1', '5', 'COUNTYFP', 'STATEFP', 'masterid', 'year']]
print(csdl.shape)
csdl.head()
csdl = csdl.merge(maxcoverage, on='masterid')
csdl['CSDL_coverage'] = csdl[['0','1','5']].sum(axis=1)
csdl['CSDL_perccov'] = csdl['CSDL_coverage'] / csdl['2018cdl_coverage']*100
# note that CDL covers lakes, rivers; CSDL does not
csdl_key = {'0':"Other",'1': "Corn", '5': "Soybeans"} 
csdl = csdl.rename(csdl_key,axis=1)
csdl.head()
crops = ["Corn", "Soybeans", "Other"]
csdl = pd.melt(csdl, id_vars=['masterid', 'year', 'COUNTYFP', 'STATEFP', 'CSDL_perccov'], value_vars=crops, value_name='CSDL_Area_m2')
csdl = csdl.rename(columns={"variable": "crop", "masterid": "fips"})
csdl.head()
df = nass.merge(cdl, on=['year', 'fips', 'crop'], how='left').merge(
    csdl, on=['year', 'fips', 'crop', 'COUNTYFP', 'STATEFP'], how='left')

# convert m2 to km2: 1e6
df['Nass_Area_km2'] = df['Nass_Area_m2'] / 1e6
df['CDL_Area_km2'] = df['CDL_Area_m2'] / 1e6
df['CSDL_Area_km2'] = df['CSDL_Area_m2'] / 1e6

print(df.shape)
df.head()
recent_years = np.arange(2008, 2019)
past_years = np.arange(1999, 2008)
df['year_group'] = '1999-2007'
df.loc[df['year'].isin(recent_years),'year_group'] = '2008-2018'

df_sub = df[df['CDL_perccov'] > 95]
print(df.shape)
print(df_sub.shape)
color_mapCSDL = { "Corn": 'orange',  "Soybeans": 'green'}
color_mapCDL = { "Corn": 'brown',  "Soybeans": 'darkseagreen'}
fsize = 15

lm = linear_model.LinearRegression()
bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.0)

fig, ax = plt.subplots(nrows=2, ncols=4,figsize=(16,8))

for i,(id,group) in enumerate(df_sub[df_sub['year_group']=='1999-2007'].groupby(['crop'])):
            
    ax[i,0].scatter(group['CDL_Area_km2'],group['Nass_Area_km2'], color = color_mapCDL[id], alpha=0.3, label=id)
    ax[i,1].scatter(group['CSDL_Area_km2'],group['Nass_Area_km2'], color = color_mapCSDL[id], alpha=0.3, label=id)

    group1= group.dropna(subset=['CDL_Area_km2','Nass_Area_km2'])
    R2 = r2_score(group1['CDL_Area_km2'].values,group1['Nass_Area_km2'].values)
    ax[i,0].text(0.05, 0.9, '$R^2$={0:.3f}'.format(R2), ha="left", va="center", size=fsize, bbox=bbox_props, transform=ax[i,0].transAxes)

    group2= group.dropna(subset=['CSDL_Area_km2','Nass_Area_km2'])
    R2 = r2_score(group2['CSDL_Area_km2'].values,group2['Nass_Area_km2'].values)
    ax[i,1].text(0.05, 0.9, '$R^2$={0:.3f}'.format(R2), ha="left", va="center", size=fsize, bbox=bbox_props, transform=ax[i,1].transAxes)

    lims = [0,2.6e+9/1e6] # km2
    # now plot both limits against each other
    ax[i,0].plot(lims, lims, 'k--', alpha=0.75, zorder=0)
    ax[i,1].plot(lims, lims, 'k--', alpha=0.75, zorder=0)
    ax[i,0].set_xlim(lims)
    ax[i,1].set_xlim(lims)
    ax[i,0].set_ylim(lims)
    ax[i,1].set_ylim(lims)  
    ax[i,0].set_aspect('equal')
    ax[i,1].set_aspect('equal')
    ax[0,0].set_xticklabels('', rotation=90)
    ax[0,1].set_xticklabels('', rotation=90)
    ax[i,1].set_yticklabels('', rotation=90)

for i,(id,group) in enumerate(df_sub[df_sub['year_group']=='2008-2018'].groupby(['crop'])):
       
    ax[i,2].scatter(group['CDL_Area_km2'],group['Nass_Area_km2'], color = color_mapCDL[id], alpha=0.3, label=id)
    ax[i,3].scatter(group['CSDL_Area_km2'],group['Nass_Area_km2'], color = color_mapCSDL[id], alpha=0.3, label=id)
    
    # set background to gray and alpha=0.2
    ax[i,2].set_facecolor((0.5019607843137255, 0.5019607843137255, 0.5019607843137255, 0.2))
    ax[i,3].set_facecolor((0.5019607843137255, 0.5019607843137255, 0.5019607843137255, 0.2))
    
    group1= group.dropna(subset=['CDL_Area_km2','Nass_Area_km2'])
    R2 = r2_score(group1['CDL_Area_km2'].values, group1['Nass_Area_km2'].values)
    ax[i,2].text(0.05, 0.9, '$R^2$={0:.3f}'.format(R2), ha="left", va="center", size=fsize, bbox=bbox_props, transform=ax[i,2].transAxes)

    group2= group.dropna(subset=['CSDL_Area_km2','Nass_Area_km2'])
    R2 = r2_score(group2['CSDL_Area_km2'].values, group2['Nass_Area_km2'].values)
    ax[i,3].text(0.05, 0.9, '$R^2$={0:.3f}'.format(R2), ha="left", va="center", size=fsize, bbox=bbox_props, transform=ax[i,3].transAxes)

    lims = [0,2.6e+9/1e6] # area in km2
    # now plot both limits against each other
    ax[i,2].plot(lims, lims, 'k--', alpha=0.75, zorder=0)
    ax[i,3].plot(lims, lims, 'k--', alpha=0.75, zorder=0)
    ax[i,2].set_xlim(lims)
    ax[i,3].set_xlim(lims)
    ax[i,2].set_ylim(lims)
    ax[i,3].set_ylim(lims)    
    ax[i,2].set_aspect('equal')
    ax[i,3].set_aspect('equal')
    ax[0,2].set_xticklabels('', rotation=90)
    ax[0,3].set_xticklabels('', rotation=90)
    ax[i,2].set_yticklabels('', rotation=90)
    ax[i,3].set_yticklabels('', rotation=90)

ax[0,0].set_ylabel('NASS county area [$km^2$]',fontsize=fsize)    
ax[1,0].set_ylabel('NASS county area [$km^2$]',fontsize=fsize)   
ax[1,0].set_xlabel('CDL county area [$km^2$]',fontsize=fsize)
ax[1,1].set_xlabel('CSDL county area [$km^2$]',fontsize=fsize)
ax[1,2].set_xlabel('CDL county area [$km^2$]',fontsize=fsize)
ax[1,3].set_xlabel('CSDL county area [$km^2$]',fontsize=fsize)
ax[0,0].set_title('1999-2007',fontsize=fsize) 
ax[0,1].set_title('1999-2007',fontsize=fsize) 
ax[0,2].set_title('2008-2018',fontsize=fsize) 
ax[0,3].set_title('2008-2018',fontsize=fsize) 

# Create the legend
legend_elements = [ Line2D([0], [0], color='k', linewidth=1, linestyle='--', label='1:1'),
                    Line2D([0], [0], marker='o', color='w', markerfacecolor='brown', markersize=15, label='Corn CDL'),
                    Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', markersize=15, label='Corn CSDL'),
                    Line2D([0], [0], marker='o', color='w', markerfacecolor='darkseagreen', markersize=15, label='Soybeans CDL'),
                    Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=15, label='Soybeans CSDL'),
                    mpatches.Patch(facecolor='gray', edgecolor='gray', alpha = 0.2, label='NASS informed by CDL')]
fig.legend(handles=legend_elements,
           loc='lower center',  
           bbox_to_anchor=(0.5,-0.01),
           fontsize = 'x-large',
           ncol=7)

fig.tight_layout(rect=[0,0.05,1,1]) # legend on the bottom
def corrfun(df, col1, col2):
    
    df = df.dropna(subset=[col2])
    
    if df.shape[0] != 0:
        r2 = r2_score(df[col1].values, df[col2].values)
        mse = mean_squared_error(df[col1].values, df[col2].values)
    else:
        r2 = np.nan
        mse = np.nan
        
    return pd.Series({'R': r2, 'mse': mse, 'Ncounties': df.shape[0]}) # return R2 (coef of determination)
totArea = df.groupby(['year', 'state', 'crop'])['Nass_Area_m2', 'CDL_Area_m2', 'CSDL_Area_m2'].sum().reset_index()
print(totArea.shape) # 20years * 13states * 2commodity = 520 rows
corr_cdl = df[df['CDL_perccov'] > 90].groupby(['state','year','crop']).apply(corrfun,'Nass_Area_m2','CDL_Area_m2').reset_index().rename(
    {'R': 'R_NASS_CDL', 'mse': 'mse_NASS_CDL', 'Ncounties': 'Ncounties_CDL'}, axis=1)
corr_csdl = df.groupby(['state','year','crop']).apply(corrfun, 'Nass_Area_m2', 'CSDL_Area_m2').reset_index().rename(
    {'R': 'R_NASS_CSDL', 'mse': 'mse_NASS_CSDL', 'Ncounties': 'Ncounties_CSDL'}, axis=1)
corr = corr_cdl.merge(corr_csdl, on=['state','year','crop'], how='outer')
corr = corr.merge(totArea, on=['state','year','crop'], how='left')
print(corr.shape)

abbr_to_state = {'IL':'Illinois', 'IA':'Iowa', 'IN':'Indiana', 'NE':'Nebraska', 'ND':'North Dakota',
                 'SD':'South Dakota', 'MN':'Minnesota', 'WI':'Wisconsin', 'MI':'Michigan',
                 'KS':'Kansas','KY':'Kentucky', 'OH':'Ohio', 'MO':'Missouri'}
corr['state_abbrs'] = corr['state']
corr['state_name'] = corr['state'].replace(abbr_to_state)
corr.head()
state_geosorted = ['North Dakota', 'Minnesota', 'Wisconsin', 'Michigan',
 'South Dakota','Iowa', 'Illinois', 'Indiana',
 'Nebraska', 'Missouri', 'Kentucky', 'Ohio',
 'Kansas']

color_mapCSDL = { "Corn": 'orange',  "Soybeans": 'green'}
color_mapCDL = { "Corn": 'brown',  "Soybeans": 'darkseagreen'}
marker_map ={ "Corn": 'o',  "Soybeans": '^'}
fsize = 15
bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.0)

fig, ax = plt.subplots(nrows=4, ncols=4,figsize=(16,12))

for i, id in enumerate(state_geosorted):
    
    group = corr[corr['state_name']==id]

    thisAx = ax[int(np.floor(i/4)), i%4]
    thisAx.text(0.05, 0.1, id, ha="left", va="center", size=18, bbox=bbox_props, transform=thisAx.transAxes)
    
    ylabels = np.arange(0, 12, 2)/10
    thisAx.set_yticks(ylabels)
    thisAx.set_yticklabels('', rotation=0)
    
    xlabels = np.arange(1999, 2019, 2)
    thisAx.set_xticks(xlabels)
    thisAx.set_xticklabels('', rotation=90)
    thisAx.fill_between(np.arange(2008, 2019), 0, 1.1, alpha = 0.2, color='gray')

    crops = ['Soybeans','Corn']
    for l,id2 in enumerate(crops):
        group2 = group[group['crop']==id2]
        group2 = group2.sort_values('year')
        
        thisAx.plot(group2['year'],group2['R_NASS_CDL'],color=color_mapCDL[id2], alpha=1, marker=marker_map[id2])
        thisAx.plot(group2['year'],group2['R_NASS_CSDL'],color=color_mapCSDL[id2], alpha=1, marker=marker_map[id2])

    thisAx.set_xlim([1999,2018])
    thisAx.set_ylim([0.0, 1.1])
    thisAx.grid(True)

ax[0,0].set_ylabel('$R^{2}$',fontsize=fsize)
ax[1,0].set_ylabel('$R^{2}$',fontsize=fsize)
ax[2,0].set_ylabel('$R^{2}$',fontsize=fsize)
ax[3,0].set_ylabel('$R^{2}$',fontsize=fsize)
ax[0,0].set_yticklabels(ylabels, rotation=0,fontsize=fsize)
ax[1,0].set_yticklabels(ylabels, rotation=0,fontsize=fsize)
ax[2,0].set_yticklabels(ylabels, rotation=0,fontsize=fsize)
ax[3,0].set_yticklabels(ylabels, rotation=0,fontsize=fsize)
ax[3,0].set_xticklabels(xlabels, rotation=90,fontsize=fsize)
ax[-1,-1].axis('off')
ax[-1,-2].axis('off')
ax[-1,-3].axis('off')

# Create the legend manually
legend_elements = [ Line2D([0], [0], color='brown',  linewidth=3, marker='o', linestyle='-', label='Corn CDL'),
                    Line2D([0], [0], color='orange',  linewidth=3, marker='o', linestyle='-', label='Corn CSDL'),
                    Line2D([0], [0], color='darkseagreen',  linewidth=3, marker='^', linestyle='-', label='Soybeans CDL'),
                    Line2D([0], [0], color='green',  linewidth=3, marker='^', linestyle='-', label='Soybeans CSDL')]
fig.legend(handles=legend_elements,
           loc="lower right",   # Position of legend
           bbox_to_anchor=(0.6, 0.05),
           fontsize = 'xx-large')

bckground_patch = mpatches.Patch(color='gray', alpha = 0.2, label='Training years')
fig.legend(handles=[bckground_patch],
           loc="lower right",
           bbox_to_anchor=(0.8, 0.1),
           fontsize = 'xx-large')

fig.tight_layout()
ax[2,1].set_xticklabels(xlabels, rotation=90,fontsize=fsize)
ax[2,2].set_xticklabels(xlabels, rotation=90,fontsize=fsize)
ax[2,3].set_xticklabels(xlabels, rotation=90,fontsize=fsize)
fig.show()
price_soybean = 2.53
price_corn = 1.00  # Assuming a base price for corn, can be any value
corn_acres = pd.read_csv(corn_acres_url)
soybean_acres = pd.read_csv(soybean_acres_url)

soybean_acres['Acres'] = soybean_acres["SOYBEANS - ACRES PLANTED  -  <b>VALUE</b>"]
corn_acres['Acres'] = corn_acres["CORN - ACRES PLANTED  -  <b>VALUE</b>"]
acres_historical = [corn_acres['Acres'],soybean_acres['Acres']]

acres_ratio = corn_acres['Acres'] / soybean_acres['Acres']
# Calculate the ratio
ratio = price_soybean / price_corn

# Print the ratio with the message
print(f"The ratio between soybean and corn is {ratio:.2f}")
