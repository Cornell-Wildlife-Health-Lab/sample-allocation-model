'''
Script Name: Input Processing for Sample Allocation using Optimal Control Model 
Author: Sarah Sernaker, ss3376@cornell.edu, Cornell University
Description: Uses sample data from the warehouse, cost model output, and hazard
 model output to creates 2 flat files that will be fed into the actual Sample 
 Allocation optimization model. One flat file will have one row
 for every subadministrative unit in the model, with model input parameters. The other
 flat file also contains parameters, in a more succinct format that is compatible with the existing code.
 
Inputs: 
  params.json
  provider.json
  sample.ndjson
  risk_model.json
 
Outputs: 
  sam_input.csv
  mode.csv
  
  info.html
  execution_log.log 
  attachments.json
'''



###############################
# Environment

import os
import pathlib

import json
import logging

import sys
import datetime

import pandas as pd

import ndjson
import itertools

###############################
# SCRIPT VARIABLES

base_path = pathlib.Path("/data")

  
## files used
provider_file_path = base_path / "provider.json"
sample_file_path = base_path / "sample.ndjson"
parameters_file_path  = base_path / "params.json"
hazards_path  = base_path / "risk_model.json"

# files to create
model_metadata_log_file = base_path / "attachments" / "info.html"
logging_path = base_path / "attachments" / "execution_log.log"
attachments_json_path = base_path / "attachments.json"

output_path  = base_path / "SAM_input.csv"
mode_path  = base_path / "mode.csv"


###############################
# FUNCTIONS

def model_log(line='', filename = model_metadata_log_file):
  """Writes a single line to the model_metadata_log text file.
  Args:
    line: The line to be written.
    filename: The name of the file.
  """
  with open(filename, 'a') as f:
    f.write(line + '<br>' + '\n')

def model_log_html(line='', html_element="p", filename=model_metadata_log_file):
    """
    Writes a single line to the model_metadata_log text file with specified HTML element.

    Args:
        line: The line to be written.
        filename: The name of the file.
        html_element: The HTML element tag to use (e.g., "h1", "h2", "p", "div").
    """
    with open(filename, 'a') as f:
        f.write(f"<{html_element}>{line}</{html_element}>" + '\n') 

def dict_to_html_list(data, list_type='unordered'):
  """
  Converts a Python dictionary to an HTML string representing a list.

  Args:
    data: The input dictionary.
    list_type: 'unordered' (default) or 'ordered' to specify the list type.

  Returns:
    An HTML string representing the dictionary.
  """

  def _dict_to_html_helper(data):
    """Recursive helper function to handle nested dictionaries."""
    html_str = ""
    if list_type == 'unordered':
      html_str += "<ul>"
    elif list_type == 'ordered':
      html_str += "<ol>"
    else:
      raise ValueError("Invalid list_type. Use 'unordered' or 'ordered'.")

    for key, value in data.items():
      html_str += f"<li>{key}: "
      if isinstance(value, dict):
        html_str += _dict_to_html_helper(value)
      elif isinstance(value, list):
        html_str += "<ul>"
        for item in value:
          html_str += f"<li>{item}</li>"
        html_str += "</ul>"
      else:
        html_str += f"{value}"
      html_str += "</li>"

    if list_type == 'unordered':
      html_str += "</ul>"
    elif list_type == 'ordered':
      html_str += "</ol>"

    return html_str

  return _dict_to_html_helper(data)

def add_item_to_json_file_list(file_path, new_item):
  """
  Adds a new item to the list within a JSON file.

  Args:
    file_path: Path to the JSON file.
    new_item: The item to be added to the list.

  Raises:
    FileNotFoundError: If the specified file does not exist.
    json.JSONDecodeError: If the file content is not valid JSON.
  """

  try:
    with open(file_path, 'r') as f:
      data = json.load(f)

    if isinstance(data, list):
      data.append(new_item)
    else:
      raise ValueError("The JSON file does not contain a list.")

    with open(file_path, 'w') as f:
      json.dump(data, f, indent=2) 

  except FileNotFoundError:
    print(f"Error: File '{file_path}' not found.")
    raise
  except json.JSONDecodeError:
    print(f"Error: Invalid JSON in '{file_path}'.")
    raise
  except ValueError as e:
    print(f"Error: {e}")
    raise



###############################
# SETUP FILE STRUCTURE

# Create the directory if it doesn't already exist)
model_metadata_log_file.parent.mkdir(parents=True, exist_ok=True)
# Create a blank HTML file
model_metadata_log_file.touch(exist_ok=True)

# Create attachments.json file which will contain a list of all attachments generated
# Initially, the attachments is simply an empty list
with open(attachments_json_path, 'w', newline='') as f:
  writer = json.dump(list(), f)


# Append execution log to attachments.json for developer feedback
attachment = {
  "filename": "execution_log.log", 
  "content_type": "text/plain", 
  "role": "downloadable"
  }
add_item_to_json_file_list(attachments_json_path, attachment)

# append info log to the attachments.json for user feedback
attachment = {
  "filename": "info.html", 
  "content_type": "text/html", 
  "role": "feedback"}
add_item_to_json_file_list(attachments_json_path, attachment)



###############################
# SETUP LOGGING

# Create the directory if it doesn't already exist
logging_path.parent.mkdir(parents=True, exist_ok=True)
# Create a blank HTML file
logging_path.touch(exist_ok=True)

logging.basicConfig(level = logging.DEBUG, # Alternatively, could use DEBUG, INFO, WARNING, ERROR, CRITICAL
                    filename = logging_path, 
                    filemode = 'w', # a is append, w is overbite
                    datefmt = '%Y-%m-%d %H:%M:%S',
                    format = '%(asctime)s - %(levelname)s - %(message)s')

# Uncaught exception handler
def handle_uncaught_exception(type, value, traceback):
  logging.error(f"{type} error has occurred with value: {value}. Traceback: {traceback}")
sys.excepthook = handle_uncaught_exception



## Initiate model metadata log

# Clear model log file contents if necessary.
open(pathlib.Path(model_metadata_log_file), 'w').close()
model_log_html("Model Execution Summary", "h2")
model_log_html("Model: Sample Allocation Model","h3")
model_log_html('Date: ' + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + ' GMT', "p")
logging.info("Model: Sample Allocation Model")
logging.info('Date: ' + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + ' GMT')
logging.info("This log records data for debugging purposes in the case of a model execution error.")




###############################
# PROCESS PARAMETERS      
#  ASSUMES COSTS INFORMATION IS INCLUDED IN PARAMS
#  ASSUMES MAX HARVEST CAPACITY SPECIFED HERE AS WELL


# Get model parameters file
try:
  with open(pathlib.Path(parameters_file_path), 'r') as f:
    params = json.load(f)
    logging.info("Parameter file loaded successfully")
except:
  # The model cannot be executed without a params file. Exit with an error immediately.
  logging.error("params.json File does not exist.")
  model_log_html("ERROR", "h4")
  model_log_html("Parameters (params.json) file not found.")
  sys.exit(1)



# If budget is not specified with model 2 & 3, throw error
model_mode = params['mode']
if 'annual budget' not in params:
  if model_mode in (2,3) :
     # The model cannot be executed without a specified budget.
      logging.error("User-defined budget is missing, but it is required for this model.")
      model_log_html("ERROR", "h4")
      model_log_html("User-defined budget is missing, but it is required for this model.")
      sys.exit(1)
  else:
    annual_budget = None
else :
  annual_budget = params['annual budget']      
      
# if no look back period then set to 0
if 'years_lookback' not in params:
  yrs_lookback = 0
else :
  yrs_lookback = params['years_lookback']
  

# if no growth rate specified, default to 0.2
if 'annual_growth_rate' not in params:
  annual_growth_rate = 0.2
else :
  annual_growth_rate = params['annual_growth_rate']
  
  
  
# extract the variable "consider": whether to include sub admin area or not
params_input = pd.DataFrame(params['consider'].items(),
             columns = ['sub_administrative_area','consider'])


if model_mode in (2,3) :
  
  ## get per sample costs
  if 'expense' not in params:
      #  This model cannot be executed without costs per sample
      logging.error("Costs per sample per sub administrative unit missing.")
      model_log_html("ERROR", "h4")
      model_log_html("Costs per sample per sub administrative unit missing.")
      sys.exit(1) 
  
  # extract the costs input into model 
  costs = pd.DataFrame(params['expense'].items(),
              columns = ['sub_administrative_area','cost'])

  # merge parameter and costs
  params_input = pd.merge(params_input,costs,
                              on = ['sub_administrative_area'],
                              how = 'outer')
  
  # if any costs are 0 break
  costs0 = ((params_input['cost'] == 0) & (params_input['consider'] == 'include')).sum()
  if costs0 > 0 :
    logging.error("There are included subadministrative units without costs.")
    model_log_html("ERROR", "h4")
    model_log_html("There are included subadministrative units without costs. Restart model without unit or add expense costs.")
    sys.exit(1)
  
  
  
  
  ## get max number of samples for surveillance if input (if missing, will be filled in as 1000)
  if 'maxsample'  in params:
      # extract the max sample capacity input into model
      maxsample = pd.DataFrame(params['maxsample'].items(),
                  columns = ['sub_administrative_area','maxsample'])
      
      # merge parameter and maxsample if it exists
      params_input = pd.merge(params_input,maxsample,
                              on = ['sub_administrative_area'],
                              how = 'outer')

     

      maxsamp0 = ((params_input['maxsample'] == 0) & (params_input['consider'] == 'include')).sum()
      # if any max sample are 0 break
      if maxsamp0 > 0 :
          logging.error("There are subadministrative units with no specified maximum samples for surveillance.")
          model_log_html("ERROR", "h4")
          model_log_html("There are subadministrative units with no specified maximum samples for surveillance.")
          model_log_html("Restart model without unit or specify more than 0 maximum samples.")
          sys.exit(1)
  else :
      params_input['maxsample'] = None

# any missing maxsample should be set to 1,000
params_input.loc[(params_input['consider'] == 'include') & 
                 (params_input['maxsample'].isna()),
                      'maxsample'] = 1000


###############################
# PROCESS PROVIDER DATA
#   Provider data has regions associated with sub-admin areas and 
#   is needed for distributing expenses 

## load data
try: 
    with open(provider_file_path, 'r') as f:
        provider = json.load(f)
        logging.info("Provider file loaded successfully")
except FileNotFoundError:
    # The model cannot be executed without a provider file. Exit with an error immediately.
    model_log_html("ERROR", "h4")
    logging.error("provider.json File does not exist.")
    sys.exit(1)


# list of all sub admin units and make data frame 
subadmins = []
for area in provider['sub_administrative_area']['area_infos']:
    subadmins.append(area['_sub_administrative_area'])
subadmins = pd.DataFrame(subadmins, columns = ['sub_administrative_area'])



# merge subadmin areas and parameters
subadmin_params = pd.merge(subadmins,
                           params_input,
                             on = ['sub_administrative_area'],
                             how = 'outer')


if model_mode in (2,3) :
  # check that all included sub admins that are "included" have costs 
  missing_cost = subadmin_params[
      (subadmin_params['consider'] == 'include') & (subadmin_params['cost'].isna())
      ]
  # if unit cost is missing then STOP
  if not missing_cost.empty :
    # Every 
      model_log_html("ERROR", "h4")
      logging.error("There are subadministrative units taht have no per unit costs.")
      sys.exit(1)


  # check whether included subadmin units have missing max sample
  missing_maxsamp = subadmin_params[
      (subadmin_params['consider'] == 'include') & (subadmin_params['maxsample'].isna())
    ]
  # if any subadmin units are missing max sampling then set to 1000
  if not missing_maxsamp.empty:
    model_log_html("There are " + str(len(missing_maxsamp)) + " selected subadministrative units that have no user-specified max sampling capacity.")
    model_log_html("A value must be defined therefore these will be set to 1,000 in order to run the model.")

    





###############################
# SAMPLE DATA
# If a person specifies 1 or more years look back then need to get sample
# data to get count by year - otherwise do not need sample data


 # get the specified target year (take first part of season year, e.g. 2024-25 -> 2024)
target_yr = pd.to_numeric(params['season_year'][0:4])


if (yrs_lookback > 0) :
    try: 
        with open(sample_file_path, 'r') as f:
            samples = ndjson.load(f)
            logging.info("Sample file loaded successfully")
    except FileNotFoundError:
        logging.error("samples.ndjson file does not exist.")
        model_log_html("ERROR", "h4")
        model_log_html("ERROR: Samples (sample.ndjson) file not found. Sample data are required to run this model. Execution halted.")
        sys.exit(1)

    
    # list of samples sub admin units
    sample_seasonyr = []
    for s in samples:
        season_year = s.get('season_year', 'NA')
        sample_id = s.get('sample_id', 'NA')
    
        # Handle nested dictionary
        subadmin_area = s.get('_sub_administrative_area', {})
        subadmin_id = subadmin_area.get('_id', 'NA') if isinstance(subadmin_area, dict) else 'NA'
    
        if season_year != 'NA' and sample_id != 'NA' and subadmin_id != 'NA':
            sample_seasonyr.append([season_year, subadmin_id, sample_id])
            
   
        
      
    samples_by_subadmin_yr = pd.DataFrame(sample_seasonyr, 
                                          columns = ['season_year','sub_administrative_area','sample_id'])
    
    
    # count number of samples by season year and sub admin area
    samples_by_subadmin_yr_long  = samples_by_subadmin_yr.groupby([
                                        'season_year',
                                        'sub_administrative_area',
                                    ], as_index=False, dropna = False).agg({
                                        'sample_id': 'count'
                                    })
    
    
                             
    # variable for how many years prior the data come from
    samples_by_subadmin_yr_long['historical_yr'] = target_yr - pd.to_numeric(samples_by_subadmin_yr_long['season_year'].str[:4])


    # generate combination of all sub admin areas and number of years look back
    combinations = list(itertools.product(subadmins['sub_administrative_area'], 
                                           range(1,yrs_lookback+1)))
    combinations_df = pd.DataFrame(combinations, 
                                    columns=['sub_administrative_area', 
                                             'historical_yr'])
    combinations_df['season'] = target_yr-combinations_df['historical_yr']
    combinations_df['season'] = combinations_df['season'].astype(str) + "-" + (combinations_df['season']+1).astype(str).str[-2:]
    
    # fill all specified historical years, fill in 0's where no samples
    all_hist = pd.merge(combinations_df,samples_by_subadmin_yr_long,
                         how = 'left', 
                         on = ['sub_administrative_area',
                               'historical_yr'])
    
               
    #  capture which season-years available and which ones not          
    yrs_using0 = all_hist[['season_year','historical_yr','season']].drop_duplicates().sort_values(by = 'historical_yr')
    yrs_using = yrs_using0['season_year'][0:yrs_lookback]
    
    
    # pivot table wide by historical year
    samples_by_subadmin_yr_wide = all_hist.pivot(index = 'sub_administrative_area',
                                      columns = 'historical_yr',
                                      values = 'sample_id').reset_index()
    # rename variables
    samples_by_subadmin_yr_wide.columns = ["sub_administrative_area"] + [f"Historical sampling - year{i}" for i in range(1, yrs_lookback+1)]
    
    # fill any missing with 0's
    samples_by_subadmin_yr_wide.fillna(0,inplace= True)
    
    
    
    # logging information
    model_log_html("Years lookback (historical samples)", "h4")

    # message if user requests more years back than are available
    if yrs_using.isna().any() :
        model_log_html("User requested " + str(yrs_lookback) + " years lookback, but only " + str(sum(yrs_using.notna())) + " were available.","i")
        model_log_html("Using sample data from season-years " + ', and '.join(yrs_using.dropna()) + " for the Sample Allocation Model.")
        model_log_html("For year(s) with no sample data (" + 
                       ', and '.join(yrs_using0['season'][yrs_using0['season_year'].isna()]) + 
                       "), the number of samples will be set to 0 in each of the subadministrative units.")
        model_log_html("*It is recommended that user's re-run the model to exclude years that have no sampling data.*")
    else :
        model_log_html("User requested " + str(yrs_lookback) + " years lookback.")
        model_log_html("Using season-years: " + ', '.join(yrs_using) + " for the Sample Allocation Model.")


    # logging information
    model_log_html("Number of samples by year", "h4")

    # number of samples by season year
    numsamps_byyr = all_hist.groupby(['season'], as_index=False, dropna = False).agg({
                            'sample_id': 'sum'
                        }).sort_values(by='season', ascending=False)
    
    for index, row in numsamps_byyr.iterrows():
        model_log_html("There are " + str(format(int(row['sample_id']),",")) + " samples from " + str(row['season']) +".")

    


    # logging information
    model_log_html("Number of subaministrative units with no samples", "h4")

    #  number of subadmins missing samples by season year
    nummissing_byyr = all_hist[all_hist['sample_id'].isna()].groupby([
                                        'season'
                                    ], as_index=False, dropna = False).agg({
                                        'sub_administrative_area': 'count'
                                    }).sort_values(by='season', ascending=False)
     
    if not nummissing_byyr.empty:
        model_log_html("Out of " + str(len(subadmins)) + " total subadministrative units for the provider:")
        for index, row in nummissing_byyr.iterrows():
             model_log_html(str(row['sub_administrative_area']) +
                            " had no samples in " +  str(row['season']),"ul"
                            )
        model_log_html("The number of samples will be set to 0 in these areas.")
        
        
      
    # merge samples over time with the parameters and subadmin units
    subadmin_params_samps = pd.merge(subadmin_params,
                                    samples_by_subadmin_yr_wide,
                                on = ['sub_administrative_area'],
                                how = 'outer')

     
else :
  model_log_html("User specified 0 years look back. No historical data will be used.")
  subadmin_params_samps = subadmin_params
             




###############################
# HAZARDS OUTPUT
# Use hazard model output to calculate introduction rate
#   1. From output of the hazard model, get proportion of the total 
#      quota allocated to each subadmin unit. 
#   2. multiply each of those proportions by the probability that disease will
#      appear somewhere in the state = .66**
#     TODO: REF PAPER WHERE THIS COMES FROM
#     **.66 was valid up to early 2025, but if any more states flip positive then we will have to update the .66 number.  


# load hazards model output
try: 
    with open(hazards_path, 'r') as f:
        hazards = json.load(f)
        logging.info("Hazard model output file loaded successfully")
except FileNotFoundError:
    # The model cannot be executed without a params file. Exit with an error immediately.
    model_log_html("ERROR", "h4")
    logging.error("risk_model.json File does not exist.")
    sys.exit(1)


# get quotas by sub admin area
subadmin_quotas = pd.DataFrame(hazards['output'])[['subadminID', 'subadminName', 'totalQuota']]

# proportion of quota allocated to each sub admin area
subadmin_quotas['total_quota'] = subadmin_quotas['totalQuota'].sum()
subadmin_quotas['prop_quota'] = subadmin_quotas['totalQuota']/subadmin_quotas['total_quota']
# multiple 0.66 for introduction rate
subadmin_quotas['introduction rate'] = 0.66 * subadmin_quotas['prop_quota']

# rename column for linking
subadmin_quotas.rename(columns = {'subadminID' : 'sub_administrative_area'}, inplace = True)



# merge in introduction rate
subadmin_params_samps_probs = pd.merge(subadmin_params_samps, 
                                      subadmin_quotas[['sub_administrative_area','introduction rate']],
                             on = ['sub_administrative_area'],
                             how = 'outer')






############################### 
# FINAL OUTPUT


# rename columns for consistency with model code
rename_map = {'sub_administrative_area':'_id',
              'consider':'Consider',
              'cost' :'unit_cost',
              'maxsample':'MaxSample'}
subadmin_params_samps_probs = subadmin_params_samps_probs.rename(columns={k: v for k, v in rename_map.items() if k in subadmin_params_samps_probs.columns})

# capture historical columns and unit cost columns if they exist
historical_cols = [col for col in subadmin_params_samps_probs.columns if col.startswith("Historical")]
maxsample_cols = [col for col in subadmin_params_samps_probs.columns if col == "MaxSample"]
unitcost_cols = [col for col in subadmin_params_samps_probs.columns if col == "unit_cost"]

final_dat = subadmin_params_samps_probs[['_id',
                                        'Consider'] +    
                                        maxsample_cols +
                                        historical_cols +
                                        ['introduction rate'] +
                                        unitcost_cols
                                        ]
final_dat.to_csv(output_path, index = False)


# create mode.csv as input for optimization models
mode_file = [params['_model'],
             params['mode'],
             annual_budget,
             params['season_year'],
             annual_growth_rate]
mode_columns = ['_model','mode_selection','budget','stat_year','annual_growth_rate']
mode_df = pd.DataFrame([mode_file],columns = mode_columns)
mode_df.to_csv(mode_path, index = False)



# append info log to the attachments.json for user feedback
attachment = {
  "filename": "sample.csv", 
  "content_type": "text/csv", 
  "role": "optimization_model_data_input"}
add_item_to_json_file_list(attachments_json_path, attachment)

# append info log to the attachments.json for user feedback
attachment = {
  "filename": "mode.csv", 
  "content_type": "text/csv", 
  "role": "optimization_model_parameter_input"}
add_item_to_json_file_list(attachments_json_path, attachment)



model_log_html("Data processing steps completed and ready for optimization steps.","h3")
logging.info("Data processing steps completed and moving to optimization steps")


model_log_html("Running model Mode " + str(params['mode']) + " for season-year " + str(params['season_year']) + ", using an annual growth rate of " + str(annual_growth_rate))
if model_mode in (2,3) :
  model_log_html("Using annual budget of $" + str(format(int(params['annual budget']),",")) )
