'''
Script Name: Input Processing for Sample Allocation using Optimal Control Model 
Author: Sarah Sernaker, ss3376@cornell.edu, Cornell University
Description: Uses sample data from the warehouse, cost model output, and hazard
 model output to creates 2 flat files that will be fed into the actual Sample 
 Allocation optimization model. One flat file will have one row
 for every subadministrative unit in the model, with model input parameters. The other
 flat file also contains parameters, in a more succinct format that is compatible with the existing code.
 It will also output a csv file of historical sampling, if used and available, this is to include
 in the organization of the final results collated in post_processing.py.
 
Inputs: 
  params.json
  sub_administrative_area.json
  sample.ndJson
  risk_model.json
 
Outputs: 
  sam_input.csv
  mode.csv
  hist_sample.csv (if specified in model)
  
  info.html
  execution_log.log 
  attachments.json
'''



###############################
# Environment

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
provider_file_path = base_path / "sub_administrative_area.ndJson"
sample_file_path = base_path / "sample.ndJson"
parameters_file_path  = base_path / "params.json"
hazards_path  = base_path / "risk_model.json"

# files to create
model_metadata_log_file = base_path / "attachments" / "info.html"
logging_path = base_path / "attachments" / "execution_log.log"
attachments_json_path = base_path / "attachments.json"

# file that will be output
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
# PROCESS PROVIDER SUBADMINISTRATIVE DATA
#   Need full list of subadministrative units

## load data
try: 
    with open(provider_file_path, 'r') as f:
        provider = ndjson.load(f)
        logging.info("Provider file loaded successfully")
except FileNotFoundError:
    # The model cannot be executed without a provider file. Exit with an error immediately.
    model_log_html("ERROR", "h4")
    logging.error("provider.json File does not exist.")
    sys.exit(1)


# list of all sub admin units and make data frame 
subadmins = []
for area in provider:
    admin_id = area.get('_id', 'NA')
    admin_name = area.get('full_name',admin_id)
    subadmins.append([admin_id,admin_name])
subadmins = pd.DataFrame(subadmins, columns = ['sub_administrative_area','sub_administrative_area_name'])





###############################
# PROCESS PARAMETERS      
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
  
# get model mode
model_mode = params['mode']

# If budget is not specified with model 2 & 3, throw error
if 'annual_budget' not in params:
  if model_mode in (2,3) :
     # The model cannot be executed without a specified budget.
      logging.error("User-defined budget is missing, but it is required for this model.")
      model_log_html("ERROR", "h4")
      model_log_html("User-defined budget is missing, but it is required for this model.")
      sys.exit(70)
  else:
    annual_budget = None
else :
  annual_budget = params['annual_budget']      
      
  
    
# if no growth rate specified, default to 0.2
if 'annual_growth_rate' not in params:
  annual_growth_rate = 0.2
else :
  annual_growth_rate = params['annual_growth_rate']
  
  
# turn subadmin parameter input into data frame
params_input = pd.DataFrame.from_dict(params['sub_administrative_area_input'], orient='index').reset_index()
params_input.rename(columns = {"index":"sub_administrative_area",
                               "expense":"cost"},inplace=True)


# need at least 2 sub admin units or break
if (params_input['include']).sum() < 2 :
    # The model cannot be executed with less than 2 sub admin units
      logging.error("Model requires 2 or more sub-administrative units,there was only 1 included.")
      model_log_html("ERROR", "h4")
      model_log_html("Model requires 2 or more sub-administrative units,there was only 1 included.")
      sys.exit(70)


# merge subadmin areas and parameters
subadmin_params = pd.merge(subadmins,
                           params_input,
                             on = ['sub_administrative_area'],
                             how = 'outer')

 # get array of sub-administrative units that were included
list_included_subadmins = subadmin_params[subadmin_params['include'] == True]['sub_administrative_area'].to_numpy()


# if for some reason there's a sub admin missing from model, compared to 
# master sub admin list, throw error
if subadmin_params['include'].isna().any():
    # The model cannot be executed without a specified budget.
     logging.error("There are some subadministrative areas not captured by the model for some reason.")
     model_log_html("ERROR", "h4")
     model_log_html("There are some subadministrative areas not captured by the model for some reason.")
     model_log_html("Review model input or contact Warehouse assistance.")
     sys.exit(1)
     
     
     
# logging information
model_log_html("Subadministrative units", "h4")
model_log_html("Out of " + str(len(subadmins)) + " total sub-administrative units in this agency:")
model_log_html(str((subadmin_params['include'] == True).sum()) + " have been included in the model.","ul")
model_log_html(str((subadmin_params['include'] == False).sum()) + " have been excluded from the model.","ul")


if model_mode in (2,3) :

    ## get per sample costs
   if 'cost' not in params_input.columns:
     #  This model cannot be executed without costs per sample
     logging.error("There are no costs provided for sub-administrative areas.")
     model_log_html("ERROR", "h4")
     model_log_html("There are no costs provided for sub-administrative areas.")
     sys.exit(70) 
  
  
   # if any costs are 0 break
   costs0_bool = (subadmin_params['cost'] == 0) & (subadmin_params['include'] == True)
   costs0 = (costs0_bool).sum()
   if costs0 > 0 :
     logging.error("There are included subadministrative units with $0 specified in costs.")
     logging.error("The following included sub-administrative units have $0 specified in costs:")
     logging.error(', '.join(sorted(subadmin_params['sub_administrative_area_name'][costs0_bool].astype(str))),"ul")
     
     model_log_html("ERROR", "h4")
     model_log_html("The following included sub-administrative units have $0 specified in costs:")
     model_log_html(', '.join(sorted(subadmin_params['sub_administrative_area_name'][costs0_bool].astype(str))),"ul")

     model_log_html("Restart model without unit or add expense costs.")

     sys.exit(70)
  
  
  
   ## get max number of samples for surveillance if input (if missing, will be filled in as 1000)
   if 'maxsample' in params_input.columns:
      
       # if someone specifies 0 as the maxsample then throw error, should just exclude from model
       maxsamp0_bool = (subadmin_params['maxsample'] == 0) & (subadmin_params['include'] == True)
       maxsamp0 = (maxsamp0_bool).sum()
       # if any max sample are 0, break
       if maxsamp0 > 0 :
           logging.error("The following subadministrative units have specified maximum of 0 samples for surveillance:")
           logging.error(', '.join(sorted(subadmin_params['sub_administrative_area_name'][maxsamp0_bool].astype(str))),"ul")
           
           model_log_html("ERROR", "h4")
           model_log_html("The following subadministrative units have specified maximum of 0 samples for surveillance:")
           model_log_html(', '.join(sorted(subadmin_params['sub_administrative_area_name'][maxsamp0_bool].astype(str))),"ul")

           model_log_html("Restart model without unit(s) or specify more than 0 maximum samples.")
           sys.exit(70)
   else :
       subadmin_params['maxsample'] = None
    
     

   # check that all included sub admins that are "included" have costs 
   missing_cost_bool = (subadmin_params['include'] == True) & (subadmin_params['cost'].isna())
   missing_cost = missing_cost_bool.sum()
   # if unit cost is missing then STOP
   if missing_cost > 0 :
      model_log_html("The following subadministrative unit(s) have no per unit costs:")
      model_log_html(', '.join(sorted(subadmin_params['sub_administrative_area_name'][missing_cost_bool].astype(str))),"ul")

      model_log_html("ERROR", "h4")
      model_log_html("The following subadministrative unit(s) have no per unit costs:")
      model_log_html(', '.join(sorted(subadmin_params['sub_administrative_area_name'][missing_cost_bool].astype(str))),"ul")

      model_log_html("Provide costs or exclude subadministrative unit from model.")
      sys.exit(70)


   # check whether included subadmin units have missing max sample
   missing_maxsamp_bool = (subadmin_params['include'] == True) & (subadmin_params['maxsample'].isna())
   missing_maxsamp = missing_maxsamp_bool.sum()
   # if any subadmin units are missing max sampling then set to 1000
   if missing_maxsamp > 0:
    model_log_html("Max sampling capacity parameter","h4")
    model_log_html("There are " + str(missing_maxsamp) + " selected subadministrative units that have no user-specified max sampling capacity.")
    model_log_html(', '.join(sorted(subadmin_params['sub_administrative_area_name'][missing_maxsamp_bool].astype(str))),"ul")

    model_log_html("A value for max sampling capacity must be defined, therefore these will be set to 1,000 in order to run the model.")

    

   # any missing maxsample should be set to 1,000
   subadmin_params.loc[(subadmin_params['include'] == True) & 
                      (subadmin_params['maxsample'].isna()),
                           'maxsample'] = 1000

   

###############################
# SAMPLE DATA
# If a person specifies 1 or more years look back then need to get sample
# data to get count by year - otherwise do not need sample data


# get the specified target year (take first part of season year, e.g. 2024-25 -> 2024)
target_yr = pd.to_numeric(params['season_year'][0:4])
num_yrs_lookback = params['years_lookback']
yrs_lookback = pd.Series(target_yr - pd.Series(range(1, num_yrs_lookback+1)))
yr_diff = pd.Series(yrs_lookback-target_yr)
yrs_lookback = yrs_lookback.astype(str) + '-' + (yrs_lookback + 1).astype(str).str[-2:]
yrs_inf = pd.DataFrame({'season_year': yrs_lookback,
                        'yr_diff' : yr_diff})

new_row_df = pd.DataFrame({'season_year': params['season_year'], 
                           'yr_diff': [0]})
yrs_inf = pd.concat([yrs_inf, new_row_df], ignore_index=True)
yrs_inf = yrs_inf.sort_values(by='yr_diff', ascending=False).reset_index(drop=True).drop_duplicates()


if (num_yrs_lookback > 0) :
    try: 
        with open(sample_file_path, 'r') as f:
            samples = ndjson.load(f)
            logging.info("Sample file loaded successfully")
    except FileNotFoundError:
        logging.error("samples.ndjson file does not exist.")
        model_log_html("ERROR", "h4")
        model_log_html("Samples (sample.ndjson) file not found. Sample data are required to run this model. Execution halted.")
        sys.exit(1)

    
    # logging information for years look back
    model_log_html("Years lookback (historical samples)", "h4")
    model_log_html("User requested " + str(num_yrs_lookback) + " year(s) of sampling lookback from the target year " + str(yrs_inf[yrs_inf['yr_diff'] == 0]['season_year'][0]) + ".")
    model_log_html("Historical sampling from season-year(s) " + ', '.join(yrs_lookback.unique()) + " will be used for the Sample Allocation Model.")


    # list of samples sub admin units
    sample_seasonyr = []
    for s in samples:
        # get season year, id, and test status
        season_year = s.get('season_year', 'NA')
        sample_id = s.get('sample_id', 'NA')
        
        if 'test_status' in s.keys() :
            test_status = s.get('test_status', 'NA')
        else :
            first_definitive = next(
                (test for test in s.get("tests", []) if test.get("selected_definitive")), 
                None
            )
            if first_definitive:
                test_status = first_definitive['result']
            else :
                test_status = None
    
        # get the sub admin id from where the sample came (nested dictionary)
        subadmin_area = s.get('_sub_administrative_area', {})
        subadmin_id = subadmin_area.get('_id', 'NA') if isinstance(subadmin_area, dict) else 'NA'
    
        if ( (season_year != 'NA') and (sample_id != 'NA') and (subadmin_id != 'NA')  ):
            sample_seasonyr.append([season_year, subadmin_id, test_status, sample_id])
            
        
    # samples by season and subadmin unit  
    samples_by_subadmin_yr = pd.DataFrame(sample_seasonyr, 
                                          columns = ['season_year','sub_administrative_area','test_status','sample_id'])
        
    
    # count number of positive samples by season year and sub admin area
    pos_subadminunit  = samples_by_subadmin_yr[samples_by_subadmin_yr['test_status'] == "Detected"].groupby([
                                        'season_year',
                                        'sub_administrative_area',
                                     #   'test_status'
                                    ], as_index=False, dropna = False).agg({
                                        'sample_id': 'count'
                                    })
    pos_subadminunit  = pd.merge(pos_subadminunit,subadmin_params[['sub_administrative_area',
                                               'sub_administrative_area_name']],
                                              how= 'left',
                                              on = 'sub_administrative_area')
    
    # filter to just included admin areas
    pos_subadminunit  = pos_subadminunit[pos_subadminunit['sub_administrative_area'].isin(list_included_subadmins) &
                                         pos_subadminunit['season_year'].isin(yrs_lookback)]              
    # if any sub admin units have positive cwd case, break
    if not pos_subadminunit.empty :
        logging.error("The following sub-administrative unit(s) have positive cases in the listed historical season-year(s) and need to be removed from the model:")
        model_log_html("ERROR", "h4")
        model_log_html("The following sub-administrative unit(s) have positive cases in the listed historical season-year(s) and need to be removed from the model:")
       
        for yr, group in pos_subadminunit.groupby('season_year'):
            years = group['sub_administrative_area_name'].unique()
            logging.error(f"{yr}: {', '.join(sorted(years))}")

            model_log_html(f"{yr}:","u") 
            model_log_html(f"{', '.join(sorted(years))}","ul") 

        sys.exit(70)


            
    # count number of samples by season year and sub admin area
    samples_by_subadmin_yr_long  = samples_by_subadmin_yr[samples_by_subadmin_yr['season_year'].isin(yrs_lookback)].groupby([
                                        'season_year',
                                        'sub_administrative_area',
                                    ], as_index=False, dropna = False).agg({
                                        'sample_id': 'count'
                                    })
    
    
    # generate combination of all sub admin areas and number of years look back
    combinations = list(itertools.product(subadmins['sub_administrative_area'], 
                                            yrs_lookback))
    combinations_df = pd.DataFrame(combinations, 
                                     columns=['sub_administrative_area', 
                                              'season_year'])
    combinations_df = pd.merge(combinations_df,subadmins,how= 'left',on = 'sub_administrative_area')
    
    # fill all specified historical years, fill in 0's where no samples
    all_hist = pd.merge(combinations_df,
                         samples_by_subadmin_yr_long,
                          how = 'outer', 
                          on = ['sub_administrative_area',
                                'season_year'])
    all_hist['sample_id'] =  all_hist['sample_id'].fillna(0).astype(int)
    all_hist.rename(columns = {'sample_id':'num_samples'},inplace = True)
    
    # all historical sampling for viz
    hist_sample = pd.merge(all_hist,yrs_inf,how  = 'outer', on = 'season_year')
    hist_sample.dropna(subset = ['sub_administrative_area'],inplace = True)
    hist_sample['target_season_year'] = params['season_year']
   

    hist_path  = base_path / "hist_sample.csv"
    hist_sample.to_csv(hist_path, index = False)
    
    
    # filter to just selected years looking back 
    all_hist = all_hist[all_hist['season_year'].isin(yrs_lookback)]

    # create index of 'historical year' for ordering/labels           
    all_hist['historical_yr'] = target_yr - pd.to_numeric(all_hist['season_year'].str[:4])
    
    
    
    
    
    # pivot table wide by historical year
    samples_by_subadmin_yr_wide = all_hist.pivot(index = 'sub_administrative_area',
                                      columns = 'historical_yr',
                                      values = 'num_samples').reset_index()
    # rename variables for file
    samples_by_subadmin_yr_wide.columns = ["sub_administrative_area"] + [f"Historical sampling - year{i}" for i in range(1, num_yrs_lookback+1)]
    
    # fill any missing with 0's
    samples_by_subadmin_yr_wide.fillna(0,inplace= True)
    
    

    # logging information for historical sampling
    model_log_html("Total number of historical samples by year", "h4")

    all_hist_sub= all_hist[all_hist['sub_administrative_area'].isin(list_included_subadmins)]
    # number of samples by season year
    numsamps_byyr = all_hist_sub.groupby(['season_year'], as_index=False, dropna = False).agg({
                            'num_samples': 'sum'
                        }).sort_values(by='season_year', ascending=False)
    
    for index, row in numsamps_byyr.iterrows():
        model_log_html("There were " + str(format(int(row['num_samples']),",")) + " samples from the selected sub-administrative units in " + str(row['season_year']) +".")

    



      
    # merge samples over time with the parameters and subadmin units
    subadmin_params_samps = pd.merge(subadmin_params,
                                    samples_by_subadmin_yr_wide,
                                on = ['sub_administrative_area'],
                                how = 'outer')



    # get list of subadmins and whether to be included
    all_hist_incl = pd.merge(subadmin_params[['sub_administrative_area','include']],
                            all_hist,
                        on = ['sub_administrative_area'],
                        how = 'outer')
   
    #  number of subadmins missing samples by season year
    nummissing_byyr_bool = ((all_hist_incl['num_samples'].isna() | (all_hist_incl['num_samples'] == 0)) &
                            (all_hist_incl['include'] == True))
    nummissing_byyr = (nummissing_byyr_bool).sum()
    if nummissing_byyr > 0 :
        # logging information
        model_log_html("Subaministrative units with no samples", "h4")
      
        nosamps = all_hist_incl[nummissing_byyr_bool].sort_values(by = 'historical_yr',ascending = True)
      
        model_log_html("Out of " + str((subadmin_params['include'] == True).sum()) + " selected subadministrative units (out of " + str(len(subadmins)) + 
                       " total) from the provider, the following subaministrative units have 0 samples in the following season-years:")
        for index in nosamps['season_year'].unique() :
             nosamps0 = nosamps[nosamps['season_year'] == index]
             logging.error(index + ": " + ', '.join(sorted(nosamps0['sub_administrative_area_name'].astype(str))))
             model_log_html(index + ": " + ', '.join(sorted(nosamps0['sub_administrative_area_name'].astype(str))),"ul")

        
        model_log_html("User's may consider re-running the model to exclude sub-administrative units or years that have no sampling data.")

        
     
else :
  model_log_html("Years lookback (historical samples)", "h4")
  model_log_html("User specified 0 years look back. No historical data will be used.")
  subadmin_params_samps = subadmin_params
             




###############################
# HAZARDS OUTPUT
# Use hazard model output to calculate introduction rate
#   1. From output of the hazard model, get proportion of the total 
#      quota allocated to each subadmin unit. 
#   2. multiply each of those proportions by the probability that disease will
#      appear somewhere in the state = 0.66
#     For more details on this value, refer to (Wang et al., 2025)


# load hazards model output
try: 
    with open(hazards_path, 'r') as f:
        hazards = json.load(f)
        logging.info("Hazard model output file loaded successfully")
except FileNotFoundError:
    # The model cannot be executed without a params file. Exit with an error immediately.
      logging.error("risk_model.json (hazards output) File does not exist.")
      model_log_html("ERROR", "h4")
      model_log_html("risk_model.json (hazards output) File does not exist.")
      sys.exit(1)


# get quotas by sub admin area
subadmin_quotas = pd.DataFrame(hazards)[['subadminID', 'subadminName', 'totalQuota']]

# proportion of quota allocated to each sub admin area
subadmin_quotas['total_quota'] = subadmin_quotas['totalQuota'].sum()
subadmin_quotas['prop_quota'] = subadmin_quotas['totalQuota']/subadmin_quotas['total_quota']
# multiple 0.66 for introduction rate
subadmin_quotas['introduction_rate'] = 0.66 * subadmin_quotas['prop_quota']

# rename column for linking
subadmin_quotas.rename(columns = {'subadminID' : 'sub_administrative_area'}, inplace = True)



# merge in introduction rate
subadmin_params_samps_probs = pd.merge(subadmin_params_samps, 
                                      subadmin_quotas[['sub_administrative_area','introduction_rate']],
                             on = ['sub_administrative_area'],
                             how = 'outer')


# if any introduction rates are 0 they should be excluded
zero_introrate_bool = (subadmin_params_samps_probs['introduction_rate'] == 0) & (subadmin_params_samps_probs['include'] == True)
zero_introrate = (zero_introrate_bool).sum()
if zero_introrate > 0 :
      # The model cannot be executed without a params file. Exit with an error immediately.
      logging.error("The following subadministrative units that have an introduction rate of 0 based on the selected hazards quota model:")
      logging.error(', '.join(sorted(subadmin_params_samps_probs['sub_administrative_area_name'][zero_introrate_bool].astype(str))))
      logging.error("These subadministrative units should be excluded or a different hazards quota model should be selected.")

      model_log_html("ERROR", "h4")
      model_log_html("The following subadministrative units that have an introduction rate of 0 based on the selected hazards quota model:")
      model_log_html(', '.join(sorted(subadmin_params_samps_probs['sub_administrative_area_name'][zero_introrate_bool].astype(str))),"ul")
      model_log_html("These subadministrative units should be excluded or a different hazards quota model should be selected.")
      sys.exit(70)





############################### 
# FINAL OUTPUT

# rename columns for consistency with model code
rename_map = {'sub_administrative_area':'_id',
              'sub_administrative_area_name':'_id_name',
              'include':'include',
              'cost' :'unit_cost',
              'maxsample':'MaxSample'}
subadmin_params_samps_probs = subadmin_params_samps_probs.rename(columns={k: v for k, v in rename_map.items() if k in subadmin_params_samps_probs.columns})

# capture historical columns and unit cost columns if they exist
historical_cols = [col for col in subadmin_params_samps_probs.columns if col.startswith("Historical")]
maxsample_cols = [col for col in subadmin_params_samps_probs.columns if col == "MaxSample"]
unitcost_cols = [col for col in subadmin_params_samps_probs.columns if col == "unit_cost"]

final_dat = subadmin_params_samps_probs[['_id',
                                         '_id_name',
                                        'include'] +    
                                        maxsample_cols +
                                        historical_cols +
                                        ['introduction_rate'] +
                                        unitcost_cols
                                        ]


final_dat.to_csv(output_path, index = False)


model_label  = str(provider[0]['_administrative_area']['administrative_area']) + "_model" + str(params['mode']) + "_" + str(params['season_year'])
# create mode.csv as input for optimization models
mode_file = [model_label,
             params['mode'],
             annual_budget,
             params['season_year'],
             annual_growth_rate]
mode_columns = ['_model','mode_selection','budget','stat_year','annual_growth_rate']
mode_df = pd.DataFrame([mode_file],columns = mode_columns)
mode_df.to_csv(mode_path, index = False)






model_log_html("Data processing steps complete and ready for modeling steps.","h3")
logging.info("Data processing steps complete and moving to optimization steps")


model_log_html("Running model Mode " + str(params['mode']) + " for the target season-year " + str(params['season_year']) + " with an annual growth rate of " + str(annual_growth_rate) + ".")
if model_mode in (2,3) :
  model_log_html("Using an annual budget of $" + str(format(int(params['annual_budget']),",")) )
