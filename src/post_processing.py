'''
Script Name: Post Processing for Sample Allocation - create JSON file for output
Author: Sarah Sernaker, ss3376@cornell.edu, Cornell University
Description:  Takes all of the csv files output from the model and pre-processing and cleans and condenses the
    information into a single JSON file to facilitate CWD Warehouse functionality.
    This script also takes the variable probability_disease_free variable in "posterior_results.csv" 
    and transforms it into the variable probability_disease_present, and the former is removed.
    Similarly, for modes 2 and 3, the "probability_disease_free.csv" model output is transformed into 
    "probability_disease_present.csv" and the former is removed.
    Additionally, this script calculates and appends the column perecent_change to "cost_analysis.csv" for mode 3.
 
Inputs: 
   Note - Input files depend on the model model, as denoted in parentheses.
   
   posterior_results.csv (Models 1, 2, and 3)
   total_unobserved_spread.csv (Models 2 and 3)
   probability_disease_free.csv (Models 2 and 3)
   probability_prevalence_1_0.csv (Models 2 and 3)
   sample_size.csv (Models 2 and 3)
   cost_analysis.csv (Model 3)

   hist_sample.csv (if specified in model)
       
 
Outputs: 
   sam_output.json (Models 1, 2, and 3)
   probability_disease_present.csv (Models 2 and 3)
   
Updates:
   posterior_results.csv (Models 1, 2, and 3)
   cost_analysis.csv (Model 3)

  info.html
  execution_log.log 
  attachments.json

'''



###############################
# Environment

import pathlib
import pandas as pd
import json
import re
import logging
import sys
import os

###############################
# FILE PATH VARIABLES

base_path = pathlib.Path("/data")

attachments_folder = base_path / "attachments" 
ATTACHMENTS_JSON = base_path / "attachments.json"

hist_path = base_path /  "hist_sample.csv"
prob_disease_pres_path = attachments_folder / "probability_disease_present.csv"
prob_disease_free_path = attachments_folder / "probability_disease_free.csv"
posterior_results_path = attachments_folder / "posterior_results.csv"
cost_analysis_path = attachments_folder / "cost_analysis.csv"

parameters_file_path  = base_path / "params.json"
output_json = attachments_folder / "sam_output.json"
attachments_json_path = base_path / "attachments.json"

model_metadata_log_file = base_path / "attachments" / "info.html"
logging_path = base_path / "attachments" / "execution_log.log"


# --- Logging setup ---
# File + append, DEBUG‑level, custom datefmt
logging.basicConfig(
    level     = logging.INFO,
    filename  = str(logging_path),
    filemode  = "a",
    datefmt   = "%Y-%m-%d %H:%M:%S",
    format    = "%(asctime)s - %(levelname)s - %(message)s"
)


# Uncaught exception handler
def handle_uncaught_exception(type, value, traceback):
  logging.error(f"{type} error has occurred with value: {value}. Traceback: {traceback}")
sys.excepthook = handle_uncaught_exception

###############################
# FUNCTIONS

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



## csv files 
csv_data = {}
for file in attachments_folder.glob("*.csv"):
    name = file.stem  # filename without .csv extension
    csv_data[name] = pd.read_csv(file)

# add historical data if exists
try: 
    csv_data['hist_sample'] = pd.read_csv(hist_path)
except FileNotFoundError:
    model_log_html('Post-processing: no historical sampling data included.')
    logging.info('Post-processing: no historical sampling data included.')
    pass


# parameter
with open(pathlib.Path(parameters_file_path), 'r') as f:
    params = json.load(f)
model_mode = params['mode']
target_season_year = params['season_year']



# Initialize output dict
combined_json = {}



## for models 2 & 3 : 
# objective
if 'total_unobserved_spread' in csv_data.keys() :
    combined_json['total_unobserved_spread'] = csv_data['total_unobserved_spread']['total_unobserved_spread'][0].item()
else:
    if model_mode in [2,3]:
      model_log_html('Post-processing: Could not find objective.csv file as expected')
      logging.info("Post-processing: Could not find objective.csv file as expected")


## for model 3
# cost_analysis
if 'cost_analysis' in csv_data.keys():
    df_cost_analysis = csv_data['cost_analysis'].copy().drop_duplicates(subset = "budget_percentage")
    combined_json['cost_analysis'] = []
    df_cost_analysis['budget_percentage'] = list(range(50, 201, 10)) + list(range(250, 401, 50))  
    df_cost_analysis['budget'] = df_cost_analysis['budget'].round(0).astype(int)

    df_cost_analysis.rename(columns = {'objective' : 'total_unobserved_spread'},inplace = True )

    val_div = df_cost_analysis["total_unobserved_spread"][df_cost_analysis['budget_percentage'] == 100].iloc[0]
    df_cost_analysis["percent_change"] = (100*(df_cost_analysis["total_unobserved_spread"]/val_div - 1)).round(1)



    df_cost_analysis.to_csv(cost_analysis_path,index = False)


    combined_json['cost_analysis'] = (df_cost_analysis[['budget_percentage', 'budget', 'total_unobserved_spread','percent_change']].to_dict(orient='records'))

else:
    if model_mode == 3:
      model_log_html('Post-processing: Could not find cost_analysis.csv file as expected')
      logging.info("Post-processing: Could not find cost_analysis.csv file as expected")






## for models 1, 2, & 3
# posterior_results - current probabilities of prevalence
if 'posterior_results' in csv_data.keys() :
    df_posterior_results = csv_data['posterior_results'].copy()
    df_posterior_results.rename(columns = {"state_0": "probability_disease_free",
                                           "state_1": "probability_prevalence_0_5",
                                           "state_2" : "probability_prevalence_1_0"},
                                inplace = True)
    df_posterior_results['probability_disease_present'] = 1 - df_posterior_results['probability_disease_free']
    
    df_posterior_results.sort_values(by = '_id_name',inplace = True)
    df_posterior_results.set_index('site_id', inplace = True)

    df_posterior_results[['_id_name',
                         'probability_disease_present',
                         'probability_prevalence_1_0']].to_csv(posterior_results_path,index = True)


    combined_json['sub_administrative_unit'] = {}
    for idx in df_posterior_results.index:
        combined_json['sub_administrative_unit'][idx] = {
            'name' : df_posterior_results.at[idx, '_id_name'],
            'posterior_results': {
                col: df_posterior_results.at[idx, col].item()
                for col in ['probability_disease_present','probability_prevalence_1_0']
            }
        }
else:
    model_log_html('Post-processing: Could not find posterior_results.csv file as expected')
    logging.info("Post-processing: Could not find posterior_results.csv file as expected")



    
    
## For models 2 & 3 
# probability_disease_free, probability_prevalence_0_5, probability_prevalence_1_0, sample_efforts, sample_size
need = ['probability_disease_free',
        'probability_prevalence_1_0',
        'sample_size']
# check that all of this output is available
if all(key in csv_data for key in need) :
    for key, df in csv_data.items():
        if 'site_id' in df.columns:
            df.sort_values(by = '_id_name',inplace = True)
            df.set_index('site_id', inplace=True)   


    #---- probability of disease free - transform into probability any disease
    df_prob_dis_free = csv_data['probability_disease_free'].copy()
    # Subtract each numeric column from 1
    df_prob_dis_free_numeric = df_prob_dis_free.select_dtypes(include='number')  # select only numeric columns
    df_prob_dis_free[df_prob_dis_free_numeric.columns] = 1 - df_prob_dis_free_numeric
    
    # write the flipped probabilities and delete the original ones
    df_prob_dis_free.to_csv(prob_disease_pres_path,index = True)
    os.remove(prob_disease_free_path)
    
    add_item_to_json_file_list(
      ATTACHMENTS_JSON,
      {"filename": "probability_disease_present.csv", 
       "content_type": "text/csv", 
       "role": "downloadable"}
    )
 
    
    # probability disease prevalence 1.0% or greater
    df_prob_prev_1_0 = csv_data['probability_prevalence_1_0'].copy()
    
    # sample size on optimal plan
    df_sample_size = csv_data['sample_size'].copy()

    # capture historical data if included
    if 'hist_sample' in csv_data.keys() :
        df_hist_sampling = csv_data['hist_sample'].copy()
        target_season = df_hist_sampling["target_season_year"][0]
    

    # Loop sub admin units to add time point measures
    for idx in df_prob_dis_free.index:
        if idx not in combined_json['sub_administrative_unit']:
            combined_json['sub_administrative_unit'][idx] = {}  # only if it doesn't already exist
        
        combined_json['sub_administrative_unit'][idx]['timepoints'] = []
        
        try:
            if not df_hist_sampling.empty  :
                sub_hist = df_hist_sampling[df_hist_sampling['sub_administrative_area'] == idx]
                for _,row in sub_hist.iterrows():
                  combined_json['sub_administrative_unit'][idx]['timepoints'].append(
                    {"sample_size" : 0 if pd.isna(row['num_samples']) else row['num_samples'],
                      "time_point" : row['yr_diff'],
                      "season_year" : row['season_year'],
                     "estimate" : "observed"}
                  )
        except  :
            pass
        
        
        
        no_cols = [f"time_{i}" for i in range(21, 31)]
        
        for col in df_prob_dis_free.columns:
            if col != 'site_id' and col != '_id_name' and (col not in no_cols):
                colnum = re.search(r'\d+', col) 
                colnum = int(colnum[0])
                
                future_yr = int(target_season_year[:4]) + colnum
                future_yr_str = str(future_yr) + "-" + str((future_yr+1))[2:]

                
                combined_json['sub_administrative_unit'][idx]['timepoints'].append(
                  {
                    'probability_disease_present': df_prob_dis_free.at[idx, col].item(),
                    'probability_prevalence_1_0': df_prob_prev_1_0.at[idx, col].item(),
                    'sample_size': df_sample_size.at[idx, col].item(),
                    'years_ahead' : colnum,
                    'season_year': future_yr_str,
                    'estimate' : "predicted"
                  }
                )
else :
  if model_mode in [2,3]:
    missing_values = [val for val in need if val not in csv_data.keys()]
    if not missing_values:
        for v in missing_values:
            model_log_html('Post-processing: Could not find' + v + '.csv file as expected')
            logging.info('Post-processing: Could not find' + v + '.csv file as expected')





with open(output_json, "w") as f:
    json.dump(combined_json, f, indent=2)


# Append execution log to attachments.json for developer feedback
attachment = {
  "filename": "sam_output.json", 
  "content_type": "application/json", 
  "role": "primary"
  }
add_item_to_json_file_list(attachments_json_path, attachment)


model_log_html("Data post-processing steps complete.")
logging.info("Data post-processing steps complete.")

