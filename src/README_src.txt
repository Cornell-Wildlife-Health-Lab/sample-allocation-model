These files are the set of python scripts that cleans the data exported by the CWD Data Warehouse (https://cwd-data.org/), runs the model and gets output, and cleans and organizes the output.

1. input_processing.py
Takes the output from the CWD warehouse, extracts and cleans necessary data and exports csv files to be fed into the model. It also initializes logging files that provide detailed updates and error messages (info.html, execution_log.log, attachments.json).
Warehouse output files are organized in JSON files, one for historical sample data (if specified in model), one for the model parameters, one for the geographic description of the agency, and one for the output of the Risk-weighted Surveillance Quotes (https://pages.github.coecis.cornell.edu/CWHL/CWD-Data-Warehouse/risk-weighted-surveillance-quotas.html).
Exports the following csv files for the modeling steps: 
	mode.csv 
	SAM_input.csv
	hist_samples.csv (if specified in model)


2. main.py
Takes the output from input_processing.py (i.e. mode.csv and SAM_input.csv) and runs models depending on the Model Mode chosen (see Warehouse documentation for model details: https://pages.github.coecis.cornell.edu/CWHL/CWD-Data-Warehouse/SAM.html). It also appends the logging files that provide detailed updates and error messages (info.html, execution_log.log, attachments.json) - if created in input_processing.py, if not they will be initialized here.
This script references all of the functions in the folder 'sam_model' as additional model and algorithm specifications.
Exports the following csv files depending on the model, as denoted in parentheses.
    posterior_results.csv (Models 1, 2, and 3)
    total_unobserved_spread.csv (Models 2 and 3)
    probability_disease_free.csv (Models 2 and 3)
    probability_prevalence_1_0.csv (Models 2 and 3)
    sample_size.csv (Models 2 and 3)
    cost_analysis.csv (Model 3)
*NOTE - Users can run the model on their own (e.g. independent of the CWD Warehouse) starting from this step if the script is supplied the mode.csv and SAM_input.csv file (see the README_data.txt file in the 'data' folder).*




3. post_processing.py
Takes all of the output from main.py and the historical sampling file (hist_sample.py) from input_processing, if available, and cleans and condenses the information into a single JSON file to facilitate CWD Warehouse functionality. This script also takes the variable probability_disease_free variable in "posterior_results.csv" and transforms it into the variable probability_disease_present, and the former is removed. Similarly, for modes 2 and 3, the "probability_disease_free.csv" model output is transformed into "probability_disease_present.csv" and the former is removed. Additionally this script calculates and appends the column perecent_change to "cost_analysis.csv" for mode 3. 
It also appends the logging files that provide detailed updates and error messages (info.html, execution_log.log, attachments.json) - if already created, if not they will be initialized here.

Exports the following files depending on the model, as denoted in parentheses.
   sam_output.json (Models 1, 2, and 3)
   probability_disease_present.csv (Models 2 and 3)
   
Updates the following files depending on the model, as denoted in parentheses.
   posterior_results.csv (Models 1, 2, and 3)
   cost_analysis.csv (Model 3)





Other files
The file run_scripts.sh in the 'scripts' folder is a Bash script that is used in the Docker containerized version to tell the computer to run the files in order, or to provide an error if it fails. The "requirements.txt" file in the 'software' folder lists the necessary Python package versions to download for the Docker container.
