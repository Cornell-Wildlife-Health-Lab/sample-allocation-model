'''
Script Name: Sample Allocation using Optimal Control Model 
Author: Yu Gong, yu.gong@queensu.ca, Queens University
Description: Uses the processed data from the warehouse with the user-specified
parameters and runs the selected model.
 
Run model selection based on user input:
  1) Posterior estimation only
  2) Posterior + single optimization
  3) Posterior + single optimization + cost‐analysis grid search
  
Inputs: 
  SAM_input.csv
  mode.csv
 
Outputs: 
  Mode 1:
    posterior_results.csv
  
  Mode 2:
    posterior_results.csv
    total_unobserved_spread.csv
    probability_disease_free.csv
    probability_prevalence_1_0.csv
    sample_size.csv
    
  Mode 3:
    posterior_results.csv
    total_unobserved_spread.csv
    probability_disease_free.csv
    probability_prevalence_1_0.csv
    sample_size.csv
    cost_analysis.csv
       
    
  info.html
  execution_log.log 
  attachments.json

'''


import sys
import math
import logging
import pandas as pd
import numpy as np
import pathlib
import json
import os 

from sam_model.posterior import posterior_calculate
from sam_model.create import create_model
from sam_model.set_parameters import set_parameters_pyomo
from sam_model.extract import extract_results_pyomo


base_path = pathlib.Path("/data")

# Inputs
MODE_PATH   = base_path / "mode.csv"
SAMPLE_PATH = base_path / "SAM_input.csv"
RESULTS_DIR     = base_path / "attachments"

ATTACHMENTS_JSON = base_path / "attachments.json"
LOG_PATH          = base_path / "attachments" / "execution_log2.log"
HTML_LOG_PATH     = base_path / "attachments" / "info.html"



# make sure results/ exists
RESULTS_DIR.mkdir(exist_ok=True)

# initialize attachments.json if missing
if not ATTACHMENTS_JSON.exists():
    ATTACHMENTS_JSON.write_text("[]")

def add_item_to_json_file_list(file_path: pathlib.Path, new_item: dict):
    """
    Append new_item into the JSON array stored at file_path.
    """
    data = json.loads(file_path.read_text())
    if not isinstance(data, list):
        raise ValueError(f"{file_path} does not hold a JSON list")
    data.append(new_item)
    file_path.write_text(json.dumps(data, indent=2))

# --- Logging setup ---
# File + append, DEBUG‑level, custom datefmt
logging.basicConfig(
    level     = logging.INFO,
    filename  = str(LOG_PATH),
    filemode  = "a",
    datefmt   = "%Y-%m-%d %H:%M:%S",
    format    = "%(asctime)s - %(levelname)s - %(message)s"
)

# Also echo INFO+ to console
console = logging.StreamHandler()
console.setLevel(logging.INFO)
console.setFormatter(logging.Formatter(
    "%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
))
logging.getLogger().addHandler(console)

# Uncaught exception handler
def handle_uncaught_exception(type, value, traceback):
  logging.error(f"{type} error has occurred with value: {value}. Traceback: {traceback}")
sys.excepthook = handle_uncaught_exception

logger = logging.getLogger(__name__)

# ─── Utility: render log → HTML ────────────────────────────────────────────────
def render_log_to_html(log_txt: pathlib.Path, html_out: pathlib.Path):
    def clean_log_entry(log):
        parts = log.split(" - ", 2)
        return parts[2] if len(parts) > 2 else log
    
    with log_txt.open("r") as fin, html_out.open("a") as fout:
        fout.write(f"<h2>Optimization module execution log</h3>\n")
        for line in fin:
            safe = (line.replace("&", "&amp;")
                        .replace("<", "&lt;")
                        .replace(">", "&gt;")
                        .rstrip())
            safe = clean_log_entry(safe)
            fout.write(f"<p>{safe}</p>\n")
        fout.write("</body></html>")




def compute_posteriors(inc_df: pd.DataFrame,
                       hist_cols: list,
                       speed_spread: float) -> dict:
    """
    Compute posterior belief vectors for each site.
    Returns a dict {site_id: np.ndarray([b0,b1,b2]) }.
    """
    post = {}
    for _, row in inc_df.iterrows():
        sid = row['_id']
        theta0 = float(row['introduction_rate'])
        U_history = [row[c] for c in hist_cols]
        post[sid] = posterior_calculate(
        theta0  = theta0,
        U_history = U_history,
        Speed_spread = speed_spread)
    return post


def run_posterior_only(mode_df, inc_df, hist_cols):
    """Mode 1: compute & save posterior only."""
    try:
        growth_rate = float(mode_df.loc[0, 'annual_growth_rate'])
        # also catch NaN
        if math.isnan(growth_rate):
            raise ValueError("annual_growth_rate is NaN")
    except Exception:
        logger.warning(
            "‘annual_growth_rate’ missing or invalid; "
            "defaulting Speed_spread to 0.2"
        )
        growth_rate = 0.2

    try:
        post = compute_posteriors(inc_df, hist_cols, growth_rate)
    except ValueError as ve:
        logger.error(f"Posterior update failed: {ve}")
        sys.exit(11)

    if not post:
        logger.error("No posterior results to save.")
        sys.exit(1)

    df = pd.DataFrame.from_dict(
        post, orient='index',
        columns=['state_0', 'state_1', 'state_2']
    )
    df.index.name = 'site_id'
    df = df.reset_index()
    
    inc_df2 = inc_df[['_id','_id_name']].rename(columns = {'_id':'site_id'})
    
    df = pd.merge(inc_df2,
             df, 
             on = 'site_id',
             how = 'right')
    

    # Define file paths
    csv_path = RESULTS_DIR / "posterior_results.csv"

    # Save CSV
    df.to_csv(csv_path, index=False)
    add_item_to_json_file_list(
        ATTACHMENTS_JSON,
        {
            "filename": csv_path.name,
            "content_type": "text/csv",
            "role": "downloadable"
        }
    )

    logger.info("The posterior update completed successfully")


def run_growth_optimization(mode_df, inc_df, hist_cols):
    """Mode 2: posterior + one optimization run."""
    try:
        budget = float(mode_df.loc[0, 'budget'])
        if math.isnan(budget):
            raise ValueError("budget is NaN")
    except Exception:
        logger.error("The budget value is missing or invalid; aborting.")
        sys.exit(20)

    try:
        growth_rate = float(mode_df.loc[0, 'annual_growth_rate'])
        if math.isnan(growth_rate):
            raise ValueError("annual_growth_rate is NaN")
    except Exception:
        logger.warning(
            "‘annual_growth_rate’ missing or invalid; "
            "defaulting Speed_spread to 0.2"
        )
        growth_rate = 0.2

    mh_s = inc_df.set_index('_id')['MaxSample']
    if mh_s.isna().any():
        missing_mh = mh_s[mh_s.isna()].index.tolist()
        logger.warning(
            f"Missing MaxSample for site {missing_mh}; "
            "defaulting to 1000"
        )
        mh_s = mh_s.fillna(1000)

    cu_s = inc_df.set_index('_id')['unit_cost']
    invalid_cu = cu_s[cu_s.isna()].index.tolist()
    if invalid_cu:
        logger.error(f"Missing unit_cost for site {invalid_cu}; aborting.")
        sys.exit(20)

    try:
        post = compute_posteriors(inc_df, hist_cols, growth_rate)
    except ValueError as ve:
        logger.error(f"Posterior update failed: {ve}")
        sys.exit(21)

    if not post:
        logger.error("No posterior results to save")
        sys.exit(1)

    # Save posterior results
    df_post = pd.DataFrame.from_dict(
        post, orient='index',
        columns=['state_0', 'state_1', 'state_2']
    )
    df_post.index.name = 'site_id'
    df_post = df_post.reset_index()

    inc_df2 = inc_df[['_id','_id_name']].rename(columns = {'_id':'site_id'})
     
    df_post = pd.merge(inc_df2,
              df_post, 
              on = 'site_id',
              how = 'right')

    csv_post = RESULTS_DIR / "posterior_results.csv"

    df_post.to_csv(csv_post, index=False)
    add_item_to_json_file_list(
        ATTACHMENTS_JSON,
        {"filename": csv_post.name, "content_type": "text/csv", "role": "downloadable"}
    )

    logger.info("The posterior update completed successfully")

    # prepare for optimization
    site_ids      = list(post.keys())
    I_max         = len(site_ids)
    index_to_site = {i+1: sid for i, sid in enumerate(site_ids)}
    th_s   = inc_df.set_index('_id')['introduction_rate']
    cu_dict         = {i: float(cu_s.at[index_to_site[i]]) for i in index_to_site}
    theta_dict      = {i: float(th_s.at[index_to_site[i]]) for i in index_to_site}
    lambda_override = {i: float(mh_s.at[index_to_site[i]]) for i in index_to_site}

    init_P0 = {i: post[index_to_site[i]][0] for i in index_to_site}
    init_P1 = {i: post[index_to_site[i]][1] for i in index_to_site}
    init_P2 = {i: post[index_to_site[i]][2] for i in index_to_site}

    # overrides & scalars
    T_values        = 30
    theta1_override = {i: growth_rate for i in index_to_site}
    p1_e_override   = {i: 1.5         for i in index_to_site}
    p2_e_override   = {i: 1.5         for i in index_to_site}
    v_override      = {(t,i):0        for t in range(T_values+1)
                                  for i in index_to_site}
    p1_val, p2_val                  = 0.005, 0.01
    alpha0_val, alpha1_val          = 0.05, 0.01
    c1_val, c2_val, cv_val          = 1, 1, 100

    # build & solve
    model = create_model(T_values, I_max)
    set_parameters_pyomo(
        model,
        T_values, I_max, budget,
        cu_dict, init_P0, init_P1, init_P2,
        lambda_override, theta_dict, theta1_override,
        v_override, p1_val, p2_val,
        alpha0_val, alpha1_val,
        p1_e_override, p2_e_override,
        c1_val, c2_val, cv_val
    )
    try:
        time, u_vals, P0_vals, P1_vals, P2_vals, obj = extract_results_pyomo(model,verbose=False)
    except Exception as e:
        logger.error(f"Optimization failed: {e}")
        sys.exit(22)
    logger.info("Optimization module completed successfully")

    # Save objective
    df_obj = pd.DataFrame({"total_unobserved_spread": [obj]})
    csv_obj = RESULTS_DIR / "total_unobserved_spread.csv"
    samples_by_site = {
        sid: float(mh_s.at[sid])
        for sid in mh_s.index
    }

    df_obj.to_csv(csv_obj, index=False)
    add_item_to_json_file_list(
        ATTACHMENTS_JSON,
        {"filename": csv_obj.name, "content_type": "text/csv", "role": "downloadable"}
    )
    

    # ─── Save  tables (sampling efforts, sampling sizes, state probability) ──────────────────────────────────
    def save_wide(data, name, period, transform=None):
        remapped = {index_to_site[i]: series for i, series in data.items()}
        if transform:
            remapped = {
                sid: [transform(val, sid) for val in series]
                for sid, series in remapped.items()
            }
        dfw = pd.DataFrame.from_dict(
            remapped, orient='index',
            columns=[f"time_{t}" for t in period]
        )
        dfw.index.name = 'site_id'
        dfw = dfw.reset_index()


        inc_df2 = inc_df[['_id','_id_name']].rename(columns = {'_id':'site_id'})
         
        dfw = pd.merge(inc_df2,
                  dfw, 
                  on = 'site_id',
                  how = 'right')

        csvw = RESULTS_DIR / f"{name}.csv"

        dfw.to_csv(csvw, index=False)
       
        

    # sampling efforts (u)
    #save_wide(u_vals, "sample_efforts", time)
    # sample sizes = floor(u * MaxSample)
    save_wide(
        u_vals, "sample_size",
        time,
        transform=lambda u_t, sid: math.floor(u_t * samples_by_site[sid])
    )
    # probabilities
    save_wide(P0_vals, "probability_disease_free", time)
    #save_wide(P1_vals, "probability_prevalence_0_5", time)
    
    
    save_wide(P2_vals, "probability_prevalence_1_0", time)
    add_item_to_json_file_list(
         ATTACHMENTS_JSON,
         {"filename": "probability_prevalence_1_0.csv", "content_type": "text/csv", "role": "downloadable"}
     )
    
    
    logger.info("Optimization results have been saved")

def run_cost_analysis(mode_df, inc_df, hist_cols):
    """Mode 3: Mode2 logic + grid search over discounted budgets."""
    # first run Mode2
    run_growth_optimization(mode_df, inc_df, hist_cols)

    # then grid‐search
    budget_input = float(mode_df.loc[0, 'budget'])
    growth_rate  = float(mode_df.loc[0, 'annual_growth_rate'])
    discounts = np.concatenate([
        # np.arange(0.01,  0.06, 0.01),
        # np.arange(0.05,  2.05, 0.05),
        np.arange(0.5,  2.1, 0.1),
        np.arange(2.0,   4.5,  0.5)
    ])

    results3 = []
    site_ids      = list(inc_df['_id'])
    I_max         = len(site_ids)
    index_to_site = {i+1: sid for i,sid in enumerate(site_ids)}

    cu_s   = inc_df.set_index('_id')['unit_cost']
    th_s   = inc_df.set_index('_id')['introduction_rate']
    mh_s   = inc_df.set_index('_id')['MaxSample']

    for d in discounts:
        # pick horizon
        if   d < 0.05:    T = 50
        elif d < 0.2:     T = 40
        elif d < 2.0:     T = 30
        elif d < 3.0:     T = 25
        else:             T = 20

        bud = budget_input * d
        post = compute_posteriors(inc_df, hist_cols, growth_rate)

        init_P0 = {i: post[index_to_site[i]][0] for i in index_to_site}
        init_P1 = {i: post[index_to_site[i]][1] for i in index_to_site}
        init_P2 = {i: post[index_to_site[i]][2] for i in index_to_site}

        cu_dict         = {i: float(cu_s.at[index_to_site[i]]) for i in index_to_site}
        theta_dict      = {i: float(th_s.at[index_to_site[i]]) for i in index_to_site}
        lambda_override = {i: float(mh_s.at[index_to_site[i]]) for i in index_to_site}

        theta1_o = {i: growth_rate for i in index_to_site}
        p1_e_o   = {i: 1.5 for i in index_to_site}
        p2_e_o   = {i: 1.5 for i in index_to_site}
        v_o      = {(t,i):0 for t in range(T+1) for i in index_to_site}
        p1v,p2v = 0.005,0.01
        a0,a1   = 0.05,0.01
        c1v,c2v,cvv = 1,1,100

        m = create_model(T, I_max)
        set_parameters_pyomo(
            m, T, I_max, bud,
            cu_dict, init_P0, init_P1, init_P2,
            lambda_override, theta_dict, theta1_o,
            v_o, p1v,p2v, a0,a1, p1_e_o,p2_e_o,
            c1v,c2v,cvv
        )
        try:
            _, _, _, _, _, obj = extract_results_pyomo(m,verbose=False)
        except Exception as e:
            logger.error(f"Cost analysis failed at {d:.3f}% of total budget: {e}")
            sys.exit(32)
        results3.append({'budget_percentage': d, 'budget': bud, 'objective': obj})
        #logger.info(f"Complete cost analysis on: Percentage={d:.3f}, Budget={bud:.3f}, Expected detection delay={obj:.4f} years")

    df3 = pd.DataFrame(results3)

    # Save CSV
    csv_ca = RESULTS_DIR / "cost_analysis.csv"
    df3.to_csv(csv_ca, index=False)
    add_item_to_json_file_list(
        ATTACHMENTS_JSON,
        {"filename": csv_ca.name, "content_type": "text/csv", "role": "downloadable"}
    )
    logger.info("Cost analysis completed and results have been saved")


def main():
    # Read inputs
    try:
        mode_df   = pd.read_csv(MODE_PATH)
        sample_df = pd.read_csv(SAMPLE_PATH)
    except Exception as e:
        logger.error(f"Failed to read input data: {e}")
        sys.exit(1)

    inc_df = sample_df[sample_df['include'] == True]
    if len(inc_df) < 2:
        logger.error(
            f"Need at least two sites included for optimization, "
            f"but found {len(inc_df)} site."
        )
        sys.exit(1)

    hist_cols = [
        c for c in inc_df.columns
        if c.lower().startswith("historical sampling - year")
    ]
    if not hist_cols:
        logger.warning(
            "No historical sampling exists in input data; "
            "The optimization thus assumes disease-free prior for all sites."
        )
    else:
        hist_cols.sort(key=lambda x: int(x.lower().split("year")[-1]))

    sel = int(mode_df.loc[0,'mode_selection'])
    if   sel == 1:
        run_posterior_only(mode_df, inc_df, hist_cols)
    elif sel == 2:
        run_growth_optimization(mode_df, inc_df, hist_cols)
    elif sel == 3:
        run_cost_analysis(mode_df, inc_df, hist_cols)
    else:
        logger.error(f"model_selection={sel} not supported")
        sys.exit(1)
    render_log_to_html(LOG_PATH, HTML_LOG_PATH)
    
    LOG_PATH2 = base_path / "attachments" / "execution_log.log"
    with  LOG_PATH2.open('a') as outfile, LOG_PATH.open() as infile:
        outfile.write(infile.read())
    os.remove(LOG_PATH)
    
    
    
if __name__ == "__main__":
    main()

