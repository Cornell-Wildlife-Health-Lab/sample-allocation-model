#!/usr/bin/env sh
#
# run_scripts.sh — cd into src/, make results/, then call main2.py

python3 "input_processing2.py"
exit_status=$?  # Capture exit code
echo "Exit status for input processing: $exit_status"

# Check if the first exit code is zero (successful)
# TODO: check this??? 
if [ $exit_status -eq 0 ]; then  # Use == for comparison
  echo "Input processing ran successfully. Run model script."

    #set -euo pipefail

    # jump into this scripts’s folder
    cd "$(dirname "$0")"

    # ensure the project root has a place to put outputs
    #mkdir -p ../results

    # now call your python entry‑point
    python3 ../main2.py

    exit_status=$?  # Capture exit code
    echo "Exit status for model/output: $exit_status"

else
  echo "Input processing failed. Exit status: $exit_status"
  exit $exit_status
fi


# Exit with the final exit status
if [ $exit_status -eq 0 ]; then  # Consistent comparison
  echo "Model and output processing script ran successfully. Exit status: $exit_status"
else
  echo "Model and output processing processing script failed. Exit status: $exit_status"
fi


exit $exit_status