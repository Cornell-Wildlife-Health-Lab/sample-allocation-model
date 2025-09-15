
python3 "input_processing.py"
exit_status=$?  # Capture exit code
echo "Exit status for input processing: $exit_status"

# Check if the first exit code is zero (successful)
if [ $exit_status -eq 0 ]; then  # Use == for comparison
  echo "Input processing ran successfully. Run model script."


  # jump into this scripts’s folder
  cd "$(dirname "$0")"

  # now call your python entry‑point
  python3 ../main.py

  exit_status=$?  # Capture exit code
  echo "Exit status for model/output: $exit_status"
else
  echo "Input processing failed. Exit status: $exit_status"
  exit $exit_status
fi




# Check if the model code ran
if [ $exit_status -eq 0 ]; then  # Use == for comparison
  echo "Model processing ran successfully. Finishing post-processing."

  # now call your python entry‑point
  python3 ../post_processing.py

  exit_status=$?  # Capture exit code
  echo "Exit status for post-processing: $exit_status"
else
  echo "Post-processing failed. Exit status: $exit_status"
  exit $exit_status
fi

# Exit with the final exit status
if [ $exit_status -eq 0 ]; then  # Consistent comparison
  echo "Model and output processing script ran successfully. Exit status: $exit_status"
else
  echo "Model and output processing processing script failed. Exit status: $exit_status"
fi


exit $exit_status