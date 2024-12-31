# Pipeline Evaluation

This directory contains the evaluation scripts for computing the challan rate as well as the metrics for triple riding and no-helmet violations, as reported in the associated paper.

Run the base association model and store the outputs for each video, execute the script `run_eval_pipeline.py`.

The outputs for all videos in the test set can be found in the following directory:  
[Precomputed Outputs](https://drive.google.com/drive/folders/1NWt42_Sr7jxuAVZq2nQOm0LiZmE_3nYz).

Within this directory, the folder `output_files/rm_preds` contains the final precomputed outputs for every video.

For all subsequent sections, you can access the relevant precomputed files in the `output_files/` directory.

## No-Helmet Violation Metrics

To compute the precision, recall, and F1-score for detecting helmet/no-helmet violations:

1. Run `det_hnh_frame.py` to perform inference using the helmet/no-helmet detector. 
2. Run `hnh_eval.py` to compute the evaluation metrics (precision, recall, F1-score).

## Triple Riding Violation Metrics

To compute the precision, recall, and F1-score for detecting triple riding violations:

1. Run `clf_infer.py` for inference using the helmet/no-helmet detector.  
   - For multi-GPU inference, use `clf_infer_multi.py`.  
2. Run `tr_eval.py` to compute the evaluation metrics (precision, recall, F1-score).

## End-to-End Pipeline: Challan Rate Calculation

To compute the challan rate for the full pipeline (both automatic and human-in-the-loop):

1. Run `challan_rate_calc.py`.

---

**Note:** Before running any scripts, update the directory paths specified at the beginning of each file to reflect the actual locations of your datasets and outputs.
