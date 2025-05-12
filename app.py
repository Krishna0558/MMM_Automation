import re
import streamlit as st
import pandas as pd
import io
import numpy as np
import plotly.express as px
from robyn import Robyn
from datetime import datetime
import subprocess
import tempfile
import os
import time
import json
from Missing_Imputation import handle_missing_values_ui
from Outliers import detect_outliers_ui
from Visualizations import show_visualizations
from Data_Insights import show_data_insights

# Initialize session state
if 'current_step' not in st.session_state:
    st.session_state.current_step = 1
if 'final_df' not in st.session_state:
    st.session_state.final_df = None
if 'transformed' not in st.session_state:
    st.session_state.transformed = False
if 'cleaned_df' not in st.session_state:
    st.session_state.cleaned_df = None
if 'outliers_treated_df' not in st.session_state:
    st.session_state.outliers_treated_df = None
if 'insights_df' not in st.session_state:
    st.session_state.insights_df = None

def pivot_data(df, date_col, pivot_var, paid_organic_vars, target_var, time_granularity, control_vars):
    df[date_col] = pd.to_datetime(df[date_col])
    if time_granularity == "Weekly":
        df['Time_Period'] = df[date_col] - pd.to_timedelta(df[date_col].dt.weekday, unit='D')
    elif time_granularity == "Monthly":
        df['Time_Period'] = df[date_col].dt.to_period("M").apply(lambda r: r.start_time)
    else:
        df['Time_Period'] = df[date_col].dt.floor('D')
    df['Time_Period'] = pd.to_datetime(df['Time_Period']).dt.date

    numeric_cols = paid_organic_vars + [target_var] + control_vars
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    pivot_grouped = df.groupby(['Time_Period', pivot_var])[paid_organic_vars].sum().reset_index()
    pivoted = pivot_grouped.pivot(index='Time_Period', columns=pivot_var, values=paid_organic_vars)
    pivoted.columns = [f"{metric}_{cat}" for metric, cat in pivoted.columns]
    pivoted = pivoted.reset_index()

    exclude_cols = [date_col, pivot_var] + paid_organic_vars
    control_vars_df = df[['Time_Period'] + control_vars].groupby('Time_Period').sum().reset_index()
    target_var_df = df[['Time_Period', target_var]].groupby('Time_Period').sum().reset_index()

    final_df = pd.merge(pivoted, control_vars_df, on='Time_Period', how='left')
    final_df = pd.merge(final_df, target_var_df, on='Time_Period', how='left')

    return final_df.rename(columns={"Time_Period": "Date"})

def clean_column_names(df):
    def make_valid_name(name):
        name = str(name).strip()
        name = re.sub(r'[^a-zA-Z0-9_]', '_', name)
        name = re.sub(r'_+', '_', name)
        # name = re.sub(r'_+', '_', name)
        if name and name[0].isdigit():
            name = f"var_{name}"
        return name
    df.columns = [make_valid_name(col) for col in df.columns]
    return df

# Main App
st.title("📊 MMM Automation UI")
st.header("Step 1: Upload and Transform Your Data")

uploaded_file = st.file_uploader("Upload a CSV or Excel file", type=['csv', 'xlsx'])


if st.session_state.current_step == 1 and uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('csv') else pd.read_excel(uploaded_file)

        st.success("✅ File uploaded successfully!")
        st.subheader("Preview of Uploaded Data")
        st.dataframe(df.head())
        st.subheader("Enter Project Details")

        # 1️⃣ Input for Username
        username = st.text_input("Enter your name")

        # 2️⃣ Auto-filled Project Name (from file name)
        project_name = uploaded_file.name

        # 3️⃣ Create a DataFrame with these two columns
        if username:
            preview_df = pd.DataFrame({
                "User Name": [username],
                "Project Name": [project_name]
            })
            st.subheader("Project Details Preview")
            st.dataframe(preview_df)
        else:
            st.info("Please enter your name to see the project details preview.")

        st.subheader("Step 2: Select Variables")

        datetime_columns = [col for col in df.columns if pd.to_datetime(df[col], errors='coerce').notna().sum() > 0]
        if not datetime_columns:
            st.error("❌ No valid date columns found!")
            st.stop()

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            date_col = st.selectbox("Date Column", datetime_columns)
            try:
                sample_date = pd.to_datetime(df[date_col].dropna().astype(str).iloc[0], errors='coerce')
                if pd.notna(sample_date):
                    valid_date = True
                else:
                    st.error("❌ Date column cannot be parsed")
                    valid_date = False
            except Exception:
                st.error("❌ Unable to parse date column")
                valid_date = False

        with col2:
            target_var = st.selectbox("Target Variable", df.columns)
            valid_target = pd.api.types.is_numeric_dtype(df[target_var])
            if not valid_target:
                st.error("❌ Target must be numeric")

        with col3:
            pivot_var = st.selectbox("Pivot Variable", df.columns)
            valid_pivot = df[pivot_var].dtype in ['object', 'category']
            if not valid_pivot:
                st.error("❌ Pivot variable must be categorical")

        with col4:
            time_base = st.selectbox("Time Granularity", ["Daily", "Weekly", "Monthly"])

        paid_organic_vars = st.multiselect(
            "Paid & Organic Variable(s)",
            [col for col in df.columns if col not in [target_var, pivot_var, date_col]]
        )

        invalid_paid_cols = [col for col in paid_organic_vars if not pd.api.types.is_numeric_dtype(df[col])]
        if invalid_paid_cols:
            st.error(f"❌ Non-numeric variables: {', '.join(invalid_paid_cols)}")
            valid_paid = False
        else:
            valid_paid = True

        control_vars = st.multiselect(
            "Control Variable(s)",
            [col for col in df.columns if col not in [target_var, pivot_var, date_col] + paid_organic_vars]
        )

        invalid_control_cols = [col for col in control_vars if not pd.api.types.is_numeric_dtype(df[col])]
        if invalid_control_cols:
            st.error(f"❌ Non-numeric variables: {', '.join(invalid_control_cols)}")
            valid_control = False
        else:
            valid_control = True

        if valid_date and valid_target and valid_pivot and valid_paid:
            if st.button("🔄 Transform Data"):
                st.session_state["target_var"] = target_var
                st.session_state["time_base"] = time_base
                final_df = pivot_data(df, date_col, pivot_var, paid_organic_vars, target_var, time_base, control_vars)
                st.session_state.final_df = final_df
                st.session_state.transformed = True

                st.success("✅ Data pivoted successfully!")
                st.write(f"The dataset contains **{final_df.shape[0]}** rows and **{final_df.shape[1]}** columns.")
                st.dataframe(final_df.head())
                row_count = len(final_df)
                if time_base == "Weekly" and not (104 <= row_count <= 156):
                    st.error("⚠️ Insufficient Data: Weekly MMM requires 2 to 2.5 years of data, equating to 104–156 weeks.")
                elif time_base == "Monthly" and not (36 <= row_count <= 48):
                    st.error("⚠️ Insufficient Data: Monthly MMM requires 3 to 4 years of data (36–48 monthly records).")
                elif time_base == "Daily" and not (365 <= row_count <= 450):
                    st.error("⚠️ Insufficient Data: Daily MMM requires 1 to 1.25 years of data (365–450 daily records).")

        if st.session_state.transformed:
            if st.button("➡️ Continue to Missing Value Treatment"):
                st.session_state.current_step = 2
                st.rerun()

    except Exception as e:
        st.error(f"❌ Error: {str(e)}")

elif st.session_state.current_step == 2 and st.session_state.final_df is not None:
    cleaned = handle_missing_values_ui(st.session_state.final_df)
    if cleaned is not None:
        st.session_state.cleaned_df = cleaned

    col1, col2 = st.columns(2)
    with col1:
        if st.button("⬅️ Back to Data Transformation"):
            st.session_state.current_step -= 1
            st.rerun()
    with col2:
        if st.session_state.cleaned_df is not None and st.button("➡️ Proceed to Outlier Detection"):
            st.session_state.current_step = 3
            st.rerun()

elif st.session_state.current_step == 3 and st.session_state.cleaned_df is not None:
    st.session_state.outliers_treated_df = detect_outliers_ui(st.session_state.cleaned_df)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("⬅️ Back to Missing Values"):
            st.session_state.current_step -= 1
            st.rerun()
    with col2:
        if st.session_state.outliers_treated_df is not None and st.button("➡️ Continue to Visualizations"):
            st.session_state.current_step = 4
            st.rerun()

elif st.session_state.current_step == 4:
    st.write(f"The dataset contains **{st.session_state.outliers_treated_df.shape[0]}** rows and **{st.session_state.outliers_treated_df.shape[1]}** columns.")
    st.header("Final Processed Data")
    st.dataframe(st.session_state.outliers_treated_df.head())

    show_visualizations(
        df=st.session_state.outliers_treated_df,
        target_var=st.session_state.get("target_var", None)
    )

    col1, col2 = st.columns(2)
    with col1:
        if st.button("⬅️ Back to Outlier Treatment"):
            st.session_state.current_step -= 1
            st.rerun()
    with col2:
        if st.button("➡️ Proceed to Data Insights"):
            st.session_state.current_step = 5
            st.rerun()

elif st.session_state.current_step == 5:
    st.header("Advanced Data Insights")
    st.dataframe(st.session_state.outliers_treated_df.head())

    num_cols = st.session_state.outliers_treated_df.select_dtypes(include=[np.number]).columns.tolist()
    spend_var = st.multiselect("Select spend variables (for ROI analysis)", [None] + num_cols)
    reach_var = st.selectbox("Select reach variable (for time-lag analysis)", [None] + num_cols)
    revenue_var = st.selectbox("Select revenue variable (for ROI analysis)", [None] + num_cols)
    frequency_var = st.selectbox("Select frequency variable (for frequency cap analysis)", [None] + num_cols)

    st.session_state.insights_df = show_data_insights(
        df=st.session_state.outliers_treated_df,
        target_var=st.session_state.get("target_var"),
        time_base=st.session_state.get("time_base"),
        spend_var=spend_var,
        reach_var=reach_var,
        revenue_var=revenue_var,
        frequency_var=frequency_var
    )

    if st.button("⬅️ Back to Visualizations"):
        st.session_state.current_step -= 1
        st.rerun()
    if st.button("➡️ Proceed to Modeling"):
        st.session_state.current_step = 6
        st.rerun()

elif st.session_state.current_step == 6:
    st.header("📈 Run Marketing Mix Model")

    if st.session_state.insights_df is not None:
        st.subheader("Configure Modeling Parameters")

        if 'rscript_path' not in st.session_state:
            st.session_state.rscript_path = r"C:\Program Files\R\R-4.4.2\bin\Rscript.exe"
        if 'robyn_script_path' not in st.session_state:
            st.session_state.robyn_script_path = r"C:\Users\MM3815\Downloads\data_cleaning_app\run_robyn.R"

        if not os.path.exists(st.session_state.rscript_path):
            st.error(f"Rscript not found at: {st.session_state.rscript_path}")
            st.stop()
        if not os.path.exists(st.session_state.robyn_script_path):
            st.error(f"Robyn script not found at: {st.session_state.robyn_script_path}")
            st.stop()

        with st.expander("Advanced Configuration"):
            st.session_state.rscript_path = st.text_input(
                "Rscript Path", value=st.session_state.rscript_path, help="Path to Rscript executable"
            )
            st.session_state.robyn_script_path = st.text_input(
                "Robyn Script Path", value=st.session_state.robyn_script_path, help="Path to run_robyn.R script"
            )

        all_vars = st.session_state.insights_df.select_dtypes(include=[np.number]).columns.tolist()
        date_var = "Date" if "Date" in st.session_state.insights_df.columns else None
        target_var = st.session_state.get("target_var", "Conversions")
        media_vars_options = [v for v in all_vars if v != target_var and v != date_var]

        col1, col2 = st.columns(2)
        with col1:
            media_vars = st.multiselect(
                "Media Variables (spend/impressions)",
                media_vars_options,
                default=media_vars_options[:min(5, len(media_vars_options))],
                help="Select variables representing media spend or impressions"
            )
            adstock = st.selectbox(
                "Adstock Transformation",
                ["geometric", "weibull"],
                index=0,
                help="Choose adstock decay type for media effects"
            )
            iterations = st.number_input("Number of Iterations", min_value=100, max_value=2000, value=500, step=100)
            trials = st.number_input("Number of Trials", min_value=1, max_value=5, value=2, step=1)
        with col2:
            control_vars = st.multiselect(
                "Control Variables (optional)",
                [v for v in all_vars if v not in media_vars and v != target_var and v != date_var],
                help="Select variables to control for external factors"
            )

        n_rows = len(st.session_state.insights_df)
        n_ind_vars = len(media_vars) + len(control_vars)
        recommended_rows = n_ind_vars * 10
        st.write(
            f"Dataset: {n_rows} rows, {n_ind_vars} independent variables (Media: {len(media_vars)}, Control: {len(control_vars)})")
        if n_ind_vars > 0 and n_rows < recommended_rows:
            st.error(
                f"❌ Insufficient data: {n_rows} rows with {n_ind_vars} independent variables. "
                f"Robyn requires at least {recommended_rows} rows (10:1 ratio)."
            )
            max_vars = n_rows // 10
            if max_vars == 0:
                st.error("Dataset is too small to model with any variables.")
                st.button("📂 Re-upload Data", on_click=lambda: st.session_state.update(current_step=1))
                st.stop()
            reduce_vars = st.checkbox(f"Reduce to {max_vars} variables", value=True)
            if reduce_vars:
                st.info(f"Limiting to {max_vars} variables.")
                total_vars = media_vars + control_vars
                total_vars = total_vars[:max_vars]
                media_vars = [v for v in total_vars if v in media_vars]
                control_vars = [v for v in total_vars if v in control_vars]
                st.write("Selected variables:", media_vars + control_vars)
            else:
                st.button("📂 Re-upload Data", on_click=lambda: st.session_state.update(current_step=1))
                st.stop()

        st.markdown("""
        **System Requirements:**
        - Expected runtime: 5–15 minutes
        """)

        if not media_vars:
            st.warning("Please select at least one media variable.")
        elif st.button("🚀 Run Robyn Model"):
            with st.spinner("Running MMM modeling (may take 5–15 minutes)..."):
                try:
                    # Create output directory and save JSON
                    output_dir = "robyn_output"
                    os.makedirs(output_dir, exist_ok=True)
                    cleaned_df = clean_column_names(st.session_state.insights_df.copy())
                    old_to_new_names = dict(zip(st.session_state.insights_df.columns, cleaned_df.columns))
                    target_var = old_to_new_names.get(target_var, target_var)
                    media_vars = [old_to_new_names.get(var, var) for var in media_vars]
                    control_vars = [old_to_new_names.get(var, var) for var in control_vars]
                    date_var = old_to_new_names.get(date_var, date_var) if date_var else None
                    config = {
                        "data_path": "",  # Will be updated after tempfile creation
                        "dep_var": target_var,
                        "media_vars": ",".join(media_vars),
                        "control_vars": ",".join(control_vars) if control_vars else "none",
                        "date_var": date_var if date_var else "none",
                        "adstock_type": adstock,
                        "iterations": str(int(iterations)),
                        "trials": str(int(trials))
                    }

                    # Clean column names and save temporary CSV

                    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp_file:
                        cleaned_df.to_csv(tmp_file.name, index=False)
                        data_path = tmp_file.name
                        config["data_path"] = data_path  # Update config with data_path

                    # Save JSON
                    config_path = os.path.join(output_dir, "config.json")
                    with open(config_path, "w") as f:
                        json.dump(config, f)
                    # st.write(f"Saved configuration to {config_path}")

                    all_experiments_path = os.path.join(output_dir, "all_experiments.json")

                    # Check if file exists → load existing data, else create new dict
                    if os.path.exists(all_experiments_path):
                        with open(all_experiments_path, "r") as f:
                            experiments = json.load(f)
                    else:
                        experiments = {}

                    exp_numbers = [int(k.split("_")[1]) for k in experiments.keys() if k.startswith("experiment_")]
                    next_exp_num = max(exp_numbers) + 1 if exp_numbers else 1
                    exp_key = f"experiment_{next_exp_num}"

                    exp = {
                        "Spend_variables": ",".join(media_vars),
                        "Control_variables": ",".join(control_vars) if control_vars else "none",
                        "Adstock": adstock,
                        "Iterations": str(int(iterations)),
                        "Trials": str(int(trials)),
                        "Timestamp": datetime.now().isoformat()
                    }


                    experiments[exp_key] = exp

                    # Save back to JSON
                    with open(all_experiments_path, "w") as f:
                        json.dump(experiments, f, indent=4)

                    # Construct and run command
                    cmd = [
                        st.session_state.rscript_path,
                        st.session_state.robyn_script_path,
                        data_path,
                        target_var,
                        ",".join(media_vars),
                        ",".join(control_vars) if control_vars else "none",
                        date_var if date_var else "none",
                        adstock,
                        str(int(iterations)),
                        str(int(trials))
                    ]

                    # st.write("Running command:", " ".join(cmd))

                    timeout_minutes = 20
                    start_time = time.time()
                    process = subprocess.Popen(
                        cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                        encoding='utf-8'
                    )

                    with st.expander("Show modeling progress", expanded=True):
                        progress_bar = st.progress(0)
                        output_container = st.empty()
                        full_output = ""
                        stages = ["Starting", "Installing packages", "Running model", "Generating outputs"]
                        stage_progress = 0
                        # st.write("first")

                        while True:
                            elapsed_time = (time.time() - start_time) / 60
                            if elapsed_time > timeout_minutes:
                                process.terminate()
                                raise TimeoutError(
                                    f"Modeling timed out after {timeout_minutes} minutes. Try reducing media variables or iterations.")

                            try:
                                output = process.stdout.readline()
                                if output == '' and process.poll() is not None:
                                    break
                                if output:
                                    full_output += output
                                    output_container.code(full_output[-2000:])

                                    if "installing" in output.lower():
                                        stage_progress = max(stage_progress, 1)
                                    elif "robyn_run started" in output.lower():
                                        stage_progress = max(stage_progress, 2)
                                    elif "robyn_outputs" in output.lower():
                                        stage_progress = max(stage_progress, 3)
                                    progress_bar.progress((stage_progress + 1) / len(stages))
                            except UnicodeDecodeError:
                                continue

                    with open("robyn_log.txt", "w", encoding='utf-8') as f:
                        f.write(full_output)
                    st.write("Debug log saved to robyn_log.txt")
                    st.subheader("Data Preview and Correlations")
                    st.write("Summary Statistics:", cleaned_df[media_vars + [target_var]].describe())
                    st.write("Correlations with Target:",
                             cleaned_df[media_vars + [target_var]].corr()[target_var])
                    with open("robyn_log.txt", "w", encoding='utf-8') as f:
                        f.write(full_output)
                    st.write(process.returncode)

                    if process.returncode == 0:
                        st.success("🎉 Modeling completed successfully!")
                        progress_bar.progress(1.0)

                        output_dir = os.path.join(os.path.dirname(st.session_state.robyn_script_path), "robyn_output")
                        plot_path = os.path.join(output_dir, "plots")
                        model_metrics_path = os.path.join(output_dir, "model_metrics.json")
                        best_model_metrics_path = os.path.join(output_dir, "Best_Models")

                        if os.path.exists(plot_path):
                            st.subheader("Model Visualizations")
                            plot_files = sorted([f for f in os.listdir(plot_path) if f.endswith(".png")])
                            for plot_file in plot_files:
                                st.write(f"**{plot_file}**")
                                st.image(os.path.join(plot_path, plot_file))

                        summary_path = os.path.join(output_dir, "summary.txt")
                        if os.path.exists(summary_path):
                            with open(summary_path, "r", encoding='utf-8') as f:
                                st.subheader("Model Summary")
                                st.text(f.read())

                        corr_path = os.path.join(output_dir, "correlations.txt")
                        if os.path.exists(corr_path):
                            with open(corr_path, "r", encoding='utf-8') as f:
                                st.write("Variable correlations with target:", f.read().splitlines())
                        input_path = os.path.join(output_dir, "inputs.txt")
                        if os.path.exists(input_path):
                            with open(input_path, "r", encoding='utf-8') as f:
                                st.write("Inputs used:", f.read().splitlines())
                        hyper_path = os.path.join(output_dir, "hyperparameters.txt")
                        if os.path.exists(hyper_path):
                            with open(hyper_path, "r", encoding='utf-8') as f:
                                st.write("Hyperparameters used:", f.read().splitlines())
                    else:

                        error_msg = process.stderr.read()
                        st.error(f"Modeling failed: {error_msg}")
                        if "not converged" in error_msg.lower():
                            st.warning(
                                "Model did not converge (DECOMP.RSSD or NRMSE issues). Try wider hyperparameter ranges, more iterations, or different media variables.")
                        elif "coefficient = 0" in error_msg.lower():
                            st.warning(
                                "Zero coefficients detected. Check robyn_output/correlations.txt for weak predictors.")
                        elif "argument is of length zero" in error_msg.lower():
                            st.warning(
                                "Generic error, likely due to plotting or empty results. Check robyn_log.txt for details.")
                        st.info("""
                                            **Troubleshooting Tips:**
                                            1. Use R 4.4.2 (confirmed working locally).
                                            2. Run in R:
                                               ```R
                                               install.packages(c('Robyn', 'dplyr', 'jsonlite', 'ggplot2', 'doSNOW'), dependencies=TRUE)""")

                except Exception as e:
                    st.error(f"Unexpected error: {str(e)}")
                finally:
                    if os.path.exists(data_path):
                        os.unlink(data_path)

        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("⬅️ Back to Data Insights"):
                st.session_state.current_step -= 1
                st.rerun()
        with col2:
            if st.button("🔄 Restart Modeling"):
                st.session_state.current_step = 6
                st.rerun()
        with col3:
            if st.button("🔄 History"):
                st.session_state.current_step = 7
                st.rerun()

elif st.session_state.current_step == 7:
    st.header("📜 Modeling History")

    output_dir = os.path.join(os.path.dirname(st.session_state.robyn_script_path), "robyn_output")
    history_path = os.path.join(output_dir, "all_experiments.json")

    if os.path.exists(history_path):
        with open(history_path, "r", encoding="utf-8") as f:
            experiments = json.load(f)

        if experiments:
            df_history = pd.DataFrame.from_dict(experiments, orient="index").reset_index()
            df_history.rename(columns={"index": "Experiment"}, inplace=True)
            st.dataframe(df_history)
        else:
            st.info("No experiments found in history.")
    else:
        st.warning("History file (all_experiments.json) not found.")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("⬅️ Back to Modeling"):
            st.session_state.current_step = 6
            st.rerun()
    with col2:
        if st.button("🏠 Back to Home"):
            st.session_state.current_step = 1
            st.rerun()

            # Always show available PNG outputs (even if modeling failed)
            # output_dir = os.path.join(os.path.dirname(st.session_state.robyn_script_path), "robyn_output")
            # plot_path = os.path.join(output_dir, "plots")
            #
            # if os.path.exists(plot_path):
            #     st.subheader("🖼️ Available Robyn Output Images")
            #     plot_files = sorted([f for f in os.listdir(plot_path) if f.endswith(".png")])
            #     for plot_file in plot_files:
            #         st.image(os.path.join(plot_path, plot_file), caption=plot_file, use_column_width=True)
