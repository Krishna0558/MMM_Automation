import streamlit as st
import pandas as pd
import io
from Missing_Imputation import handle_missing_values_ui
from Outliers import detect_outliers_ui
from Visualizations import show_visualizations

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
# Comment--testing

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
    numeric_df = df.drop(columns=exclude_cols, errors='ignore').select_dtypes(include='number')
    # others_summed = df[['Time_Period']].join(numeric_df).groupby('Time_Period').sum().reset_index()
    control_vars_df = df[['Time_Period'] + control_vars].groupby('Time_Period').sum().reset_index()
    #added
    target_var_df = df[['Time_Period', target_var]].groupby('Time_Period').sum().reset_index()

    # final_df = pd.merge(pivoted, others_summed, on='Time_Period', how='left')
    final_df = pd.merge(pivoted, control_vars_df, on='Time_Period', how='left')
    #added
    final_df = pd.merge(final_df, target_var_df, on='Time_Period', how='left')


    return final_df.rename(columns={"Time_Period": "Date"})


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
                final_df = pivot_data(df, date_col, pivot_var, paid_organic_vars, target_var, time_base, control_vars)
                st.session_state.final_df = final_df
                st.session_state.transformed = True

                st.success("✅ Data pivoted successfully!")
                st.write(f"The dataset contains **{final_df.shape[0]}** rows and **{final_df.shape[1]}** columns.")
                st.dataframe(final_df.head())
                row_count = len(final_df)
                if time_base == "Weekly" and not (104 <= row_count <= 156):
                    st.error("⚠️ Insufficient Data: Weekly MMM requires 2 to 2.5 years of data, equating to 104–156 weeks.")
                    # st.radio("Please choose an option:", ["Continue to the next step", "Re-upload the file"])
                elif time_base == "Monthly" and not (36 <= row_count <= 48):
                    st.error("⚠️ Insufficient Data: Monthly MMM requires 3 to 4 years of data (36–48 monthly records).")
                    # st.radio("Please choose an option:", ["Continue to the next step", "Re-upload the file"])
                elif time_base == "Daily" and not (365 <= row_count <= 450):
                    st.error("⚠️ Insufficient Data: Daily MMM requires 1 to 1.25 years of data (365–450 daily records).")
                    # st.radio("Please choose an option:", ["Continue to the next step", "Re-upload the file"])

        if st.session_state.transformed:
            if st.button("➡️ Continue to Missing Value Treatment"):#chnage this to whole code
                st.session_state.current_step = 2
                st.rerun()
        # if st.session_state.transformed:
        #     # Define the radio button and capture the selected option
        #     user_choice = st.radio("Please choose an option:", ["Continue to the next step", "Re-upload the file"])
        #
        #     # If the user chooses to continue, move to the next step (Missing Value Treatment)
        #     if user_choice == "Continue to the next step":
        #         st.session_state.current_step = 2
        #         st.rerun()


    except Exception as e:
        st.error(f"❌ Error: {str(e)}")

# elif st.session_state.current_step == 2 and st.session_state.final_df is not None:
#     st.session_state.cleaned_df = handle_missing_values_ui(st.session_state.final_df)
#     if st.session_state.cleaned_df is not None and st.button("➡️ Proceed to Outlier Detection"):
#         st.session_state.current_step = 3
#         st.rerun()
#
# elif st.session_state.current_step == 3 and st.session_state.cleaned_df is not None:
#     st.session_state.outliers_treated_df = detect_outliers_ui(st.session_state.cleaned_df)
#     if st.session_state.outliers_treated_df is not None and st.button("➡️ Complete Pipeline"):
#         st.session_state.current_step = 4
#         st.balloons()
#         st.success("🎉 Pipeline completed!")
#
# elif st.session_state.current_step == 4:
#     st.header("Final Processed Data")
#     st.dataframe(st.session_state.outliers_treated_df.head())

# elif st.session_state.current_step == 2 and st.session_state.final_df is not None:
#     st.session_state.cleaned_df = handle_missing_values_ui(st.session_state.final_df)
# added for the not getting the correct data
elif st.session_state.current_step == 2 and st.session_state.final_df is not None:
    cleaned = handle_missing_values_ui(st.session_state.final_df)
    if cleaned is not None:
        st.session_state.cleaned_df = cleaned

    # Add navigation buttons here
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

    # Add navigation buttons here
    col1, col2 = st.columns(2)
    with col1:
        if st.button("⬅️ Back to Missing Values"):
            st.session_state.current_step -= 1
            st.rerun()
    with col2:
        if st.session_state.outliers_treated_df is not None and st.button("➡️ Continue to step-5"):
            st.session_state.current_step = 4
            # st.balloons()
            # st.success("🎉 Pipeline completed!")
            st.rerun()

# elif st.session_state.current_step == 4:
#     st.header("Final Processed Data")
#     st.dataframe(st.session_state.outliers_treated_df.head())
#
#     # Add back button for final step
#     if st.button("⬅️ Back to Outlier Treatment"):
#         st.session_state.current_step -= 1
#         st.rerun()

elif st.session_state.current_step == 4:
    st.write(f"The dataset contains **{st.session_state.outliers_treated_df.shape[0]}** rows and **{st.session_state.outliers_treated_df.shape[1]}** columns.")
    st.header("Final Processed Data")
    st.dataframe(st.session_state.outliers_treated_df.head())

    # Import and call the visualization function


    show_visualizations(
        df=st.session_state.outliers_treated_df,
        target_var=st.session_state.get("target_var", None)
    )

    # Add back button
    if st.button("⬅️ Back to Outlier Treatment"):
        st.session_state.current_step -= 1
        st.rerun()