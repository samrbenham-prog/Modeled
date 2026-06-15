import streamlit as st
import pandas as pd
import plotly.express as px

from fitness_model import run_fitness_model
from strava_api import get_strava_activities
from strava_processing import build_weekly_training_data
from strava_auth import (
    build_authorization_url,
    exchange_code_for_token,
)


# ----------------------------
# Constants
# ----------------------------

MIN_VALID_HR_WEEKS = 8
# Weeks used to let the recursive fitness/fatigue model stabilize
# before showing results to the user.
MIN_AVG_HR_COVERAGE = 0.50
# At least 50% of the runs need HR data
HIDDEN_WARMUP_WEEKS = 8
FEEDBACK_FORM_URL = "https://docs.google.com/forms/d/e/1FAIpQLSe41DSa9itRq03XgmYu3l3ruuWjkpp5LQi6S6L5LZc57CeVfg/viewform?usp=publish-editor"
LOGO_FILE = "Modeled_Logo_Beta.png"
PRESENTATION_URL = "https://1drv.ms/p/c/319e282c26fe4f7e/IQApfUOK8YXYQKVeV20dySQmAckZwOjSGmMyZJ6spBSXL3I?e=EM7rWy"

# ----------------------------
# Helper functions
# ----------------------------

def format_pace(decimal_minutes):
    """
    Convert decimal minutes per mile into M:SS pace format.

    Examples:
    7.30 -> 7:18/mi
    7.50 -> 7:30/mi
    6.75 -> 6:45/mi
    """
    if pd.isna(decimal_minutes):
        return "N/A"

    total_seconds = int(round(decimal_minutes * 60))
    minutes = total_seconds // 60
    seconds = total_seconds % 60

    return f"{minutes}:{seconds:02d}/mi"


def show_model_explanation():
    with st.expander("What does this model measure?"):
        st.write(
            "This model ESTIMATES running fitness from weekly training volume, pace, "
            "and heart-rate efficiency. It looks at how much training an athlete did, "
            "how fast they were running, and whether their average running heart rate "
            "was better or worse than expected for that pace."
        )

        st.write(
            "Higher scores mean the athlete appears fitter relative to their own selected data range. "
            "This is not a medical metric, and it is not a guaranteed race predictor. "
            "It works best when the athlete has consistent GPS and heart-rate data."
        )

        st.write(
            "**Mayo miles** are the app's training-volume estimate. Running miles count directly, "
            "and cross-training minutes are converted into mileage by dividing by 10. "
            "For example, 60 minutes of cross-training adds 6 Mayo miles."
        )

        st.write(
            "The 0–100 Fitness Score is not a percentile. It is a scaled score within the selected "
            "visible range. A score of 100 is the highest modeled fitness week in the selected range, "
            "and 0 is the lowest modeled fitness week in that selected range."
        )

        st.write(
            "The first several weeks are hidden from the graph because the model needs time to stabilize. "
            "Those early weeks are still used in the calculations, but they are not displayed."
        )

        st.write(
            "Weeks with manual uploads can still contribute to mileage and training volume, "
            "but if they do not include heart-rate data, the HR-based part of the model becomes less reliable."
        )

def show_privacy_and_data_use():
    with st.expander("Privacy and Data Use"):
        st.write(
            "Modeled uses your authorized Strava activity data only to calculate your personal "
            "training dashboard."
        )

        st.write(
            "The app uses activity details such as activity type, start date, moving time, "
            "distance, average pace, and average heart rate to estimate weekly fitness trends."
        )

        st.write(
            "Your activity data is shown only to you inside your own dashboard. Modeled does not "
            "display your private Strava activity data to other users."
        )

        st.write(
            "Modeled does not sell your Strava data and does not use your Strava activity data "
            "to train AI models."
        )

        st.write(
            "Because this is an early testing version, results should be treated as experimental. "
            "The model is intended to help athletes better understand training trends, not to provide "
            "medical advice or guaranteed race predictions."
        )

def show_feedback_form():
    st.divider()
    st.subheader("Give Feedback")

    st.write(
        "This model is still being tested. If you have a minute, please share what worked, "
        "what felt confusing, and what you would want changed."
    )

    st.link_button("Open Feedback Form", "https://docs.google.com/forms/d/e/1FAIpQLSe41DSa9itRq03XgmYu3l3ruuWjkpp5LQi6S6L5LZc57CeVfg/viewform?usp=publish-editor")

def show_project_presentation():
    st.divider()
    st.subheader("Learn More About the Project")

    st.write(
        "Want to understand the model, the goal of the app, and how it compares to platforms "
        "like Strava and Garmin? View the project presentation below."
    )

    st.link_button("View Project Presentation", PRESENTATION_URL)

def show_data_quality_summary(df):
    st.subheader("Data Quality Summary")

    if df is None or len(df) == 0:
        st.error("No weekly training data is available.")
        return

    total_weeks = len(df)

    if "totalrunningmiles" in df.columns:
        running_weeks = int((df["totalrunningmiles"] > 0).sum())
        total_running_miles = float(df["totalrunningmiles"].sum())
    else:
        running_weeks = 0
        total_running_miles = 0.0

    if "totaltimext" in df.columns:
        total_xt_minutes = float(df["totaltimext"].sum())
    else:
        total_xt_minutes = 0.0

    if "avgrunhr" in df.columns:
        weeks_with_hr = int(df["avgrunhr"].notna().sum())
    else:
        weeks_with_hr = 0

    weeks_missing_hr = max(running_weeks - weeks_with_hr, 0)

    if "hr_coverage" in df.columns and running_weeks > 0:
        avg_hr_coverage = float(
            df.loc[df["totalrunningmiles"] > 0, "hr_coverage"].mean()
        )
    else:
        avg_hr_coverage = weeks_with_hr / running_weeks if running_weeks > 0 else 0

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Weekly Rows", total_weeks)
        st.metric("Running Weeks", running_weeks)

    with col2:
        st.metric("Weeks With HR", weeks_with_hr)
        st.metric("Weeks Missing HR", weeks_missing_hr)

    with col3:
        st.metric("Avg HR Coverage", f"{avg_hr_coverage * 100:.1f}%")
        st.metric("Total Running Miles", f"{total_running_miles:.1f}")

    st.metric("Total Cross-Training Minutes", f"{total_xt_minutes:.0f}")

    if running_weeks == 0:
        st.error(
            "No running weeks were found. The model needs running data to work."
        )

    elif weeks_with_hr < MIN_VALID_HR_WEEKS:
        st.error(
            f"This athlete does not have enough heart-rate data to run the HR-based fitness model. "
            f"The model needs at least {MIN_VALID_HR_WEEKS} weeks with both pace and average running HR. "
            f"This athlete currently has {weeks_with_hr} valid HR weeks."
        )

    elif avg_hr_coverage < 0.50:
        st.warning(
            "Heart-rate coverage is low. The model can still run, but results may be less reliable "
            "because many running weeks are missing HR data."
        )

    elif avg_hr_coverage < 0.75:
        st.info(
            "Heart-rate coverage is moderate. The model should run, but some weeks may rely more heavily on pace/volume."
        )
    else:
        st.success("Data quality looks strong enough for the HR-based fitness model.")


def validate_model_data(df):
    if df is None or len(df) == 0:
        return False, "No weekly training data is available."

    required_cols = ["week", "avgpace", "avgrunhr", "totalmayomiles"]

    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        return False, (
            "The model is missing required columns: "
            + ", ".join(missing_cols)
        )

    if "totalrunningmiles" in df.columns:
        running_weeks = int((df["totalrunningmiles"] > 0).sum())

        if running_weeks == 0:
            return False, "No running weeks were found."

    else:
        running_weeks = len(df)

    valid_hr_weeks = df.dropna(subset=["avgpace", "avgrunhr"])

    if len(valid_hr_weeks) < MIN_VALID_HR_WEEKS:
        return False, (
            "Modeled could not run because there is not enough heart-rate data. "
            f"The model needs at least {MIN_VALID_HR_WEEKS} weeks with both pace and average running HR. "
            f"This selected range has {len(valid_hr_weeks)} valid HR weeks. "
            "Try selecting a longer date range, syncing more activities with HR data, "
            "or using a chest strap/watch that records heart rate."
        )

    if "hr_coverage" in df.columns and "totalrunningmiles" in df.columns:
        running_df = df[df["totalrunningmiles"] > 0].copy()

        if len(running_df) > 0:
            avg_hr_coverage = float(running_df["hr_coverage"].fillna(0).mean())

            if avg_hr_coverage < MIN_AVG_HR_COVERAGE:
                return False, (
                    "Modeled could not run because heart-rate coverage is too low. "
                    f"The model needs at least {MIN_AVG_HR_COVERAGE * 100:.0f}% average HR coverage "
                    f"across running weeks. This selected range has {avg_hr_coverage * 100:.1f}% HR coverage. "
                    "Manual uploads and activities without HR can still count toward mileage, "
                    "but the HR-based fitness model needs enough heart-rate data to estimate fitness reliably."
                )

    if df["totalmayomiles"].std(ddof=1) == 0:
        return False, (
            "Training volume does not vary enough in the selected range. "
            "Try selecting a longer period with more variation in training."
        )

    return True, ""


# ----------------------------
# App header
# ----------------------------

st.image(LOGO_FILE, width=220)

st.title("Running Fitness Model")

st.write(
    "Connect your Strava account to pull recent activities and estimate weekly running fitness."
)

show_model_explanation()
show_privacy_and_data_use()
show_project_presentation()


# ----------------------------
# Session state setup
# ----------------------------

if "weekly_df" not in st.session_state:
    st.session_state["weekly_df"] = None

if "data_source_name" not in st.session_state:
    st.session_state["data_source_name"] = None


# ----------------------------
# Read Strava redirect info from URL
# ----------------------------

query_params = st.query_params

strava_code = query_params.get("code", None)
strava_scope = query_params.get("scope", None)

default_data_source_index = 1 if strava_code else 0


# ----------------------------
# Data source selection
# ----------------------------

data_source = "Connect Strava"


# ----------------------------
# Automatic Strava OAuth Flow
# ----------------------------

if data_source == "Connect Strava":
    st.subheader("Connect with Strava")

    authorization_url = build_authorization_url()

    st.markdown(
        f"[Click here to connect your Strava account]({authorization_url})"
    )

    st.write(
        "After you authorize Strava, you should be sent back to this app automatically."
    )

    activities_to_pull = st.number_input(
        "Maximum number of Strava activities to pull",
        min_value=200,
        max_value=4000,
        value=1000,
        step=200
    )

    max_pages = int((activities_to_pull + 199) // 200)

    st.caption(
        "Strava returns up to 200 activities per page. "
        f"This setting will pull up to {activities_to_pull:,} activities "
        f"using {max_pages} API request page(s)."
    )

    if strava_code:
        st.success("Strava authorization code received.")

#        if strava_scope:
#            st.write(f"Accepted scope: {strava_scope}")

        if strava_scope is None or "activity:read_all" not in strava_scope:
            st.warning(
                "The token may not have activity:read_all permission. "
                "Make sure you authorized activity access."
            )

        if st.button("Pull Strava Data"):
            token_response = exchange_code_for_token(strava_code)

            if "access_token" not in token_response:
                st.error("Token exchange failed.")
                st.json(token_response)
            else:
                access_token = token_response["access_token"]

                st.success("Strava authorization worked.")

                st.info("Pulling activities from Strava...")

                activities = get_strava_activities(
                    access_token,
                    max_pages=int(max_pages)
                )

                # Keep only the requested maximum number of activities
                activities = activities.head(int(activities_to_pull))

                if activities is None or len(activities) == 0:
                    st.error(
                        "No Strava activities were pulled. "
                        "Try increasing the number of pages or checking Strava permissions."
                    )
                    st.stop()

                st.write(f"Pulled {len(activities)} activities.")

                with st.expander("Raw Strava Activities Preview"):
                    st.dataframe(activities.head())

                st.info("Converting activities into weekly training data...")

                try:
                    df = build_weekly_training_data(activities)
                except Exception as e:
                    st.error("The app could not convert Strava activities into weekly training data.")
                    st.write(str(e))
                    st.stop()

                if df is None or len(df) == 0:
                    st.error(
                        "No weekly training rows were created from the Strava data. "
                        "This usually means no usable running or cross-training activities were found."
                    )
                    st.stop()

                st.session_state["weekly_df"] = df
                st.session_state["data_source_name"] = "Strava"

                st.subheader("Weekly Training Data from Strava")
                st.dataframe(df.head())

                st.write(f"Created {len(df)} weekly rows.")

                st.query_params.clear()

    else:
        st.info(
            "Click the Strava link above, authorize your account, and you should return here automatically."
        )


# ----------------------------
# Model Range + Warm-Up Selection
# ----------------------------

df = st.session_state["weekly_df"]
model_df = None
visible_start_original_week = None
visible_end_original_week = None

if df is not None:
    st.subheader("Choose Model Range and Warm-Up")

    st.write(
        "You can use the full available history, or choose a specific range of weeks. "
        "If you choose a specific range, the app automatically includes hidden warm-up weeks before "
        "the selected start so the first visible week is not distorted by model initialization."
    )

    df = df.copy()
    df = df.sort_values("week").reset_index(drop=True)

    df["original_week"] = df["week"]

    if "week_start" in df.columns:
        df["week_display"] = (
            "Week "
            + df["original_week"].astype(str)
            + " — "
            + df["week_start"].astype(str)
        )
    else:
        df["week_display"] = (
            "Week " + df["original_week"].astype(str)
        )

    range_option = st.radio(
        "How much training history should the model use?",
        ["Use full available history", "Choose start and end week"]
    )

    if range_option == "Use full available history":
        model_df = df.copy()
        
        min_original_week = int(model_df["original_week"].min())
        max_original_week = int(model_df["original_week"].max())

        # The model still uses the first 8 weeks, but the app does not display them.
        visible_start_original_week = min_original_week + HIDDEN_WARMUP_WEEKS
        visible_end_original_week = max_original_week

        st.write("Model will use all available weekly training history.")

        st.caption(
            f"The first {HIDDEN_WARMUP_WEEKS} weeks are used as hidden warm-up weeks "
            "so the fitness/fatigue model has time to stabilize before results are displayed."
        )

    else:
        start_week_display = st.selectbox(
            "Choose the first visible week:",
            df["week_display"].tolist()
        )

        selected_start_week = int(
            df.loc[
                df["week_display"] == start_week_display,
                "original_week"
            ].iloc[0]
        )

        possible_end_df = df[df["original_week"] >= selected_start_week].copy()

        end_week_display = st.selectbox(
            "Choose the last visible week:",
            possible_end_df["week_display"].tolist(),
            index=len(possible_end_df) - 1
        )

        selected_end_week = int(
            possible_end_df.loc[
                possible_end_df["week_display"] == end_week_display,
                "original_week"
            ].iloc[0]
        )

        calculation_start_week = max(
            int(df["original_week"].min()),
            selected_start_week - HIDDEN_WARMUP_WEEKS
        )

        model_df = df[
            (df["original_week"] >= calculation_start_week)
            & (df["original_week"] <= selected_end_week)
        ].copy()

        visible_start_original_week = selected_start_week
        visible_end_original_week = selected_end_week

        st.success(
            f"Model will display Week {selected_start_week} through Week {selected_end_week}."
        )

        st.caption(
            "The app automatically uses a short hidden warm-up period before the selected start week "
            "to make the first visible weeks more stable."
        )

    model_df = model_df.drop(columns=["week_display"], errors="ignore")

    model_df = model_df.reset_index(drop=True)
    model_df["week"] = range(1, len(model_df) + 1)

    show_data_quality_summary(model_df)


# ----------------------------
# Run Model + Dashboard
# ----------------------------

if model_df is not None:
    is_valid, validation_message = validate_model_data(model_df)

    if not is_valid:
        st.error(validation_message)
        st.stop()

    try:
        results, model = run_fitness_model(model_df)
    except ValueError as e:
        st.error(str(e))
        st.stop()
    except Exception as e:
        st.error("The model could not run on this athlete's data.")
        st.write(str(e))
        st.stop()

    if "original_week" not in results.columns and "original_week" in model_df.columns:
        results["original_week"] = model_df["original_week"].values

    visible_results = results[
        (results["original_week"] >= visible_start_original_week)
        & (results["original_week"] <= visible_end_original_week)
    ].copy()

    visible_results = visible_results.reset_index(drop=True)

    if len(visible_results) == 0:
        st.error("No visible model results were created for the selected range.")
        st.stop()

    visible_results["week_number"] = range(1, len(visible_results) + 1)
    visible_results["week_label"] = visible_results["week_number"].apply(
        lambda x: f"Week {x}"
    )

    min_fit = visible_results["z_fitness_index"].min()
    max_fit = visible_results["z_fitness_index"].max()

    if max_fit == min_fit:
        visible_results["fitness_score_100"] = 50
    else:
        visible_results["fitness_score_100"] = (
            (visible_results["z_fitness_index"] - min_fit)
            / (max_fit - min_fit)
        ) * 100

    st.subheader("Modeled Fitness Index")

    chart_data = visible_results.copy()

    chart_data["Fitness Score"] = chart_data["fitness_score_100"].round(1)
    chart_data["Visible Week"] = chart_data["week_number"].astype(int)

    if "week_start" in chart_data.columns:
        chart_data["Calendar Week"] = chart_data["week_start"].astype(str)
    else:
        chart_data["Calendar Week"] = ""

    if "totalrunningmiles" in chart_data.columns:
        chart_data["Running Miles"] = chart_data["totalrunningmiles"].round(1)
    else:
        chart_data["Running Miles"] = 0

    if "totalmayomiles" in chart_data.columns:
        chart_data["Mayo Miles"] = chart_data["totalmayomiles"].round(1)
    else:
        chart_data["Mayo Miles"] = 0

    if "avgrunhr" in chart_data.columns:
        chart_data["Avg Run HR"] = chart_data["avgrunhr"].round(1)
    else:
        chart_data["Avg Run HR"] = None

    if "avgpace" in chart_data.columns:
        chart_data["Avg Pace"] = chart_data["avgpace"].apply(format_pace)
    else:
        chart_data["Avg Pace"] = "N/A"

    fig = px.line(
        chart_data,
        x="Visible Week",
        y="Fitness Score",
        markers=True,
        title="Modeled Fitness Index Over Time",
        custom_data=[
            "Calendar Week",
            "Running Miles",
            "Mayo Miles",
            "Avg Run HR",
            "Avg Pace",
        ],
    )

    fig.update_traces(
        hovertemplate=(
            "<b>Week %{x}</b><br>"
            "Calendar week: %{customdata[0]}<br>"
            "Running miles: %{customdata[1]:.1f}<br>"
            "Mayo miles: %{customdata[2]:.1f}<br>"
            "Avg run HR: %{customdata[3]:.1f}<br>"
            "Avg pace: %{customdata[4]}<br>"
            "Fitness score: %{y:.1f}/100"
            "<extra></extra>"
        )
    )

    fig.update_layout(
        xaxis_title="Week",
        yaxis_title="Fitness Score",
        hovermode="x unified",
    )

    st.plotly_chart(fig, use_container_width=True)

    fitness_series = visible_results.dropna(subset=["fitness_score_100"])

    if len(fitness_series) == 0:
        st.error("No valid fitness scores were created.")
        st.stop()

    latest = fitness_series.iloc[-1]
    peak = fitness_series.loc[
        fitness_series["fitness_score_100"].idxmax()
    ]

    st.subheader("Dashboard Metrics")

    st.metric(
        "Current Fitness Score",
        f"{latest['fitness_score_100']:.1f}"
    )

    st.metric(
        "Peak Fitness Score",
        f"{peak['fitness_score_100']:.1f}"
    )

    st.write(f"Peak occurred in {peak['week_label']}")

    if "week_start" in latest:
        latest_week_date = pd.to_datetime(latest["week_start"]).strftime("%Y-%m-%d")
        st.write(f"Latest calendar week: {latest_week_date}")

    if len(fitness_series) >= 5:
        four_weeks_ago = fitness_series.iloc[-5]

        four_week_change_pct = (
            (
                latest["fitness_score_100"]
                - four_weeks_ago["fitness_score_100"]
            )
            / four_weeks_ago["fitness_score_100"]
        ) * 100

        st.metric(
            "4-Week Fitness Change",
            f"{four_week_change_pct:+.1f}%"
        )
    else:
        st.write("Not enough visible weeks to calculate 4-week fitness change.")

    st.subheader("Latest Reading")

    st.write(f"Visible Week: {latest['week_label']}")

    if "original_week" in latest:
        st.write(f"Original Data Week: Week {int(latest['original_week'])}")

    st.write(f"Fitness Score: {latest['fitness_score_100']:.1f}/100")
    st.write(f"Original Z-Score Fitness Index: {latest['z_fitness_index']:.2f}")

    st.subheader("Full Model Output")

    display_results = visible_results.copy()

    display_results["Visible Week"] = display_results["week_number"].astype(int)

    if "original_week" in display_results.columns:
        display_results["Original Data Week"] = display_results["original_week"].astype(int)

    if "week_start" in display_results.columns:
        display_results["Calendar Week"] = display_results["week_start"]

    if "avgpace" in display_results.columns:
        display_results["Avg Pace Display"] = display_results["avgpace"].apply(format_pace)

    display_results = display_results.drop(
        columns=[
            "week",
            "week_number",
            "week_label",
            "original_week"
        ],
        errors="ignore"
    )

    front_cols = []

    if "Visible Week" in display_results.columns:
        front_cols.append("Visible Week")

    if "Original Data Week" in display_results.columns:
        front_cols.append("Original Data Week")

    if "Calendar Week" in display_results.columns:
        front_cols.append("Calendar Week")

    if "Avg Pace Display" in display_results.columns:
        front_cols.append("Avg Pace Display")

    other_cols = [col for col in display_results.columns if col not in front_cols]

    display_results = display_results[front_cols + other_cols]

    with st.expander("View Full Model Output"):
        st.dataframe(display_results, hide_index=True)

    csv = display_results.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download Results CSV",
        data=csv,
        file_name="fitness_model_results.csv",
        mime="text/csv"
    )
    
    show_feedback_form()
