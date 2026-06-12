import pandas as pd
import numpy as np

def build_weekly_training_data(activities):
    df = pd.DataFrame(activities)

    # Convert dates
    df["start_date"] = pd.to_datetime(df["start_date"])

    # Monday-start week
    df["week_start"] = df["start_date"].dt.to_period("W-SUN").apply(lambda r: r.start_time)

    # Convert Strava units
    df["minutes"] = df["moving_time"] / 60
    df["miles"] = df["distance"] / 1609.34

    # Define activity types
    run_types = ["Run"]
    xt_types = ["Ride", "VirtualRide", "Swim", "Elliptical", "EBikeRide", "Workout"]

    df["is_run"] = df["type"].isin(run_types)
    df["is_xt"] = df["type"].isin(xt_types)

    # Run-only variables
    df["run_minutes"] = np.where(df["is_run"], df["minutes"], 0)
    df["run_miles"] = np.where(df["is_run"], df["miles"], 0)

    # Cross-training only
    df["xt_minutes"] = np.where(df["is_xt"], df["minutes"], 0)

    # HR exists only for some runs
    df["has_run_hr"] = df["is_run"] & df["average_heartrate"].notna()

    # HR-weighted contribution
    df["run_hr_minutes"] = np.where(
        df["has_run_hr"],
        df["minutes"] * df["average_heartrate"],
        0
    )

    # Minutes of running that actually have HR
    df["run_minutes_with_hr"] = np.where(
        df["has_run_hr"],
        df["minutes"],
        0
    )

    weekly = df.groupby("week_start").agg(
        totaltimerunning=("run_minutes", "sum"),
        totalrunningmiles=("run_miles", "sum"),
        totaltimext=("xt_minutes", "sum"),
        run_hr_minutes=("run_hr_minutes", "sum"),
        run_minutes_with_hr=("run_minutes_with_hr", "sum")
    ).reset_index()

    # Average HR only across running minutes that actually have HR
    weekly["avgrunhr"] = (
        weekly["run_hr_minutes"] / weekly["run_minutes_with_hr"]
    )

    # Share of running minutes with HR data
    weekly["hr_coverage"] = (
        weekly["run_minutes_with_hr"] / weekly["totaltimerunning"]
    )

    # Average pace = total running minutes / total running miles
    weekly["avgpace"] = (
        weekly["totaltimerunning"] / weekly["totalrunningmiles"]
    )

    weekly["totaltimeadded"] = (
        weekly["totaltimerunning"] + weekly["totaltimext"]
    )

    weekly["totalmayomiles"] = (
        weekly["totalrunningmiles"] + weekly["totaltimext"] / 10
    )

    # Clean impossible values
    weekly.loc[weekly["totaltimerunning"] == 0, "avgrunhr"] = np.nan
    weekly.loc[weekly["totaltimerunning"] == 0, "hr_coverage"] = 0
    weekly.loc[weekly["totalrunningmiles"] == 0, "avgpace"] = np.nan

    weekly["hr_coverage"] = weekly["hr_coverage"].replace([np.inf, -np.inf], np.nan).fillna(0)

    weekly = weekly.sort_values("week_start").reset_index(drop=True)
    weekly["week"] = range(1, len(weekly) + 1)

    return weekly[[
        "week",
        "week_start",
        "totaltimerunning",
        "totaltimext",
        "totalrunningmiles",
        "avgrunhr",
        "avgpace",
        "hr_coverage",
        "totaltimeadded",
        "totalmayomiles"
    ]]