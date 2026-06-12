import pandas as pd
import numpy as np
import statsmodels.api as sm

def run_fitness_model(df, decay_fit=0.78, decay_fat=0.37):
    df = df.copy()
    df = df.sort_values("week").reset_index(drop=True)

    # Fill missing week numbers if there are gaps
    week_diffs = df["week"].diff().dropna()

    if (week_diffs > 1).any():
        full_range = pd.RangeIndex(
            int(df["week"].min()),
            int(df["week"].max()) + 1
        )

        df = (
            df.set_index("week")
            .reindex(full_range)
            .reset_index()
            .rename(columns={"index": "week"})
        )

        # For inserted missing weeks, assume no training
        df["totalmayomiles"] = df["totalmayomiles"].fillna(0)

        # If hr_coverage exists, missing weeks should have no HR coverage
        if "hr_coverage" in df.columns:
            df["hr_coverage"] = df["hr_coverage"].fillna(0)

    # Treat 0 pace/HR as missing
    df["avgpace"] = df["avgpace"].replace(0, np.nan)
    df["avgrunhr"] = df["avgrunhr"].replace(0, np.nan)

    # If hr_coverage is not present, create a basic version:
    # 1 = HR exists for the week, 0 = HR missing
    if "hr_coverage" not in df.columns:
        df["hr_coverage"] = df["avgrunhr"].notna().astype(float)

    df["hr_coverage"] = (
        pd.to_numeric(df["hr_coverage"], errors="coerce")
        .fillna(0)
        .clip(0, 1)
    )

    # HR efficiency regression: HR unexplained by pace
    eff_data = df.dropna(subset=["avgpace", "avgrunhr"])

    if len(eff_data) < 8:
        raise ValueError(
            "Not enough heart-rate data to run the HR-based fitness model. "
            "The model needs at least 8 weeks with both pace and average running HR."
        )
    
    X = sm.add_constant(eff_data["avgpace"])
    model_eff = sm.OLS(eff_data["avgrunhr"], X).fit()

    df["hr_eff"] = np.nan
    df.loc[eff_data.index, "hr_eff"] = model_eff.resid

    # Standardize pace and HR
    df["z_pace"] = (
        df["avgpace"] - df["avgpace"].mean()
    ) / df["avgpace"].std(ddof=1)

    df["z_hr"] = (
        df["avgrunhr"] - df["avgrunhr"].mean()
    ) / df["avgrunhr"].std(ddof=1)

    # ----------------------------
    # Standardized fitness factor
    # ----------------------------

    # Full version: uses both pace and HR
    df["S_FF_full"] = -(df["z_pace"] + df["z_hr"]) / 2

    # Pace-only version: useful for manual-upload weeks with no HR
    df["S_FF_pace_only"] = -df["z_pace"]

    # Start with the original/full version
    df["S_FF"] = df["S_FF_full"]

    # Both missing = true no-run/rest week
    both_missing = df["avgpace"].isna() & df["avgrunhr"].isna()

    # One missing = manual/partial data week
    one_missing = df["avgpace"].isna() ^ df["avgrunhr"].isna()

    # Mixed week = some HR coverage, but not complete HR coverage
    mixed_week = (
        (df["hr_coverage"] > 0) &
        (df["hr_coverage"] < 1) &
        df["S_FF_full"].notna() &
        df["S_FF_pace_only"].notna()
    )

    # For mixed manual/GPS weeks:
    # blend pace+HR score with pace-only score based on HR coverage
    df.loc[mixed_week, "S_FF"] = (
        df.loc[mixed_week, "hr_coverage"] * df.loc[mixed_week, "S_FF_full"]
        + (1 - df.loc[mixed_week, "hr_coverage"]) * df.loc[mixed_week, "S_FF_pace_only"]
    )

    # For fully manual weeks or one-missing cases:
    # use whichever z-score exists so S_FF does not become NaN
    df.loc[one_missing, "S_FF"] = (
        -df.loc[one_missing, ["z_pace", "z_hr"]].mean(axis=1)
    )

    # For true rest/no-run weeks:
    # neutral efficiency; trainingload will be driven by low/zero totalmayomiles
    df.loc[both_missing, "S_FF"] = 0

    # Final guardrail: no NaN should reach trainingload through S_FF
    df["S_FF"] = df["S_FF"].fillna(0)

    # Standardize training volume
    df["z_TMM"] = (
        df["totalmayomiles"] - df["totalmayomiles"].mean()
    ) / df["totalmayomiles"].std(ddof=1)

    # Training load
    df["trainingload"] = df["z_TMM"] * (1 - df["S_FF"])

    # Final guardrail: no NaN should enter the recursive loop
    df["trainingload"] = df["trainingload"].fillna(0)

    # Recursive fitness/fatigue
    df["fit"] = np.nan
    df["fat"] = np.nan

    df.loc[0, "fit"] = df.loc[0, "trainingload"]
    df.loc[0, "fat"] = df.loc[0, "trainingload"]

    for i in range(1, len(df)):
        df.loc[i, "fit"] = (
            df.loc[i, "trainingload"]
            + decay_fit * df.loc[i - 1, "fit"]
        )

        df.loc[i, "fat"] = (
            df.loc[i, "trainingload"]
            + decay_fat * df.loc[i - 1, "fat"]
        )

    # Regression: hr_eff on fit and fat
    reg_data = df.dropna(subset=["hr_eff", "fit", "fat"])

    X2 = sm.add_constant(reg_data[["fit", "fat"]])
    model_fitness = sm.OLS(reg_data["hr_eff"], X2).fit(cov_type="HC1")

    b_fit = model_fitness.params["fit"]
    b_fat = model_fitness.params["fat"]

    # Modeled fitness index
    df["fitness_index"] = -(b_fit * df["fit"] + b_fat * df["fat"])

    df["z_fitness_index"] = (
        df["fitness_index"] - df["fitness_index"].mean()
    ) / df["fitness_index"].std(ddof=1)

    # Drop internal helper columns so output stays cleaner
    df = df.drop(
        columns=["S_FF_full", "S_FF_pace_only"],
        errors="ignore"
    )

    return df, model_fitness