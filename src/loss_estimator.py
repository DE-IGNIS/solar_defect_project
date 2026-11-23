
import pandas as pd
import matplotlib.pyplot as plt

from pvlib.location import Location

from src.config import METRICS_DIR, FIGURES_DIR


# Project assumptions: refine with literature in later phases.
LOSS_FRAC = {
    "No-Anomaly": 0.00,
    "Hot-Spot": 0.05,
    "Hot-Spot-Multi": 0.15,
    "Cell": 0.05,
    "Cell-Multi": 0.15,
    "Cracking": 0.05,
    "Diode": 0.33,
    "Diode-Multi": 0.66,
    "Offline-Module": 1.00,
    "Shadowing": 0.15,
    "Soiling": 0.10,
    "Vegetation": 0.20,
}

ACTIONS = {
    "Soiling": ("Clean", 20),
    "Vegetation": ("Remove vegetation", 50),
    "Shadowing": ("Remove obstruction", 100),
    "Hot-Spot": ("Inspect cell", 500),
    "Hot-Spot-Multi": ("Inspect module", 1000),
    "Cell": ("Inspect cell", 500),
    "Cell-Multi": ("Replace module", 8000),
    "Cracking": ("Inspect/replace", 4000),
    "Diode": ("Replace junction box", 1500),
    "Diode-Multi": ("Replace junction box", 2500),
    "Offline-Module": ("Check wiring/replace", 3000),
    "No-Anomaly": ("None", 0),
}


# Site assumptions from the project guide: Bhopal, Madhya Pradesh.
SITE = Location(
    latitude=23.26,
    longitude=77.41,
    tz="Asia/Kolkata",
    altitude=500,
)

PANEL_POWER_WP = 400
TARIFF_INR_PER_KWH = 5.0


def calculate_peak_sun_hours():
    times = pd.date_range(
        "2025-01-01",
        "2025-12-31 23:00",
        freq="h",
        tz=SITE.tz,
    )

    clear_sky = SITE.get_clearsky(times)

    daily_psh = clear_sky["ghi"].sum() / 1000 / 365

    # Approximate cloud/haze derating from the guide.
    adjusted_psh = daily_psh * 0.75

    return adjusted_psh


def estimate_loss(defect_class, psh):
    loss_fraction = LOSS_FRAC[defect_class]

    monthly_kwh = (
        PANEL_POWER_WP / 1000
        * psh
        * 30
        * loss_fraction
    )

    monthly_inr = monthly_kwh * TARIFF_INR_PER_KWH

    action, repair_cost = ACTIONS[defect_class]

    payback_months = (
        repair_cost / monthly_inr
        if monthly_inr > 0
        else float("inf")
    )

    return {
        "defect_class": defect_class,
        "assumed_loss_fraction": loss_fraction,
        "estimated_kwh_month": monthly_kwh,
        "estimated_inr_month": monthly_inr,
        "recommended_action": action,
        "assumed_repair_cost_inr": repair_cost,
        "estimated_payback_months": payback_months,
    }


def main():
    psh = calculate_peak_sun_hours()

    results = [
        estimate_loss(defect_class, psh)
        for defect_class in LOSS_FRAC
    ]

    df = pd.DataFrame(results)

    # Higher monthly financial loss receives higher priority.
    df = df.sort_values(
        "estimated_inr_month",
        ascending=False,
    ).reset_index(drop=True)

    df["priority_rank"] = range(1, len(df) + 1)

    output_path = METRICS_DIR / "energy_loss_estimates.csv"
    df.to_csv(output_path, index=False)

    # Plot estimated monthly energy loss by defect class.
    plt.figure(figsize=(11, 6))

    plt.bar(
        df["defect_class"],
        df["estimated_kwh_month"],
    )

    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Estimated energy loss (kWh/month)")
    plt.title("Estimated Monthly Energy Loss by Defect Class")
    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR / "energy_loss_by_class.png",
        dpi=200,
    )
    plt.close()

    print(f"Peak Sun Hours: {psh:.2f} kWh/m²/day")
    print(f"\nSaved: {output_path}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
