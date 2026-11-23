"""Plot and fit a random-intercept mixed model for one repeated-measures outcome."""

from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
from patsy import build_design_matrices
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import chi2, norm, ttest_rel
import statsmodels.formula.api as smf
from statsmodels.stats.anova import AnovaRM
from statsmodels.stats.multitest import multipletests


results_path = (
    Path.home() / "OneDrive - Αριστοτέλειο Πανεπιστήμιο Θεσσαλονίκης"
    / "My Files" / "Projects" / "Inclined gait" / "Results" / "Peak_interval_results.xlsx"
)
outcome = "DCCA_Long_SI_EMG_Gast"
inclines = (0, 12.5, 25)
signals = ("HR", "EMG_Gast", "EMG_VL", "SI")
pairs = ("SI_EMG_Gast", "SI_EMG_VL", "EMG_Gast_EMG_VL")
outcomes = (
    [f"{metric}_{signal}" for metric in ("CV", "DFA") for signal in signals]
    + [f"DCCA_{band}_{pair}" for band in ("Short", "Long") for pair in pairs]
)


def load_outcome(path, outcome_name):
    """Return one row per participant and incline, including missing trials."""
    return reshape_outcome(pd.read_excel(path, sheet_name="Results").set_index("ID"), outcome_name)


def reshape_outcome(wide, outcome_name):
    column_to_incline = {f"{outcome_name}_{angle:g}deg": angle for angle in inclines}
    long = wide.loc[:, list(column_to_incline)].reset_index().melt(
        id_vars="ID", var_name="Column", value_name="Value",
    )
    long["Incline"] = long.pop("Column").map(column_to_incline)
    return long.sort_values(["ID", "Incline"], ignore_index=True)


def plot_outcome(data, outcome_name):
    """Connect observations from the same participant and show missing trials as gaps."""
    figure = px.line(
        data, x="Incline", y="Value", color="ID", markers=True,
        labels={"Incline": "Treadmill incline (°)", "Value": outcome_name, "ID": "Participant"},
        title=f"Participant profiles · {outcome_name}",
        template="plotly_white",
    )
    figure.update_traces(connectgaps=False)
    figure.update_xaxes(tickvals=inclines)
    return figure


def fit_mixed_model(data):
    """Estimate incline differences from 0° with a random intercept per participant."""
    observed = data.dropna(subset=["Value"])
    return smf.mixedlm(
        "Value ~ C(Incline, Treatment(reference=0.0))",
        data=observed,
        groups=observed["ID"],
    ).fit(reml=True)


def plot_residuals(data, model):
    observed = data.loc[model.fittedvalues.index].assign(
        Fitted=model.fittedvalues, Residual=model.resid,
    )
    figure = px.scatter(
        observed, x="Fitted", y="Residual", color=observed["Incline"].map("{:g}°".format),
        hover_name="ID", labels={"color": "Incline", "Fitted": "Fitted DCCA"},
        title="Residuals vs. fitted DCCA", template="plotly_white",
    )
    figure.add_hline(y=0, line_dash="dash")
    return figure


def incline_estimates(model):
    """Return all pairwise incline differences and pointwise Wald intervals."""
    reference = pd.DataFrame({"Incline": inclines})
    design = np.asarray(build_design_matrices([model.model.data.design_info], reference)[0])
    fixed = model.fe_params
    covariance = model.cov_params().loc[fixed.index, fixed.index].to_numpy()
    rows = []
    for lower, higher in combinations(range(len(inclines)), 2):
        contrast = design[higher] - design[lower]
        estimate = contrast @ fixed.to_numpy()
        standard_error = np.sqrt(contrast @ covariance @ contrast)
        margin = norm.ppf(0.975) * standard_error
        z = estimate / standard_error
        p_value = 2 * norm.sf(abs(z))
        rows.append((f"{inclines[higher]:g}° vs {inclines[lower]:g}°",
                     estimate, estimate - margin, estimate + margin, p_value, standard_error, z))
    return pd.DataFrame(rows, columns=[
        "Comparison", "Estimate", "CI lower", "CI upper", "p", "SE", "Wald z",
    ])


def overall_incline_test(model):
    """Jointly test that all fixed incline contrasts versus 0° are zero."""
    effects = model.fe_params.iloc[1:].to_numpy()
    covariance = model.cov_params().loc[model.fe_params.index[1:], model.fe_params.index[1:]]
    wald = effects @ np.linalg.solve(covariance.to_numpy(), effects)
    df = len(effects)
    return wald, df, chi2.sf(wald, df)


def summarize_mixed_models(models):
    rows = []
    for name, (_, model) in models.items():
        wald, df, p = overall_incline_test(model)
        participant_variance = model.cov_re.iloc[0, 0]
        fixed_variance = np.var(model.model.exog @ model.fe_params.to_numpy(), ddof=1)
        total_variance = fixed_variance + participant_variance + model.scale
        rows.append({
            "Outcome": name,
            "Effect": "Incline",
            "Wald chi-square": wald,
            "df": df,
            "p": p,
            "Participants": model.model.n_groups,
            "Observations": int(model.nobs),
            "Participant variance": participant_variance,
            "Residual variance": model.scale,
            "Marginal R²": fixed_variance / total_variance,
            "Conditional R²": (fixed_variance + participant_variance) / total_variance,
        })
    summary = pd.DataFrame(rows)
    if not np.isfinite(summary["p"]).all():
        raise RuntimeError("At least one overall incline test did not yield a finite p-value.")
    summary.insert(5, "Holm p", multipletests(summary["p"], method="holm")[1])
    return summary


def analyze_outcomes(wide, compare_anova=False):
    models = {}
    tables = []
    for name in outcomes:
        data = reshape_outcome(wide, name)
        model = fit_mixed_model(data)
        if not model.converged:
            raise RuntimeError(f"The mixed model for {name} did not converge.")
        estimates = incline_estimates(model)
        estimates.insert(0, "Outcome", name)
        estimates["Participants"] = model.model.n_groups
        estimates["Observations"] = int(model.nobs)
        estimates["Participant variance"] = model.cov_re.iloc[0, 0]
        if compare_anova:
            complete = data.pivot(index="ID", columns="Incline", values="Value").dropna()
            anova_data = complete.reset_index().melt(id_vars="ID", value_name="Value")
            anova = AnovaRM(anova_data, "Value", "ID", within=["Incline"]).fit()
            estimates["ANOVA overall p"] = anova.anova_table.at["Incline", "Pr > F"]
            _, _, estimates["LMM overall p"] = overall_incline_test(model)
            estimates["Complete participants"] = len(complete)
            estimates["Paired p"] = [
                ttest_rel(complete[higher], complete[lower]).pvalue
                for lower, higher in combinations(inclines, 2)
            ]
        tables.append(estimates)
        models[name] = (data, model)

    results = pd.concat(tables, ignore_index=True)
    if not np.isfinite(results["p"]).all():
        raise RuntimeError("At least one incline contrast did not yield a finite p-value.")
    results["Holm p"] = multipletests(results["p"], method="holm")[1]
    if compare_anova:
        results["Paired Bonferroni p"] = results.groupby("Outcome")["Paired p"].transform(
            lambda values: multipletests(values, method="bonferroni")[1]
        )
    return results, models


def build_report(data, model, estimates):
    figure = make_subplots(
        rows=3, cols=1, vertical_spacing=0.13,
        subplot_titles=[
            "1. Observed trials and model means",
            "2. All three incline comparisons",
            "3. Residuals by fitted value",
        ],
    )
    for trace in plot_outcome(data, outcome).data:
        trace.update(opacity=0.35, showlegend=False)
        figure.add_trace(trace, row=1, col=1)
    means = model.predict(pd.DataFrame({"Incline": inclines}))
    figure.add_trace(go.Scatter(
        x=inclines, y=means, mode="lines+markers", name="Model means",
        line={"color": "#17283c", "width": 4}, showlegend=False,
        hovertemplate="%{x}°: %{y:.3f}<extra>Model mean</extra>",
    ), row=1, col=1)
    figure.update_xaxes(title_text="Incline (°)", tickvals=inclines, row=1, col=1)
    figure.update_yaxes(title_text="DCCA coefficient", row=1, col=1)
    figure.add_annotation(
        x=12.5, y=means.iloc[1], text="Dark line: model means<br>Faint lines: individual participants",
        ax=-95, ay=-50, bgcolor="white", row=1, col=1,
    )

    low = estimates["CI lower"]
    high = estimates["CI upper"]
    center = estimates["Estimate"]
    figure.add_trace(go.Scatter(
        x=center, y=estimates["Comparison"], mode="markers+text",
        text=[f"{value:+.3f}" for value in center], textposition="top center",
        error_x=dict(type="data", symmetric=False, array=high - center, arrayminus=center - low),
        customdata=estimates[["CI lower", "CI upper"]],
        hovertemplate="%{y}: %{x:+.3f}<br>95% CI [%{customdata[0]:+.3f}, %{customdata[1]:+.3f}]<extra></extra>",
        marker={"size": 10, "color": "#176797"}, showlegend=False,
    ), row=2, col=1)
    figure.add_vline(x=0, line_dash="dash", row=2, col=1)
    figure.update_yaxes(autorange="reversed", row=2, col=1)
    figure.update_xaxes(title_text="DCCA difference · Approximate 95% CIs, unadjusted", row=2, col=1)
    figure.update_xaxes(range=[min(0, low.min()) - 0.035, high.max() + 0.05], row=2, col=1)
    last = estimates.iloc[-1]
    figure.add_annotation(
        x=last["CI lower"], y=last["Comparison"],
        text="Dot: estimated difference<br>Bar: confidence interval<br>Dashed line: zero difference",
        ax=126, ay=-30, bgcolor="white", row=2, col=1,
    )

    for trace in plot_residuals(data, model).data:
        figure.add_trace(trace, row=3, col=1)
    figure.add_hline(y=0, line_dash="dash", row=3, col=1)
    figure.update_xaxes(title_text="Fitted DCCA", row=3, col=1)
    figure.update_yaxes(title_text="Observed − fitted", row=3, col=1)
    figure.add_annotation(
        x=float(means.mean()), y=0,
        text="Above zero: observed was higher<br>Look for similar scatter by incline",
        ax=-96, ay=-56, bgcolor="white", row=3, col=1,
    )
    figure.update_layout(
        title="Long-range DCCA · Stride and gastrocnemius",
        template="plotly_white", height=1100, margin=dict(l=115, r=55, t=90, b=65),
        legend_title_text="Incline",
    )
    return figure


def main():
    wide = pd.read_excel(results_path, sheet_name="Results").set_index("ID")
    results, models = analyze_outcomes(wide, compare_anova=True)
    for column, heading in (
        ("p", "Unadjusted Wald p-values"),
        ("Holm p", f"Holm-adjusted Wald p-values across {len(results)} tests"),
        ("Paired Bonferroni p", "ANOVA follow-ups: Bonferroni-adjusted paired t-tests (3 per outcome)"),
    ):
        print(f"\n{heading}")
        table = results.pivot(index="Outcome", columns="Comparison", values=column)
        print(table.reindex(outcomes).to_string(float_format="{:.4g}".format))
    print("\nOverall incline tests (unadjusted)")
    overall_columns = ["Outcome", "Participants", "Complete participants", "ANOVA overall p", "LMM overall p"]
    print(results[overall_columns].drop_duplicates().to_string(index=False, float_format="{:.4g}".format))

    data, model = models[outcome]
    estimates = results.loc[results["Outcome"].eq(outcome)]
    report_path = Path(__file__).with_name("Incline_mixed_model.html")
    build_report(data, model, estimates).write_html(report_path, include_plotlyjs=True)
    print(f"Saved {report_path}")


if __name__ == "__main__":
    main()
