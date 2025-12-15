from collections import defaultdict

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

from config.constants import Paths
from evaluation.data import DataExtractor
from utils.misc import value_and_error_to_latex, median_and_ci_to_latex


def generate_report(generate_figures=True,
                    generate_latex_tables=True,
                    generate_csv_tables=True) -> None:
    Paths.REPORT.mkdir(parents=True, exist_ok=True)

    de = DataExtractor(ds_tuples=None,  # [("wind", "T11"), ("pv", "01")],
                       models=None,  # ["mlp", "rf", "lgbm", "xgboost"],
                       fs_methods=None,
                       clustering_method_tuples=None,  # [("random", "3"), ("singletons", None)],
                       feature_sets=None,  # ["digital_twin"],
                       n_features_list=None,
                       )
    all_results_df = de.get_dataframe(
        columns=["name", "domain", "features", "n_features", "fs_method", "model", "clustering_method", "mae", "rmse",
                 "r2",
                 "mdae", "me", "mde", "fs_runtime", "fitted_models", "selected_features", "feature_level_entries",
                 "fallbacks", "feature_level_total_evaluated", "explanation", "fast_stops", "total_iterations"],
        # , "validity_check"],
        aggregation_method="none")  # "median")

    # Figures
    if generate_figures:
        Paths.FIGURES.mkdir(parents=True, exist_ok=True)
        agg_plot(all_results_df)
        runtime_plot(all_results_df)
        rmse_over_n_plot(all_results_df)

    # Tables
    tables = dict()
    if generate_latex_tables or generate_csv_tables:
        tables["best_features"] = best_features(all_results_df, n_features=10)
    for table_name, table in tables.items():
        if generate_latex_tables:
            (Paths.TABLES / "latex").mkdir(parents=True, exist_ok=True)
            with open(Paths.TABLES / "latex" / f"{table_name}.tex", "w") as f:
                f.write(table.to_latex())
        if generate_csv_tables:
            (Paths.TABLES / "csv").mkdir(parents=True, exist_ok=True)
            table.to_csv(Paths.TABLES / "csv" / f"{table_name}.csv")

    # Build report
    report = "# Summary of results reported in the paper"
    report += "\n\n"
    report += runtime_comparison_report(all_results_df)
    report += "\n\n"
    report += early_removal_report(all_results_df)
    report += "\n\n"
    report += csfs_explanation_report(all_results_df)

    with open(Paths.REPORT / "report.md", "w") as f:
        f.write(report)


def agg_plot(all_results_df: pd.DataFrame) -> None:
    with plt.style.context(Paths.MPLSTYLES / 'lncs.mplstyle'):
        titles = [["WT-S1", "WT-S2"],
                  ["PV-S1", "PV-S2"]]
        metric = "rmse"
        fig, axs = plt.subplots(2, 2, sharex=True, sharey=False, figsize=(7.5, 3.5))
        curr_results = all_results_df.explode(metric)
        for row_idx, domain in enumerate(["wind", "pv"]):
            for col_idx, features in enumerate(["digital_twin", "forecast_available"]):
                # sns.boxplot(curr_results, x="fs_method_plot", y="rmse")
                # plt.title(f"Domain: {domain} | Features: {features}")
                # plt.show()

                # sns.pointplot(curr_results[row_idx][col_idx], x="fs_method_plot", y=metric, ax=axs[row_idx, col_idx], markers="x", color="black",
                #               linestyle='none', markersize=3,  # marker scaling
                #               err_kws={  # make errorbars thin
                #                   "linewidth": 0.8,
                #                   "color": "black",
                #               },
                #               errorbar="sd")

                sns.boxplot(curr_results.query(f"(`domain` == '{domain}') & (`features` == '{features}')"),
                            x="fs_method_plot", y=metric, ax=axs[row_idx, col_idx])

                # axs[row_idx, col_idx].set_title(f"Domain: {domain} | Features: {features}")
                axs[row_idx, col_idx].set_title(titles[row_idx][col_idx])
                y_label = metric.upper() + (" [MW]" if domain == "pv" else " [kW]")
                axs[row_idx, col_idx].set_ylabel(y_label)
                axs[row_idx, col_idx].set_xlabel("")

        plt.tight_layout()
        plt.savefig(Paths.FIGURES / "agg_rmse.pdf")
        plt.show()


def rmse_over_n_plot(all_results_df: pd.DataFrame) -> None:
    metric = "rmse"
    models = ["mlp", "mlp"]
    features = "digital_twin"
    clustering_methods = ['correlation', 'singletons', None]
    with plt.style.context(Paths.MPLSTYLES / 'lncs.mplstyle'):
        fig, axs = plt.subplots(nrows=2, figsize=(7, 6))
        i = 0
        for ds_plot, model in zip(["WT-S1", "PV-S1"], models):
            ax = axs[i]
            curr_results = all_results_df.query(
                f"(`ds_plot` == '{ds_plot}') & (`model`== '{model}')").explode(metric)
            sns.boxplot(curr_results.query(f"(`clustering_method` in {clustering_methods})"),
                        # & (`fs_method_plot` != 'F-value')"),
                        x="n_features", y="rmse", hue="fs_method_plot", ax=ax)
            # set legend title to "FS method"
            # ax.legend(title="FS method")
            ax.set_ylabel("RMSE " + ("[MW]" if ds_plot.startswith("PV") else "[kW]"))
            ax.set_xlabel("$|\\mathcal{F'}|$")
            ax.set_title(ds_plot)

            if i == 0:
                fig.legend(ncols=2, bbox_to_anchor=(0.97, 0.93))
                ax.set_xlabel("")
            ax.get_legend().remove()
            i += 1

        plt.tight_layout()
        plt.savefig(Paths.FIGURES / f"{model}_{clustering_methods[0]}_rmse_over_n.pdf")
        plt.show()


def runtime_plot(all_results_df: pd.DataFrame) -> None:
    curr_results = all_results_df.loc[~all_results_df["clustering_method"].isna(), :]
    fs_order = ["CSFS", "mutual_info", "f_value", "RF_FI"]
    with plt.style.context(Paths.MPLSTYLES / 'lncs.mplstyle'):
        fig, ax = plt.subplots(ncols=4, figsize=(7, 2.5))
        i = 0
        for ds_plot in ["WT-S1", "WT-S2", "PV-S1", "PV-S2"]:
            sns.boxplot(curr_results.query(f"`ds_plot` == '{ds_plot}'"),
                        x="fs_method_plot", y=curr_results["fs_runtime"] / 3600,
                        # order=fs_order,
                        hue="fs_method_plot",
                        log_scale=False, orient="v", ax=ax[i])
            ax[i].set_xticks([])
            ax[i].set_xlabel(ds_plot)
            if i == 0:
                ax[i].set_ylabel("Runtime [h]")
                ax[i].get_legend().remove()
                # ax[i].legend(bbox_to_anchor=(1.05, 1), ncols=2)
                fig.legend(ncols=1, bbox_to_anchor=(0.942, 0.95))
            else:
                ax[i].set_ylabel("")
                # hide legend
                ax[i].get_legend().remove()

            i += 1
        plt.tight_layout()
        plt.savefig(Paths.FIGURES / "runtime_comparison.pdf")
        plt.show()


def best_features(all_results_df: pd.DataFrame, n_features: int) -> pd.DataFrame:
    df = (
        all_results_df
        .query(f"`n_features` == {n_features}")
    )
    df.loc[:, "rmse_mean"] = df.loc[:, "rmse"].apply(np.mean)
    df.loc[:, "rmse_std"] = df.loc[:, "rmse"].apply(np.std)
    df.loc[:, "r2_mean"] = df.loc[:, "r2"].apply(np.mean)
    df.loc[:, "r2_std"] = df.loc[:, "r2"].apply(np.std)
    df.loc[:, "mae_mean"] = df.loc[:, "mae"].apply(np.mean)
    df.loc[:, "mae_std"] = df.loc[:, "mae"].apply(np.std)

    best = df.loc[df.groupby(["domain", "features"])["rmse_mean"].idxmin()]

    table_data = defaultdict(list)
    metrics = ["rmse", "mae", "r2"]
    i = 0
    for (domain, features), ds_scenario in zip(
            [("wind", "digital_twin"), ("wind", "forecast_available"), ("pv", "digital_twin"),
             ("pv", "forecast_available")], ["WT-S1", "WT-S2", "PV-S1", "PV-S2"]):
        _curr_best = best.query(f"(`domain` == '{domain}') & (`features` == '{features}')")
        assert len(_curr_best) == 1
        _curr_features = sorted(_curr_best.selected_features.item())
        _curr_model = _curr_best.model.item()
        if i == 0:
            table_data[0].extend([f"feature {i}" for i in range(len(_curr_features))])
            table_data[0].append("Model")
        table_data[ds_scenario].extend(_curr_features)
        table_data[ds_scenario].append(_curr_model)
        for m in metrics:
            if i == 0:
                table_data[0].append(m)
            table_data[ds_scenario].append(
                value_and_error_to_latex(_curr_best[f"{m}_mean"].item(), _curr_best[f"{m}_std"].item(), precision=6))
        i += 1
    best_features_df = pd.DataFrame(table_data)

    return best_features_df


def runtime_comparison_report(all_results_df: pd.DataFrame) -> str:
    res = "## Comparison of CSFS and SFS runtime \n\n"
    res += "### Relative improvements by dataset scenarios:\n"
    ours = ["feature_importance", "correlation", "random"]
    baseline = "singletons"
    for ds_plot in ["WT-S1", "WT-S2", "PV-S1", "PV-S2"]:
        ours_df = all_results_df.query(f"(`ds_plot` == '{ds_plot}') & (`clustering_method` in {ours})")
        baseline_df = all_results_df.query(f"(`ds_plot` == '{ds_plot}') & (`clustering_method` == '{baseline}')")
        merged_df = ours_df.merge(baseline_df, how="right", on=["model", "n_features"])
        diff = 100 * (merged_df.loc[:, "fs_runtime_y"] - merged_df.loc[:, "fs_runtime_x"]) / merged_df.loc[
            :, "fs_runtime_y"]

        res += f"{ds_plot}: {median_and_ci_to_latex(diff, p=0.95, precision=1)}\n"

    ours_df = all_results_df.query(f"(`clustering_method` in {ours})")
    baseline_df = all_results_df.query(f"(`clustering_method` == '{baseline}')")
    merged_df = ours_df.merge(baseline_df, how="left", on=["model", "n_features", "ds_plot"])
    diff = 100 * (merged_df.loc[:, "fs_runtime_y"] - merged_df.loc[:, "fs_runtime_x"]) / merged_df.loc[
        :, "fs_runtime_y"]
    # median_imp = np.median(diff)
    # ci_low, ci_high = np.percentile(diff, [2.5, 97.5])
    res += f"Overall: {median_and_ci_to_latex(diff, p=0.95, precision=1)}\n"
    res += f"Overall mean improvement: {diff.mean()}\n"

    return res


def early_removal_report(all_results_df: pd.DataFrame) -> str:
    res = "## How often were clusters early removed in CSFS? \n\n"

    curr_results = all_results_df.query("`clustering_method` in ['correlation', 'feature_importance', 'random']")
    res += f"Total CSFS across all experiments: {curr_results['total_iterations'].sum()}\n"
    res += f"Among those, {curr_results['fast_stops'].sum()} ({curr_results['fast_stops'].sum() / curr_results['total_iterations'].sum()}) were early cluster removals.\n"

    return res


def csfs_explanation_report(all_results_df: pd.DataFrame) -> str:
    res = "## CSFS explainability \n\n"
    res += "The CSFS Feature Selection procedure produces explainable results. In the results dataframe produced by the DataExtractor class, the column 'explanation' provides information why a feature (cluster) was discarded in each iteration. One of these explanations looks as follows:\n"

    df = all_results_df.loc[~all_results_df.explanation.isna(), :]
    df.loc[:, "rmse_mean"] = df.loc[:, "rmse"].apply(np.mean)

    explanation = df.loc[df["rmse_mean"].idxmin(), "explanation"]

    res += explanation

    return res


if __name__ == "__main__":
    generate_report()
