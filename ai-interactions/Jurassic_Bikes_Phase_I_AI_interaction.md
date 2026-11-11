# AI4VIS — TiRex-2 Visualization Interaction Log

**Project:** Jurassic Bikes — AI-aided pair, Phase I  
**AI assistant:** ChatGPT  
**Scope:** Phase I, TiRex-2 external input/output behavior 
**Evidence:** Final executed Washington, DC notebook (`Jurassic_Bikes_Phase_I_AI_part.ipynb`)


## Interaction 1 — First comparison and visualizations (May–June 2024)

**User:**

> Given the data generation script which produces `capital_hourly.csv` with hourly Washington D.C. Capital Bikeshare data and Open-Meteo weather features, write a complete, executable Python script using TiRex-2 to perform the following forecasting experiments:
>
> **Experiment 1: Calendar-Only Forecasting**
> - Configure TiRex-2 using the historical `rides_started` target sequence and future-known calendar features (`hour`, `dow`, `month`, `is_weekend`, `is_holiday`) as covariates.
> - Run zero-shot forecasting for the test period and calculate evaluation metrics (RMSE, MAE).
>
> **Experiment 2: Calendar + Weather Forecasting**
> - Configure TiRex-2 using the historical `rides_started` target sequence, the calendar features, plus the weather covariates (`temp`, `humidity`, `windspeed`, `precip`, `cloudcover`).
> - Run zero-shot forecasting for the same test period and calculate evaluation metrics (RMSE, MAE).
>
> **Comparison & Visualization:**
> - Print a comparison table of RMSE and MAE between Experiment 1 (Calendar-Only) and Experiment 2 (Calendar + Weather).
> - Generate a clean Matplotlib plot comparing the actual `rides_started` against both model predictions over the test timeframe.

**ChatGPT:**

I recommend treating this as a controlled comparison of two configurations of the **same pretrained TiRex-2 model**. Our central question is: *Does historical weather provide additional predictive information beyond historical bike demand and calendar features?*

For a manageable first evaluation, I suggest starting with **May 1–June 30, 2024**. I recommend a **14-day (336-hour) context** and a **24-hour forecast horizon**. Both configurations should use identical forecast dates. Calendar features can be known for the forecast period, but actual observed future weather must not be supplied as a forecasting input.

For the first visualization, I would draw actual hourly rentals and both median forecasts on the same 24-hour line chart. We can then compare MAE and RMSE over a larger collection of dates. A single attractive forecast plot is useful for explaining the task, but it is not enough to establish which configuration is better.

**Decision — Accepted, then refined.** We kept a 24-hour forecast plot as a sanity check but moved to repeated evaluation windows for the substantive findings. The initial May–June period was a convenient pilot, **not** a scientific claim that those two months represented all weather regimes. The model was used zero-shot; no retraining took place.

**Implementation check — Important limitation identified:** The initial request mentioned `month`, `is_holiday`, and `cloudcover`. The executed notebook actually provides cyclic hour/day-of-week encodings plus a weekend flag as calendar covariates, and uses `temp`, `precip`, `humidity`, and `windspeed` as past weather features. **Explicit month, holiday, and cloud-cover inputs are not implemented.** The rolling evaluation reports MAE and coverage but does not produce the requested RMSE comparison as a final result. A strong AI workflow must verify that the final input tensors and metrics match the original requirements instead of assuming every requested item was included.

---

## Interaction 2 — Daily rolling errors and predictive uncertainty

**User:**

> Yes, I agree with your structured approach. A single representative day is too limited because I suppose bike-sharing demand patterns shift between seasons (e.g., cold winter commuting vs. pleasant spring/summer riding, or rainy days vs. clear days).  
> Keep the rolling evaluation across the test period to calculate reliable MAE and RMSE metrics. I want to see `calendar_error` and `weather_error` (calculated as predicted minus actual), plus a boolean column `inside_p10_p90` indicating whether the actual value fell within the p10–p90 prediction interval.

**ChatGPT:**

I agree. I recommend **rolling-origin backtesting** instead of judging the model from a single 24-hour window. For each evaluated date, we take the preceding 336 observed hours and forecast the next 24 hours. Moving the origin lets us see how performance changes from day to day.

Let's preserve one record per forecasted hour in `rolling_dc`, including the date, local hour, observed rentals, both median forecasts, signed errors, and the weather-conditioned p10 and p90 forecasts. The errors should be defined as **prediction minus actual**: positive means overprediction; negative means underprediction.

The `inside_p10_p90` flag will let us measure the fraction of observed hourly counts falling inside the model's nominal **80% prediction interval**. We should assess that coverage over many hours, not one day.

I recommend three complementary plots:

1. **Daily MAE over time**, with one line for each configuration.
2. **Daily weather improvement**, defined as calendar MAE minus weather-conditioned MAE. Values above zero mean weather helped; values below zero mean it hurt.
3. A **Date × Hour error heatmap** to see whether certain days and hours are consistently difficult.

For the heatmap, signed errors would reveal systematic overprediction and underprediction, while absolute errors would reveal where error magnitudes are largest. Which question matters more to you?

**Decision — Accepted** We retained signed prediction errors and p10–p90 interval coverage in the hourly evaluation table. 

**Evidence — Five genuine rows from the final `rolling_dc` output:**

| Date | Local hour | Actual | Calendar p50 | Weather p50 | Calendar error | Weather error | Inside p10–p90 |
|---|---:|---:|---:|---:|---:|---:|---|
| 2023-12-15 | 00 | 121 | 121.52 | 124.99 | +0.52 | +3.99 | True |
| 2023-12-15 | 01 | 75 | 48.15 | 50.98 | −26.85 | −24.02 | True |
| 2023-12-15 | 02 | 34 | 20.42 | 23.33 | −13.58 | −10.67 | True |
| 2023-12-15 | 03 | 16 | 15.50 | 17.44 | −0.50 | +1.44 | True |
| 2023-12-15 | 04 | 18 | 31.70 | 31.97 | +13.70 | +13.97 | True |

These are **real predictions**, not invented example values. For example, `calendar_error = calendar_p50 - actual`. The full table also retains timestamps, p10 and p90, and precipitation. Note that the p10 forecast can be negative even though rental counts cannot, which is a model-output limitation worth reporting.

---

## Interaction 3 — Expanding the pilot to include winter

**User** 

> These graphs cover only May and June 2024. May and June are generally not very cold or rainy. If we want to understand the effect of weather, we should examine winter months as well.

**ChatGPT:**

I think that is an important limitation of our initial evaluation. MAE is the average absolute error in **rentals per hour**, while WAPE measures the total absolute error relative to the total observed demand, expressed as a percentage. WAPE is particularly useful when low-demand and high-demand periods have very different scales.

I recommend extending the dataset to **December 2023–June 2024**. That gives us winter, spring, and early summer conditions, including colder and potentially wetter hours. We can keep the same context length, forecast horizon, two model configurations, and visualization methods. This changes the **evaluation coverage**, not the pretrained model's weights.

I would also avoid concluding that a temperature or rainfall category *causes* an error pattern just because the groups have different MAEs. Demand magnitude, season, and commuting hours may also matter.

**User** 

> Please update the Python scripts to use data from December 2023 through June 2024.

**ChatGPT:**

We can retain the same visualization and evaluation design, but I recommend checking the data-generation dates and the timestamp handling. The expanded period includes a daylight-saving transition in Washington, DC, so hourly observations should be aligned consistently in UTC while calendar features are computed using local time.

Once the new data is prepared, we should regenerate the rolling forecasts and the plots. We should not reuse May–June performance numbers as if they were measured over the longer interval.

**Decision — User-initiated refinement, accepted.** The user identified a sampling limitation that the AI's initial convenient date choice did not adequately address. We broadened the evaluation to include winter, without changing the pretrained model or the 14-day / 24-hour protocol.

**— Change to the data-generation period:**

```python
# Original pilot (May–June 2024)
START_DATE = "2024-05-01"
END_DATE   = "2024-07-01"  # exclusive

# Revised seasonal coverage (December 2023–June 2024)
START_DATE = "2023-12-01"
END_DATE   = "2024-07-01"  # exclusive
```

The executed notebook uses `--start 2023-12-01 --end 2024-07-01` and contains **5,111 consecutive UTC hourly rows**. It correctly crosses the March 2024 daylight-saving transition. The transition day's nonstandard local duration is excluded from 24-hour *daily* backtests. This is a concrete example of a change in data scope requiring code and validation beyond merely changing a plot title.

---

## Interaction 4 — Choosing a signed-error heatmap to investigate prediction bias

**User:**

> I want to analyze the macro-level error patterns across our full coverage period from December to June. Could we use a Date × Hour heatmap to identify when the model makes particularly large forecasting errors? I was considering absolute errors for the calendar-only baseline.


**ChatGPT:**

An absolute-error heatmap would highlight when forecasting errors are largest, regardless of their direction. However, I recommend retaining my earlier suggestion to use signed errors if our objective is to investigate systematic prediction bias.

Using weather_error = weather_p50 - actual, we can identify both overprediction and underprediction:

Red: Positive errors — the model overpredicts bike demand.

Blue: Negative errors — the model underpredicts bike demand.

White: Errors close to zero.

I recommend using calendar dates as rows, local hours (0–23) as columns, and a diverging red–blue colormap (RdBu_r) centered at zero.

This visualization could reveal whether the model systematically overpredicts or underpredicts during particular hours or periods.

One limitation is that displaying approximately 198 days may make the heatmap visually dense. We should therefore reduce the number of date labels while preserving the underlying observations.

**User:**

I agree with your suggestion. Let's keep the signed-error heatmap because distinguishing overprediction from underprediction is more informative for investigating model bias.

**Decision — Accepted.** We selected the signed-error Date × Hour heatmap for the weather-conditioned model. The final notebook uses weather_error with the diverging RdBu_r colormap. This allows us to investigate systematic overprediction and underprediction rather than error magnitude alone.

**AI-assisted visualization limitation:** A heatmap containing nearly 200 dates can become difficult to read. Showing fewer date labels improves readability without removing observations.

---

## Interaction 5 — Rain, temperature, and forecasting errors

**User:**

> Now I want to focus on analyzing the impact of **weather conditions (specifically rainy days and temperature)** on model performance and demand errors.
>
> Please write a Python script using `pandas`, `seaborn`, and `matplotlib` to investigate this:
>
> 1. Create a categorical split or filter for rainy vs. dry hours (e.g., `observed_precip > 0` vs `== 0`).
> 2. Generate a visualization (such as a `sns.scatterplot` or `sns.lmplot`) showing the relationship between **temperature (`temp`)** and **bike demand/errors**, split or color-coded by precipitation status (rainy vs. dry).
> 3. Alternatively, calculate and display summary metrics (like MAE or median error) grouped by temperature bins and rainy/dry conditions to see where the calendar model fails and the weather model succeeds.

**ChatGPT:**

I recommend using **two complementary views**. First, we can make a scatterplot of temperature versus observed hourly rentals, color-coded as rainy or dry, and another scatterplot of temperature versus the calendar-only absolute error. These would help us see whether observed demand and forecasting difficulty vary across temperatures and precipitation conditions.

Second, I recommend grouped **MAE comparisons by temperature bin and rainy/dry status**. The scatterplots are useful for exploration, but thousands of overlapping points may be difficult to interpret; the grouped summaries will make the differences between the calendar-only and weather-conditioned forecasts clearer.

We should join the observed weather at each forecast timestamp **only for retrospective analysis**. These are the real weather conditions associated with the errors, not future weather values that were supplied to TiRex-2.

I would always show the number of hours in each category. A small rainy-and-freezing subset is less reliable than a category containing hundreds or thousands of hours. And I would describe any difference as an **association**, not proof of a causal weather effect.

As a next step, I recommend trying separate **dry, light-rain, and heavier-rain** bins rather than just a rainy/dry split.

**Decision — Refined.** We moved from an exploratory rainy/dry and temperature scatterplot proposal to a stronger **grouped MAE by weather category** comparison, because it answers the baseline-versus-weather question more directly. The final notebook implements grouped precipitation and temperature analyses; the separately proposed two-panel rainy/dry scatterplot was not independently verified in the executed notebook.

**Important evaluation safeguard:** Weather observed during the 24-hour *test* window may be merged with errors to identify the actual conditions, but it must **not** be supplied as a future-known model input. The inference experiment uses only *historical* observed weather.

---

## Interaction 6 — Severity bins and the assumption that weather must help

**User:**

> Now I want to evaluate and compare the **calendar-only model vs. the weather-conditioned model** across different weather conditions or intensity bins.
>
> Please write a Python script using `pandas`, `seaborn`, and `matplotlib` to:
>
> 1. Bin or categorize the data by weather severity (e.g., low, medium, high wind/humidity, or clear vs. cloudy vs. rainy).
> 2. Calculate and plot the **Mean Absolute Error (MAE)** or **RMSE** for both the calendar-only model and the weather-conditioned model within each weather bin.
> 3. Generate a clean grouped bar chart or line plot showing how the weather-conditioned model successfully reduces errors compared to the baseline calendar model as weather conditions worsen.

**ChatGPT:**

I agree with comparing the two configurations across severity bins, but I would change one part of the research question. We should not design the chart to show that the weather-conditioned model **successfully reduces error** before we know whether it does. I recommend testing *if* and *where* it helps, including cases where it makes performance worse.

I would use precipitation as the primary severity variable because its units and interpretation are straightforward. We can classify observations as **dry/trace (≤0.2 mm/hour), light rain (0.2–2 mm/hour), and more rain (>2 mm/hour)**. Wind speed and relative humidity could be explored separately, but they should not automatically be combined into a single severity score.

For the main figure, I recommend a **grouped MAE bar chart**, showing calendar-only and calendar-plus-past-weather MAE side by side for each precipitation category. I would add the category sample sizes. A second chart could show **MAE improvement = calendar MAE − weather MAE**, with a horizontal zero line. This makes it immediately clear when weather helps or hurts.

The key is to compare the two models **within each condition**, rather than infer improvement merely because rainy hours have larger errors than dry hours. We should also remember that these are *historical* weather covariates, not forecasts of future rainfall.

**Decision — Hypothesis corrected and focused.** The user prompted a visualization that would show historical weather *successfully* reducing errors in adverse conditions. I rejected that presupposition and suggested a neutral **test of whether** it helps, complete with negative improvements when it hurts. We chose precipitation intensity as the primary severity axis. Wind and humidity severity charts were suggested as exploratory alternatives; **they are not presented as completed final notebook outputs**.

**— Actual binning logic:**

```python
    wx_eval_dc["rain_category"] = pd.cut(
        wx_eval_dc["precip"],
        bins=[-np.inf, 0.2, 2.0, np.inf],
        labels=[
            "Dry (<=0.2 mm)",
            "Light rain (0.2–2 mm)",
            "More rain (>2 mm)"
        ]
    )
# Compare calendar MAE and past-weather MAE within each category.
```

In the final notebook, this is applied to the observed evaluation-hour `precip` series (`wx_eval_dc["precip"]`), with the longer descriptive labels shown in the plots.

---

## Interaction 7 — Day-of-week, demand levels, and abrupt changes 

**User** 

> Weather is not the only condition that might affect forecasting errors. Could we also compare weekdays with weekends, low-demand days with high-demand days, and periods when bike-sharing demand changes suddenly? I would like to visualize the model's behavior in these different situations.

**ChatGPT:**

Yes. I recommend dividing it into three behavior-oriented comparisons.

First, compare **weekday and weekend MAE**. That may tell us whether the model handles regular commuting patterns differently from less regular weekend travel.

Second, divide evaluation days into **low-, medium-, and high-demand groups** based on their observed daily rental totals. I recommend reporting **WAPE alongside MAE** here, because a model can have smaller absolute errors on low-demand days while still making much larger *relative* errors.

Third, identify **abrupt hourly demand changes** using large absolute changes in the observed hourly rental counts, for example the top 10% of hour-to-hour changes. Then compare MAE for those hours against the remaining, more regular hours. This would help reveal whether the model smooths over sudden rises or drops.

For visualization, I would use clear grouped bars or compact comparison panels, with sample sizes and consistent units. I would avoid calling these differences causal. Also, if all forecasts begin at local midnight, **forecast step and time of day are confounded**, so a peak in error at a particular step does not by itself prove a horizon-length effect.

**Decision — Accepted.** Section 12 includes weekday/weekend errors, WAPE for low/medium/high observed-demand days, a forecast-step error profile, and a comparison of abrupt versus regular hourly demand changes. These behavioral plots go beyond simply reporting one overall accuracy score, in line with the assignment's Phase I goal.

**Selected observed results:** Weather-conditioned WAPE was **38.19%** for low-demand days versus **14.55%** for high-demand days. Weather-conditioned MAE was **168.59** during abrupt changes versus **114.87** during regular changes. The sharp-change groups are defined from *observed* within-day changes, which is a retrospective diagnostic rather than an advance warning signal.

---

## Interaction 8 — Using all eligible evaluation dates


**User:**

> I've run the notebook. Let's discuss our results.

**ChatGPT:**

I recommend treating the full **198-day December–June evaluation** as our final reference for the visualization results. The calendar-only MAE is approximately **116.63 rentals/hour**, and calendar plus past weather is approximately **116.49 rentals/hour**. That is a reduction of only **0.13 rentals/hour**—a negligible overall difference.

At the same time, the grouped rainfall results are more informative about behavior. The weather-conditioned MAE is about **107.44** in dry/trace conditions, **190.40** in light rain, and **244.30** in heavier rain. Adding historical weather does not reduce the error in the rainy groups.

I would therefore highlight two separate findings in the plots: **historical weather contributes very little to overall accuracy**, while **forecasting errors are strongly associated with rainfall categories**. We should preserve the sample sizes and avoid presenting this as proof that rainfall causes the errors.

**Decision — Accepted and validated using the final executed notebook.** The final evaluation uses **198 of 198 eligible days (4,752 hourly forecasts)**. Changing the evaluated origins also changed the measured overall weather effect relative to the 64-day sample; that is why the plots and presentation must be regenerated after changing the evaluation sample.

| Final evaluation metric | Calendar only | Calendar + past weather |
|---|---:|---:|
| MAE (rentals/hour) | 116.63 | 116.49 |
| Days with lower daily MAE after adding weather | — | 109 of 198 |
| p10–p90 empirical coverage (weather configuration) | — | 78.6% |

**Precipitation behavior (calendar MAE → weather MAE):**

| Evaluation condition | Evaluated hours | Calendar MAE | Weather MAE |
|---|---:|---:|---:|
| Dry / trace (≤0.2 mm) | 4,298 | 107.87 | 107.44 |
| Light rain (0.2–2 mm) | 355 | 187.98 | 190.40 |
| More rain (>2 mm) | 99 | 240.73 | 244.30 |

The two configurations have **near-identical overall performance**. Both struggle more in rainy observed hours, and adding historical weather slightly worsens MAE within the two rainy subgroups. These are associations, not evidence that rain alone causes the errors. Because some groups have small samples, these patterns should be checked alongside demand levels and time-of-day effects.

---

## AI4VIS — Decisions, failure patterns, and best practices

The interaction record above shows **how AI assistance influenced the visualization process**, not just the final graphics. 

### Which suggestions were accepted, refined, or rejected?

| Interaction | AI suggestion | User decision / status | Why it mattered |
|---|---|---|---|
| 1 | Start with May–June; show a 24-hour forecast and metrics | **Accepted as pilot; later refined** | A quick sanity check was useful, but seasonal coverage was insufficient. |
| 2 | Rolling-origin forecasts, signed errors, interval coverage, daily MAE and bias views | **Accepted** | Enabled analysis of errors over time, not a single illustrative forecast. |
| 3 | Keep the initial short period | **Refined by the user** to December–June | Included winter and broader observed weather conditions. |
| 4 | Signed-error Date × Hour heatmap | **Accepted** Implemented the weather-conditioned signed-error heatmap using weather_error and the diverging RdBu_r colormap. | Signed errors reveal systematic overprediction and underprediction across dates and hours, helping us investigate model bias rather than only error magnitude. |
| 5 | Temperature/rain scatterplots and grouped comparisons | **Refined** toward grouped MAE comparisons | Improved readability and directly compared the two input configurations. |
| 6 | User requested a chart showing successful weather improvements; AI proposed a hypothesis-neutral test | **Unsupported premise rejected; analytical framing refined** | A diagnostic chart must allow weather to help, hurt, or have no effect. |
| 7 | Weekday/weekend, demand-regime and abrupt-change plots | **Accepted** | Exposed behavior hidden by aggregate MAE. |
| 8 | Evaluate 64 sampled days only | **Refined** to all 198 eligible days | Produced more complete evidence and required updating all dependent conclusions. |

### Failure patterns observed during AI-assisted analysis

1. **Convenience sampling masked seasonal coverage.** The AI initially suggested May–June because it was easy to process. The user noticed that this omitted cold-weather conditions and expanded the period. *Improvement:* match temporal coverage to the research question before drawing general conclusions.
2. **Original requested inputs were not fully reflected in the implementation.** `month`, `is_holiday`, and `cloudcover` were requested but absent as explicit covariates in the executed configuration. *Improvement:* inspect actual model input tensors and document differences from the prompt.
3. **A single successful-looking forecast is not reliable evidence.** The 24-hour sanity check made the task understandable but could not establish overall performance. *Improvement:* use rolling forecasts and evaluate multiple conditions.
4. **Visualization choice can obscure the question.** A signed-error heatmap reveals direction while an absolute-error heatmap reveals magnitude; long date-by-hour matrices also become dense. *Improvement:* select encodings based on the analytical task and use aggregated companion views where helpful.
5. **AI responses can inherit an unsupported desired conclusion.** The request to show how historical weather *successfully* reduces error needed reframing, especially since weather did not help in rainy groups. *Improvement:* compare configurations neutrally and show negative as well as positive effects.
6. **Observed covariates can cause look-ahead leakage if used incorrectly.** The actual weather of a held-out forecast hour may explain that hour's error retrospectively, but must not enter the model as if it were forecast-known weather. *Improvement:* separate forecast inputs from evaluation-only metadata.
7. **Aggregate MAE hides demand scale and sample imbalance.** High-demand hours can have higher absolute errors; rainy-intensity groups have substantially different sample sizes. *Improvement:* report WAPE and group sizes, and avoid causal claims from subgroup comparisons.
8. **Changing the evaluation sample invalidates old headline figures.** Results from 64 sampled days should not be reported alongside 198-day figures as one experiment. *Improvement:* regenerate dependent plots, tables, and presentation numbers when the backtest selection changes.

### Emerging best practices for effective AI-assisted visualization

- **Start with an explicit behavior question.** Ask *when*, *where*, and *under what conditions* the model fails, rather than requesting only a performance bar chart.
- **Use AI proposals as hypotheses, not verified answers.** Require AI to explain what each visual would reveal and what it cannot establish.
- **Preserve human editorial control.** Challenge the AI's initial data scope, plot design, and assumptions; keep decisions and reasons in the interaction record.
- **Ground visualizations in actual executed outputs.** Check data columns, timestamps, sample sizes, and model configurations before reporting findings. This log includes five real output rows for reproducibility.
- **Compare like with like.** Keep forecast origins, target hours, context length, and horizon identical between calendar-only and calendar-plus-weather configurations.
- **Protect against leakage.** Past observed weather can be an inference input; actual future weather is evaluation metadata only unless using genuine available-at-forecast-time predictions.
- **Combine complementary views.** Use a representative forecast for intuition, daily MAE for variability, heatmaps for temporal structure, and grouped weather/demand comparisons for failure patterns.
- **Show uncertainty and limitations.** Report p10–p90 coverage, use MAE with relative metrics when comparing demand scales, and avoid treating observational associations as causal effects.
- **Mark proposed analyses separately from completed ones.** A suggested plot is not experimental evidence until run and inspected.



