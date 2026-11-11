# AI4VIS — Phase I Midpoint Reflection

**Project:** Jurassic Bikes — TiRex-2 and Washington, DC Capital Bikeshare  
**AI-aided pair:** İlknur Ulaş and David Kopper  
**Phase I question:** How does the model behave?  
**Interaction record:** [`ai-interactions/Jurassic_Bikes_Phase_I_AI_interaction.md`](ai-interactions/Jurassic_Bikes_Phase_I_AI_interaction.md)

## How AI assistance influenced our analysis

We used ChatGPT as a collaborator in designing visualizations for the **external input–output behavior** of a pretrained TiRex-2 forecasting model. Our central comparison was between **calendar-only** forecasts and forecasts that also received **historical weather** covariates. AI assistance helped us turn an initial actual-versus-predicted line plot into a broader diagnostic workflow: rolling-origin forecasts, daily MAE comparisons, a signed-error Date × Hour heatmap, and weather- and demand-conditioned error analysis.

The most useful AI contribution was not any individual plot. It was the ability to propose complementary views and explain what each could reveal. For instance, a 24-hour forecast illustrates the task, daily MAE shows variability between dates, and a signed-error heatmap reveals systematic overprediction and underprediction. We kept these questions separate rather than treating one aggregate accuracy score as a complete description of the model.

## Decisions we accepted, challenged, or refined

We accepted the suggestion to compare the two model configurations over identical forecast windows using **336 hours of context and a 24-hour prediction horizon**. We also retained signed forecast errors and p10–p90 interval coverage so that we could examine bias and uncertainty, not just average error.

The most important **correction** concerned the evaluation period. AI initially proposed **May–June 2024** as a manageable pilot. After examining the daily-error visualizations, we recognized that these months offered limited cold-weather coverage. We extended the data to **December 2023–June 2024**, retaining the experimental design but checking timestamp alignment and the daylight-saving transition. Later, we replaced a 64-day evaluation sample with **all 198 eligible days**, requiring the plots and reported metrics to be updated.

We also refined the visualization question. Although an absolute-error heatmap could show where errors are largest, we ultimately **accepted the signed-error heatmap** of the weather-conditioned forecast, using a diverging red–blue scale to distinguish overprediction from underprediction. For adverse-weather comparisons, we rejected the assumption that supplying historical weather must improve predictions. Instead, we compared both configurations within observed precipitation categories and allowed for positive, negative, or negligible effects.

## Failure patterns and limitations of AI assistance

- **Convenient defaults may narrow the research question.** The initial May–June sample was computationally manageable but did not provide the seasonal coverage we wanted.
- **Generated implementation can diverge from the prompt.** The original request included explicit month, holiday, and cloud-cover covariates and RMSE reporting; these were not all present in the final notebook. Checking actual model inputs and outputs was essential.
- **A plausible visualization can still be misleading.** A single forecast cannot represent the full evaluation period, and a dense daily heatmap can be difficult to read. Plot type, color encoding, and sample size must fit the question.


## What the completed evaluation taught us

On **198 evaluated days (4,752 hourly forecasts)**, calendar-only MAE was approximately **116.63 rentals/hour**, compared with **116.49 rentals/hour** with historical weather: a negligible difference. Historical weather lowered daily MAE on **109 of 198 days**. Meanwhile, forecast errors were substantially higher in observed rainy hours, and the analysis revealed difficulties around abrupt changes in demand. The weather-conditioned p10–p90 interval covered approximately **78.6%** of observed hourly counts.

These results illustrate why our visualizations needed to go beyond a headline MAE comparison. The most informative finding was not that historical weather improved the overall forecast—it barely did—but that **model errors varied across conditions and demand regimes**.


Our main lesson is to treat AI recommendations as **proposals to verify**, not as verified findings. 