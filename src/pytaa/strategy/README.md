# Portfolio Strategies
Below is a brief overview of assumptions and assets used in tactical asset allocation. Many of these can be combined or slightly tweaked to investor preferences without losing the core of the strategy. All strategy are relatively low frequency (monthly rebalanced).

### Ivy
- Mimicks the Harvard/Yale endowment strategies (SR: 0.8, MDD: 13%)
- Equal weights in `VTI`, `VEU`, `VNQ`, `AGG`, `DBC`, rebalanced end of month
- On last trading day, if price > 10m SMA then buy asset else put share in cash

### Robust Asset Allocation Balanced 
- Used by Gray by application of dual momentum (SR 0.8, MDD: -12%)
- 15% `VNQ`, 20% `IEF`, 20% `DBC`, 20% `MTUM`, 10% `IWD`, 10% `EFA`, 10% `EFV`, rebalanced end of month
- On last trading day, if 12m return > 12m return of `BIL` and if price > 12m SMA then invest 100% in asset, if both are not then allocate to cash, if only one condition then put 50% in asset and 50% in cash

### Diversified GEM Dual Momentum
- Dual momentum but slices portfolio into seven parts each with their own lookback period (SR: 0.8, MDD: 19%)
- 43% `SPY`, 30% `AGG`, 27% `EFA`, rebalanced end of month
- On last trading day, determine if return of `SPY` > `AGG` for each of the seven periods in 6m, 7m, ..., 12m. If true, then allocate the % to greater of `SPY` or `EFA`, else allocate to `AGG`.

### Vigilant Asset Allocation G12
- Allocates in Risk Assets (`SPY`, `IWM`, `QQQ`, `VGK`, `EWJ`, `EEM`, `VNQ`, `DBC`, `GLD`, `TLT`, `LQD`, and `HYG`) and Safe Assets (`IEF`, `LQD` and `BIL`), monthly rebalanced (SR: 1, MDD: 10%)
- On last trading day, calculate momentum score using lagged prices at 1m, 3m, 6m, 12m horizon: $Z=12\frac{p_t}{p_{t-1}}+4\frac{p_t}{p_{t-3}}+2\frac{p_t}{p_{t-6}}+\frac{p_t}{p_{t-12}}-19$
- Determine number of assets $n$ with negative $Z$, if $n>4$ allocate 100% in safe asset with highest momentum score, if $n=3$ put 75% in safest asset, remaining 25% is split equally in 5 risk assets with highest momentum, if $n=2$ put 50% in safest asset, 50% split evenly top 5 risk assets. Same logic applies for $n=1$ and $n=0$. 

### Vigilant Asset Allocation G4
- Allocates in risk (`SPY`, `EFA`, `EEM`, `AGG`) and safety (`LQD`, `IEF`, `BIL`) based on breadth momentum, rebalanced end of month (SR: 1, MDD: 16%)
- Similar to previous strategy, calculate $Z$ for each asset, if all risk assets have positive momentum then put 100% in highest scorer, if any asset has negative score then put 100% in defensive asset with highest score

### Kipnis Defensive Adaptive Asset Allocation
- Blends different TAA strategies together, uses dual momentum and adaptive asset allocation (minimum variance optimization) to determine weights, rebalanced end of month (SR: 1.1, MDD: 9%)
- Uses "canary universe" (`EEM`, `AGG`) to allocate into defensives
- Investment universe (`SPY`, `VGK`, `EWJ`, `EEM`, `VNQ`, `RWX`, `IEF`, `TLT`, `DBC`, `GLD`) and crash protection (`IEF`, Cash)
- Calculate $Z$ for investment universe, select top 5 positives, weight assets using covariance matrix with 1m vol estimates: $\rho_i^*=\frac{1}{19}\left(12\rho_1 + 4\rho_3 + 2\rho_6 +\rho_{12}\right)$
- If both canary assets have positive $Z$, .then invest 100% in top 5 assets in universe. If one canary assets has $Z>0$, invest 50% in crash protection and 50% in investment universe. If neither have positive $Z$, put 100% in crash protection. Allocate to cash when `IEF` does not have $Z>0$.

### Generalized Protective Momentum
- Allocates to risk assets (`SPY`, `QQQ`, `IWM`, `VGK`, `EWJ`, `EEM`, `VNQ`, `DBC`, `GLD`, `HYG`, `LQD`) and safety assets(`BIL`, `IEF`), rebalanced monthly (SR: 1, MDD: 10%)
- On last trading day, calculate average return $r_{i,t}$ for $t\in[1,3,6,12]$ months, the 12 month correlation $\rho_i$  between asset and equal weighted risk assets
- Calculate score $M=r_i(1-\rho_i)$ and determine $n$ assets where $M>0$. If $n\leq6$ then fully invest in safety asset with largest $M$. If $n\gt6$ then invest $(12-n)/6$ percent in safety and equally allocate remainder to assets with highest $M$.

### Trend is Our Friend
- Blends risky parity and trend following to determine asset allocation, rebalanced monthly (SR: 0.8, MDD: 9%). On avg holds 20% equities, 26% bonds, 54% in cash, commodities and real estate.
- On last trading day, calculate portfolio weights using 12m risk parity from AQR: $w_{t,i}=k_t\hat\sigma_{t,i}^{-1}$ where $k=1 / \sum_i \hat\sigma_{t,i}^{-1}$ is a leverage control variable, constant across assets. The volatility $\hat\sigma_{t,i}$ is estimated using 3y excess returns.
- Invest in assets if price is above 10m SMA and else allocate to cash

### Global Tactical Asset Allocation
- Detailed by Meb Faber, rebalanced monthly (SR: 08, MDD: 13%).
- Portfolio: 5% `IWD`, 5% `MTUM`, 5% `IWN`, 5% `DWAS`, 10% `EFA`, 10% `EEM`, 5% `IEF`, 5% `BWX`, 5% `LQD`, 5% `TLT`, 10% `DBC`, 10% `GLD`, 20% `VNQ`)
- On last trading day, buy assets above 10m SMA, else allocate to cash.

### Defensive Asset Allocation
- Uses a momentum approach (breadth score) that is skewed towards recent months, rebalanced monthly (SR: 1.1, SR: 12%)
- Universe (`SPY`, `IWM`, `QQQ`, `VGK`, `EWJ`, `EEM`, `VNQ`, `DBC`, `GLD`, `TLT`, `HYG`, `LQD`),  protective (`SHY`, `IEF`, `LQD`) and canary (`EEM`, `AGG`)
- Calculate $n=\sum_1^2 c_{\mathbb 1_{Z<0}}$ for canary , if $n=2$ then 100% in protective asset with highest $Z$, if $n=1$ split equally across risk and protective, if $n=0$ invest equally across 6 risk assets with highest $Z$

### Protective Asset Allocation
- Uses dual momentum and has a very aggressive protection mechanism, rebalanced monthly (SR: 0.9, MDD: 10%)
- Universe: 51% `IEF`, 6% `IWM`, 6% `QQQ`, 5% `VNQ`, 5% `SPY`, 5% `VGK`, 5% `EEM`, 4% `EWJ`, 3% `DBC`, 3% `TLT`, 3% `GLD`, 2% `HYG`, 2% `LQD`
- On the last trading day, calculate $S=\frac{p_t}{SMA(12)}-1$ and calculate $n$ assets with positive score. If $n\lt6$ invest in sage asset only (`IEF`). Else, invest remainder of $\frac{12-n}{6}$ into top 6 risk assets, equally weighted.

### Adaptive Asset Allocation
- Pioneered by ReSolve Asset Management, monthly rebalanced (SR: 0.9, MDD: 11%)
- Average weighting: 30% `IEF`, 17% `SPY`, 10% `VNQ`, 8% `RWX`, 7% `DBC`, 7% `EEM`, 6% `VGK`, 6% `TLT`, 5% `GLD`, 4% `EWJ`
- On last trading day, buy top 5 assets based on 6m total return, weighted according th their 6m correlation and 1m volatility

### GEM Dual Momentum
- Invest in one asset at a time, rebalanced monthly (SR: 0.8, MDD: -20%)
- Avg. Assets: 45% `SPY`, 28% `AGG`, 27% `EFA`
- If 12m return of `SPY` > `BIL` then invest 100% in `SPY` if `SPY` > `EFA`, else in `EFA`. Otherwise invest in `AGG`.

### Quint Switching Filtered
- Pair switching model using 6 ETFs, rebalanced monthly (SR: 0.8, MDD: 17%)
- Defensive asset (75% `IEF`) and risky assets  (10% `QQQ`, 8% `EEM`, 4% `EFA`, 2% `TLT`, 1% `SPY`)
- On last trading day: if any risk asset has negative 3m return then put 100% in `IEF`, else put 100% in asset with highest 3m return.

### Composite Dual Momentum
- Similar to permanent portfolio but each 25% slice is adjusted monthly based on trend and cross-sectional momentum (SR: 0.9, MDD: 11%)
- Equities (`SPY`, `EFA`), Bonds (`HYG`, `LQD`), Real Estate (`VNQ`, `REM`), Stress (`GLD`, `TLT`)
- On last trading day: calculate 12m return for each class, invest in asset with highest 12m return if higher than that of `BIL`. Else put slice in cash.

### HMM Regime Switching
- Define market turbulence as Mahalanobis distance of sector and G10 FX returns (10y for equities, 3y for FX)
- Calibrate 2-regime HMM to both turbulences, inflation, economic growth separately
- Tilts weights given predicted regime in next period, rebalanced monthly (SR: 1, MDD: 33%)
- Risk assets: global stocks - bonds, small-caps, equity momentum, equity HFs, EM-DM, credit spread, HY spread, 2s10s, EM spread, FX carry
- Defensives: Gold - cash, TIPS - nominals, US cyclicals spread, FX value
- Default weights are 10% for risk and 0% for defensives: if turbulence predicts change then decrease risk by 5% and increase FX value by 10%. Same goes for recession and inflation. Weights in the Kritzman paper change given scenario.

### Robeco Dynamic Strategic Asset Allocation
- Allocation based on four regimes (Expansion, Peak, Recession, Recovery) using credit spread, earnings yield, ISM and unemployment
- Assets: 25% large caps, 25% treasuries, 5% small caps, 5% credit, 5% commodities
- TAA optimizes above static weights for maximum expected return using tracking error constraint (from static weights). Others do in-sample optimization over 15 years.
- See [here](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1343063) 

### Lyxor Strategic Asset Allocation
- See [here](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2154021)