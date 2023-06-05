# Tactical Asset Allocation (pyTAA)
This package features a set of tools to backtest systematic, low-frequency strategies and compare various tactical asset allocation (TAA) programs. Asset allocation in general is about finding a balance between risk and reward whilst accounting for investment goals, time frames and risk preferences. Asset allocation often comes in three forms: Strategic, Tactical and Dynamic. Tactical asset allocation takes a more active investment approach and can be characterized as follows:
- Active management of portfolio strategy that shifts allocation based on market trends or economic conditions (e.g. stocks, bonds, cash, commodities)
- Benefits are diversification, drawdown control and overall risk management (MDD <20%, SR > 0.75)
- Typically strategies are absolute return (returns uncorrelated to markets/betas), relative return (beat benchmark) and total return (targeted return)

## Package
This package is a current WIP and I update it whenever I find time. The goal of this package is to demonstrate different TAA techniques and how they perform through different economic cycles. Whilst each strategy is unique, they all share certain common characteristics, e.g. momentum signals or dual momentum sorts. The package tries to break down these characteristics into individual modules that can be used on their own.

## Installation
Eventually, this will be `pip` installable via:

```
pip install pytaa
```

## Portfolio Strategies
For an overview of various TAA strategies please have a look at the `src/taa/strategy/README.md` file. You can also check out the reference list at the bottom for more information. Many of these strategies have been taken from [this](https://twitter.com/WifeyAlpha/status/1502576435134877701) source.

## References
- Antonacci, Gary. ‘Risk Premia Harvesting Through Dual Momentum’. SSRN Scholarly Paper. Rochester, NY, 1 October 2016. [https://doi.org/10.2139/ssrn.2042750](https://doi.org/10.2139/ssrn.2042750).
- Blitz, David, and Pim van Vliet. ‘Dynamic Strategic Asset Allocation: Risk and Return Across Economic Regimes’. SSRN Scholarly Paper. Rochester, NY, 1 July 2009. [https://doi.org/10.2139/ssrn.1343063](https://doi.org/10.2139/ssrn.1343063).
- Butler, Adam, Mike Philbrick, Rodrigo Gordillo, and David Varadi. ‘Adaptive Asset Allocation: A Primer’. SSRN Scholarly Paper. Rochester, NY, 31 May 2012. [https://doi.org/10.2139/ssrn.2328254](https://doi.org/10.2139/ssrn.2328254).
- Eychenne, Karl, Stéphane Martinetti, and Thierry Roncalli. ‘Strategic Asset Allocation’. SSRN Scholarly Paper. Rochester, NY, 1 March 2011. [https://doi.org/10.2139/ssrn.2154021](https://doi.org/10.2139/ssrn.2154021).
- Keller, Wouter J., and Jan Willem Keuning. ‘Breadth Momentum and the Canary Universe: Defensive Asset Allocation (DAA)’. SSRN Scholarly Paper. Rochester, NY, 12 July 2018. [https://doi.org/10.2139/ssrn.3212862](https://doi.org/10.2139/ssrn.3212862).
- Keller, Wouter J., and Jan Willem Keuning. ‘Protective Asset Allocation (PAA): A Simple Momentum-Based Alternative for Term Deposits’. SSRN Scholarly Paper. Rochester, NY, 5 April 2016. [https://doi.org/10.2139/ssrn.2759734](https://doi.org/10.2139/ssrn.2759734).
- Kritzman, Mark, Sébastien Page, and David Turkington. ‘Regime Shifts: Implications for Dynamic Strategies’. _Financial Analysts Journal_, 2012, 18.
- WifeyAlpha, Tactical Asset Allocation, https://twitter.com/WifeyAlpha/status/1502576435134877701

## Disclaimer
The content is for informational purposes only, you should not construe any such information or other material as legal, tax, investment, financial, or other advice.
