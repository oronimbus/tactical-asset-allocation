"""List strategies in serializable format."""


STRATEGIES = [
    {
        "tag": "IVY",
        "name": "Ivy",
        "riskAssets": ["VTI", "VEU", "VNQ", "AGG", "DBC"],
        "safeAssets": ["BIL"],
        "weights": [0.2, 0.2, 0.2, 0.2, 0.2],
        "frequency": "M",
    },
    {
        "tag": "RAAB",
        "name": "Robust Asset Allocation Balanced",
        "riskAssets": ["VNQ", "IEF", "DBC", "MTUM", "IWD", "EFA", "EFV"],
        "safeAssets": ["BIL"],
        "weights": [0.15, 0.2, 0.2, 0.2, 0.1, 0.1, 0.1],
        "frequency": "M",
    },
    {
        "tag": "DGADM",
        "name": "Diversified GEM Dual Momentum",
        "riskAssets": ["SPY", "AGG", "EFA"],
        "weights": [0.43, 0.3, 0.27],
        "frequency": "M",
    },
    {
        "tag": "VAAG12",
        "name": "Vigilant Asset Allocation G12",
        "riskAssets": [
            "SPY",
            "IWM",
            "QQQ",
            "VGK",
            "EWJ",
            "EEM",
            "VNQ",
            "DBC",
            "GLD",
            "TLT",
            "LQD",
            "HYG",
        ],
        "safeAssets": ["IEF", "LQD", "BIL"],
        "frequency": "M",
    },
    {
        "tag": "VAAG4",
        "name": "Vigilant Asset Allocation G4",
        "riskAssets": ["SPY", "EFA", "EEM", "AGG"],
        "safeAssets": ["LQD", "IEF", "BIL"],
        "frequency": "M",
    },
    {
        "tag": "KDAAA",
        "name": "Kipnis Defensive Adaptive Asset Allocation",
        "riskAssets": [
            "SPY",
            "VGK",
            "EWJ",
            "EEM",
            "VNQ",
            "RWX",
            "IEF",
            "TLT",
            "DBC",
            "GLD",
        ],
        "safeAssets": ["IEF"],
        "canaryAssets": ["EEM", "AGG"],
        "frequency": "M",
    },
    {
        "tag": "KDAAA",
        "name": "Generalized Protective Momentum",
        "riskAssets": [
            "SPY",
            "QQQ",
            "IWM",
            "VGK",
            "EWJ",
            "EEM",
            "VNQ",
            "DBC",
            "GLD",
            "HYG",
            "LQD",
        ],
        "safeAssets": ["BIL", "IEF"],
        "frequency": "M",
    },
    {
        "tag": "TIOF",
        "name": "Trend is Our Friend",
        "riskAssets": ["SPY", "AGG", "DBC", "VNQ"],
        "safeAssets": ["BIL"],
        "frequency": "M",
    },
    {
        "tag": "GTAA",
        "name": "Global Tactical Asset Allocation",
        "riskAssets": [
            "IWD",
            "MTUM",
            "IWN",
            "DWAS",
            "EFA",
            "EEM",
            "IEF",
            "BWX",
            "LQD",
            "TLT",
            "DBC",
            "GLD",
            "VNQ",
        ],
        "safeAssets": ["BIL"],
        "weights": [
            0.05,
            0.05,
            0.05,
            0.05,
            0.1,
            0.1,
            0.05,
            0.05,
            0.05,
            0.05,
            0.1,
            0.1,
            0.2,
        ],
        "frequency": "M",
    },
    {
        "tag": "DAA",
        "name": "Defensive Asset Allocation",
        "riskAssets": [
            "SPY",
            "QQQ",
            "VGK",
            "EWJ",
            "EEM",
            "VNQ",
            "IWM",
            "DBC",
            "TLT",
            "HYG",
            "TLT",
            "LQD",
            "GLD",
        ],
        "safeAssets": ["SHY", "IEF", "LQD"],
        "canaryAssets": ["EEM", "AGG"],
        "frequency": "M",
    },
    {
        "tag": "PAA",
        "name": "Protective Asset Allocation",
        "riskAssets": [
            "IEF",
            "IWM",
            "QQQ",
            "VNQ",
            "SPY",
            "VGK",
            "EEM",
            "EWJ",
            "DBC",
            "TLT",
            "GLD",
            "HYG",
            "LQD",
        ],
        "safeAssets": ["IEF"],
        "weights": [
            0.51,
            0.06,
            0.06,
            0.05,
            0.05,
            0.05,
            0.05,
            0.04,
            0.03,
            0.03,
            0.03,
            0.02,
            0.02,
        ],
        "frequency": "M",
    },
    {
        "tag": "AAA",
        "name": "Adaptive Asset Allocation",
        "riskAssets": [
            "IEF",
            "SPY",
            "VNQ",
            "RWX",
            "DBC",
            "EEM",
            "VGK",
            "TLT",
            "GLD",
            "EWJ",
        ],
        "weights": [0.3, 0.17, 0.1, 0.08, 0.07, 0.07, 0.06, 0.06, 0.05, 0.04],
        "frequency": "M",
    },
    {
        "tag": "GDM",
        "name": "GEM Dual Momentum",
        "riskAssets": ["SPY", "AGG", "EFA"],
        "canaryAssets": ["BIL"],
        "frequency": "M",
    },
    {
        "tag": "QSF",
        "name": "Quint Switching Filtered",
        "riskAssets": ["QQQ", "EEM", "EFA", "TLT", "SPY"],
        "safeAssets": ["IEF"],
        "frequency": "M",
    },
    {
        "tag": "CDM",
        "name": "Composite Dual Momentum",
        "riskAssets": ["SPY", "EFA", "HYG", "LQD", "VNQ", "REM", "GLD", "TLT"],
        "assetClasses": [
            "Equity",
            "Equity",
            "Bonds",
            "Bonds",
            "Real Estate",
            "Real Estate",
            "Stress",
            "Stress",
        ],
        "safeAssets": ["BIL"],
        "frequency": "M",
    },
    {
        "tag": "HMMRS",
        "name": "HMM Regime Switching",
        "riskAssets": [],
        "safeAssets": [],
        "frequency": "M",
    },
    {
        "tag": "RDSAA",
        "name": "Robeco Dynamic Strategic Asset Allocation",
        "riskAssets": ["VV", "IEF", "VB", "LQD", "DBC"],
        "weights": [0.25, 0.25, 0.05, 0.05, 0.05],
        "frequency": "M",
    },
]