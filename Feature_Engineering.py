def add_mmm_features(df):
    # 1. Lag Features (3-12 month lags for marketing variables)
    for lag in [1, 2, 3, 6, 12]:
        for col in df.filter(regex='paid_').columns:
            df[f'{col}_lag_{lag}'] = df[col].shift(lag)

    # 2. Rolling Averages (4-8 week windows)
    for window in [4, 8, 12]:
        for col in df.filter(regex='paid_').columns:
            df[f'{col}_rolling_{window}'] = df[col].rolling(window).mean()

    # 3. Adstock Transformation (Decay effect)
    def adstock(x, decay=0.5):
        return [sum(x[:i + 1] * (decay ** (i - np.arange(i + 1))) for i in range(len(x))]

        for col in df.filter(regex='paid_').columns:
            df[f'{col}_adstock'] = adstock(df[col].values)

    return df