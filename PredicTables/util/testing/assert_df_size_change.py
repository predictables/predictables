from PredicTables.util import to_pl_df


def assert_df_size_change(df, df1, row=0, col=0):
    df = to_pl_df(df)
    df1 = to_pl_df(df1)
    # if isinstance(df, pl.LazyFrame):
    #     df = df.collect()
    # if isinstance(df1, pl.LazyFrame):
    #     df1 = df1.collect()

    try:
        assert df.shape[0] + row == df1.shape[0]
        assert df.shape[1] + col == df1.shape[1]

        if col > 0:
            if col > 10:
                print(f"Added {col} columns.")
            else:
                print(f"Added columns:\n\n{set(df1.columns) - set(df.columns)}")

        return True
    except:
        assert df.shape[0] + row == df1.shape[0], f"rows before: {df.shape[0]}\nrows after: {df1.shape[0]}\n\
expected row change: {row}\nactual row change: {df1.shape[0] - df.shape[0]}"

        assert df.shape[1] + col == df1.shape[1], f"columns before: {df.shape[1]}\ncolumns after: {df1.shape[1]}\n\
expected column change: {col}\nactual column change: {df1.shape[1] - df.shape[1]}\
\n\nLikely due to: {set(df1.columns) - set(df.columns) if df1.shape[1] > df.shape[1] else set(df.columns) - set(df1.columns)}"
