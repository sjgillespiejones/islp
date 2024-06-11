def success_rate(table):
    true_negatives = table.loc[0, 0]
    true_positives = table.loc[1, 1]
    return (true_positives + true_negatives) / (table.loc[0].sum() + table.loc[1].sum())

def error_rate(table):
    success = success_rate(table)
    return 1 - success