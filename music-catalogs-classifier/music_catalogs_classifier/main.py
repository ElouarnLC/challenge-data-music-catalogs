import datetime

# Save 5% of the rows of the predictions as csv files in the data folder in predictions folder with timestamp
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

np.savetxt(
    f"../data/predictions/y_true_{timestamp}.csv",
    y_true[: int(0.05 * len(y_true))],
    delimiter=",",
)
np.savetxt(
    f"../data/predictions/y_pred_{timestamp}.csv",
    y_pred[: int(0.05 * len(y_pred))],
    delimiter=",",
)