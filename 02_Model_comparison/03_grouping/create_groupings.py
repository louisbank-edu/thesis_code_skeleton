# What must happen here: reading in the datafile and then create different groupings,
# saving them as grouping_xxx.csv, which then the forecasting is based on
#it should just save a csv file with columns "Material" and GroupID
import pandas as pd

data_file = "PATH_TO_YOUR_DATAFILE/monthly_complete.csv"
df = pd.read_csv(data_file)



# globale group
df["GroupID"] = 1
df = df[["Material", "GroupID"]].drop_duplicates()
df.to_csv("PATH_TO_OUTPUT/grouping_all.csv", index=False)


# material-wise group( also jedes material seine eigene unique gruppe)
df["GroupID"] = df.groupby("Material").ngroup() + 1
df = df[["Material", "GroupID"]].drop_duplicates()
df.to_csv("PATH_TO_OUTPUT/grouping_materialwise.csv", index=False)


# grouping based on "intermittent", "erratic", "regular" and "lumpy", getting the group id from the classification file
classification_path = "PATH_TO_OUTPUT/intermittent_demand_classification.csv"
class_df = pd.read_csv(classification_path)
# only keep the columns "Material" and "Demand_Type"
class_df = class_df[["Material", "Demand_Type"]]
# rename the columns to "Material" and "GroupID", so rename "Demand_Type" to "GroupID"
class_df = class_df.rename(columns={"Demand_Type": "GroupID"})
# drop duplicates
class_df = class_df.drop_duplicates()
class_df.to_csv("PATH_TO_OUTPUT/grouping_adi_cv2.csv", index=False)
