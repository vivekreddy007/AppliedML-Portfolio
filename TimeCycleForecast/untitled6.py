
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error,r2_score

df=pd.read_csv("train.csv")
df.head()

df["datetime"]=pd.to_datetime(df["datetime"])
df["dayofweek"]=df["datetime"].dt.dayofweek
df["month"]=df["datetime"].dt.month
df["hour"]=df["datetime"].dt.hour
df["weekend"]=df["dayofweek"].isin([5,6]).astype(int)
df["1_console_lag"]=df["count"].shift(1)
df["24_hours_back"]=df["count"].shift(24)
df["mean_of_last_24hours"]=df["count"].rolling(24).mean()
df=df.dropna().reset_index(drop=True)
df.head()

x=df.drop(["datetime","count","casual","registered"],axis=1)
y=df["count"]
categorical_columns=["season","weather","dayofweek","hour","month"]
numerical_columns=[i  for i in x.columns if i not in categorical_columns]

preprocessor=ColumnTransformer(transformers=[
    ("cat",OneHotEncoder(drop="first",handle_unknown="ignore"),categorical_columns),
    ("numerical","passthrough",numerical_columns)

])

tscv=TimeSeriesSplit(n_splits=5)
rmse_scores=[]
r2_scores=[]
scores=[]

for fold,(idx_train,idx_test) in enumerate(tscv.split(x),1):
  x_train=x.iloc[idx_train]
  x_test=x.iloc[idx_test]
  y_train=y.iloc[idx_train]
  y_test=y.iloc[idx_test]

  x_train=preprocessor.fit_transform(x_train)
  x_test=preprocessor.transform(x_test)

  model=Ridge(alpha=1.0)
  model.fit(x_train,y_train)

  y_pred=model.predict(x_test)

  mse=mean_squared_error(y_test,y_pred)
  rmse=np.sqrt(mse)
  r2=r2_score(y_test,y_pred)

  rmse_scores.append(rmse)
  r2_scores.append(r2)
  scores.append(model.score(x_test,y_test))

  print(f"{fold} --> RMSE : {rmse} and R2_score : {r2}")

print(f"Average RMSE : {np.mean(rmse_scores)}")
print(f"Average R2_scores : {np.mean(r2_scores)}")
print(f"Average model_scores : {np.mean(scores)}")

