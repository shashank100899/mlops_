import mlflow 
from mlflow import MlflowClient

def best_run(experiment_name): 
    experiment_id = [mlflow.get_experiment_by_name(name = experiment_name).experiment_id]
    q = "metrics.mae  < 0.5"
    runs = MlflowClient().search_runs(experiment_ids=experiment_id , filter_string = q)
    mse_low = None
    #print(runs)
    for run in runs:
        if mse_low == None or run.data.metrics["mse"] < mse_low:
            mse_low = run.data.metrics["mse"]
            best_run_path = run.info.artifact_uri
            best_run_id = run.info.run_id
    return best_run_path , best_run_id


path , run_id = best_run("Mlflow_demo")


#method-1 to predict 
loaded_model = mlflow.sklearn.load_model(path+"/model")
print(loaded_model,end="\n\n")

#method-2 to predict
loaded_model_method2 = mlflow.pyfunc.load_model("runs:/"+run_id+"/model")
print(loaded_model_method2)
