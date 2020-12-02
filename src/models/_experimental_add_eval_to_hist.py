from utils_func import update_history_add_eval

metrics = {"mae": 0.87, "mse": 1.27}
models_hist_path = "models/models_history.json"
model_id = "ujwlhl"

update_history_add_eval(
    models_hist_path, model_id=model_id, model_name=None, metrics=metrics
)
