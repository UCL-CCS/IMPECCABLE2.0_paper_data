from atomsci.ddm.pipeline import compare_models as cm
import shutil

model_store_path = "./model_store"

perf_df = cm.get_filesystem_perf_results(model_store_path, pred_type='regression')

print(perf_df.columns)

top_model=perf_df.sort_values(by="best_valid_r2_score", ascending=False).iloc[0]
top_model_file_path = top_model['model_path']

shutil.copy(top_model_file_path, "../selected_trained_model.tar.gz")