import os
import json
from projection.util_projection import MeshRenderer3D
from tqdm import tqdm
prompt_path = "data/MATE-3D/prompt_MATE_3D.json"
data_path = 'data/MATE-3D/'

save_root_path = 'data/projection/'
os.makedirs(save_root_path, exist_ok=True)

model_list = ["dreamfusion","magic3d", "sjc", "textmesh", "3dtopia", "consistent3d", "latentnerf", "one2345++"]

with open(prompt_path, 'r', encoding='latin1') as file:
    content = file.read()
    prompt_lists = json.loads(content.strip())
renderer = MeshRenderer3D()

for model_name in model_list:
    for i,prompt in tqdm(enumerate(prompt_lists),total=len(prompt_lists), smoothing=0.9, leave=False):
        prompt_name = prompt.replace(" ", "_")
        save_path = os.path.join(save_root_path, model_name, prompt_name)
        os.makedirs(save_path, exist_ok=True)
        obj_path = os.path.join(data_path, model_name, prompt_name, "model.obj")
        renderer.render_views(obj_path,save_path)

        
