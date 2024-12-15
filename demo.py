import torch
import argparse
from torchvision import transforms
from HyperScore import HyperScore
from projection.util_projection import MeshRenderer3D

def parse_args():
    args = argparse.Namespace()
    args.n_ctx = 12
    args.ctx_init = ''
    args.class_token_position = 'front'
    args.csc = True  
    args.prec = "fp32"  
    args.subsample_classes = "all" 
    return args

def convert_tensor(img_list):
    transformations = transforms.Compose([
        transforms.Resize(224),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img_transformed = torch.zeros([6, 3, 224, 224])

    for i in range(6):
        img = img_list[i]
        img = img.convert('RGB')
        img = transforms.ToTensor()(img)
        img = transformations(img)
        img_transformed[i] = img
    img_transformed = img_transformed.unsqueeze(0)
    return img_transformed

def load_model(device, args, model_path):
    quality_perspectives = ['alignment quality', 'geometry quality', 'texture quality', 'overall quality']
    model = HyperScore(device=device, args=args, quality_perspectives=quality_perspectives)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model

def main():
    args = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    renderer = MeshRenderer3D()

    # Load the pre-trained model
    model_path = 'checkpoint/model.pth'
    model = load_model(device, args, model_path)

    # Render the texture mesh
    obj_path = "demo/A_canned_Coke/model.obj"
    img_list = renderer.render_views_eval(obj_path)
    img = convert_tensor(img_list)
    prompt = 'A canned Coke'

    # Evaluation
    predictions = []
    with torch.no_grad(): 
        img = img.to(device)
        out = model(img, prompt)
        predictions.append(out['score_list'].reshape(1, 4))

    # Print results
    predictions_tensor = torch.cat(predictions, dim=0)
    predictions_cpu = predictions_tensor.cpu().numpy()
    perspective_list = ['Alignment', 'Geometry', 'Texture', 'Overall']
    print("Predicted quality scores:")
    for i, perspective in enumerate(perspective_list):
        print(f"{perspective}: {predictions_cpu[0, i]}")

  

if __name__ == "__main__":
    main()
