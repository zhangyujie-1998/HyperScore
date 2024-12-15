import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import model.clip as clip
import numpy as np
from torch import einsum
from model.networks import TextEncoder, PromptLearner, HyperNet, TargetNet
from thop import profile

clip_vis = 'ViT-B/16'

class HyperScore(nn.Module):
    def __init__(self, device, args, quality_perspectives):
        super(HyperScore,self).__init__()
        self.device = device
        self.clip_model, _ = clip.load(clip_vis)
        self.clip_model = self.clip_model.to(torch.float32)
        self.dtype = self.clip_model.dtype
        self.prompt_learner = PromptLearner(self.device, args, quality_perspectives, self.clip_model)
        self.tokenized_prompt = self.prompt_learner.tokenized_prompts
        self.text_encoder = TextEncoder(self.clip_model)
        self.fc_quality= nn.Linear(512,224)
        self.fc_condition = nn.Linear(512,5488)
        self.hypernet = HyperNet()
        self.relu = nn.ReLU()
        for param in self.text_encoder.parameters():
            param.requires_grad = False
        
    def forward(self, texture_imgs, prompts):
        batch_size, num_views, channels, image_height, image_width = texture_imgs.shape
        texture_imgs = texture_imgs.reshape(-1, channels, image_height, image_width).type(self.dtype)
        prompts = clip.tokenize(prompts).to(self.device)
        prompt_embedding = self.clip_model.token_embedding(prompts).to(self.device)

        feature_texture = self.clip_model.encode_image_allpatch(texture_imgs)
        num_patches = feature_texture.shape[1]
        feature_texture =  feature_texture.reshape(batch_size, num_views * num_patches, -1)
        feature_prompt_eot, feature_prompt = self.text_encoder(prompt_embedding, prompts)

        prompt_learner = self.prompt_learner()
        feature_condition, _ = self.text_encoder(prompt_learner,self.tokenized_prompt)
        feature_condition_expand = feature_condition.repeat(batch_size,1,1)
        loss_dis = self.calculate_cos(feature_condition)

        sim_texture = einsum('b i d, b j d -> b j i', F.normalize(feature_prompt,dim=2), F.normalize(feature_texture,dim=2))
        sim_cond = einsum('b i d, b j d -> b j i', F.normalize(feature_prompt,dim=2), F.normalize(feature_condition_expand,dim=2))
        patch_weight = einsum('b i d, b j d -> b j i', sim_cond, sim_texture)
        patch_weight = F.softmax(patch_weight, dim=1)

        feature_condition = self.fc_condition(feature_condition)
        feature_condition = feature_condition.reshape(-1,112,7,7)

        score_list = torch.Tensor([]).to(self.device)
        for i in range(0,feature_condition.shape[0]):
            feature_texture_fused = torch.sum(feature_texture * patch_weight[:,:,i].unsqueeze(2), dim=1)
            feature_fused = torch.mul(feature_texture_fused, feature_prompt_eot) 
            feature_quality = self.fc_quality(feature_fused).unsqueeze(2).unsqueeze(3)
            param = self.hypernet(feature_condition[i])
     
            targetnet = TargetNet(param)
            score = targetnet(feature_quality)
            score_list = torch.cat([score_list, score],dim=1)
            
        out = {}
        out['score_list'] = score_list
        out['cos'] = loss_dis
        return out

    def calculate_cos(self, features):

        margin = torch.tensor([0.0], device=features.device)
        loss = torch.tensor([], device=features.device)
        num_cond, _ = features.shape
        for i in range(0,num_cond):
            for j in range(i+1,num_cond):
                cos = torch.maximum(F.cosine_similarity(features[i].unsqueeze(0),features[j].unsqueeze(0)),margin)
                loss = torch.cat([loss, cos])
        loss = torch.mean(loss) 

        return loss
 
    
