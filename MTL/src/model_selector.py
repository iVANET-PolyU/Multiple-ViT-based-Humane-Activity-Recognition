from multi_vision_transformer import ViT, mlp_head
from MultiResNet import ResNet, BinaryDecoder, BasicBlock
import torch.nn as nn

def get_model(config, device):
    model_type = config['model']
    if 'ViT' in model_type:
        model = {}
        model['rep'] = ViT(dim=config['dim'], depth=config['depth'], heads=config['heads'],
                           mlp_dim=config['mlp_dim'], pool='cls', dim_head=300,
                           dropout=0., emb_dropout=0.)
        if config['parallel']:
            model['rep'] = nn.DataParallel(model['rep'])
        model['rep'].to(device)
        for t in config['tasks']:
                model[t] = mlp_head(dim=config['dim'])
                if config['parallel']:
                    model[t] = nn.DataParallel(model[t])
                model[t].to(device)
        return model

    if 'ResNet' in model_type:
        model = {}
        model['rep'] = ResNet(BasicBlock, [2, 2, 2, 2])
        if config['parallel']:
            model['rep'] = nn.DataParallel(model['rep'])
        model['rep'].cuda()
        for t in config['tasks']:
            model[t] = BinaryDecoder()
            if config['parallel']:
                model[t] = nn.DataParallel(model[t])
            model[t].cuda()
        return model

