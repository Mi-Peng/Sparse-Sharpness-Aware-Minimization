from utils.register import Registry 

MODELS_REGISTRY = Registry("Models")

def build_model(args):
    model = MODELS_REGISTRY.get(args.model)(args)
    model = model.cuda()
    return model