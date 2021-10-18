def create_model(opt):
    model = None
    print(opt.model)
    if opt.model == 'resvit_many':
        from .resvit_many import ResViT_model
        model = ResViT_model()
    elif opt.model == 'resvit_one':
        from .resvit_one import ResViT_model
        model = ResViT_model()
    elif opt.model == 'test':
        assert(opt.dataset_mode == 'single')
        from .test_model import TestModel
        model = TestModel()
    else:
        raise NotImplementedError('model [%s] not implemented.' % opt.model)
    model.initialize(opt)
    print("model [%s] was created" % (model.name()))
    return model
