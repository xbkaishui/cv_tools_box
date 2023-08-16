import pytest
from loguru import logger
import timm

def load_model_with_ckpt():
    backbone = 'wide_resnet50_2'
    pre_trained = True
    ptcfg = timm.models.get_pretrained_cfg('wide_resnet50_2')
    ptcfg.url = None
    ckpt_file = '/root/.cache/ckpts/wide_resnet50_racm-8234f177.pth'
    # ptcfg.file = '/root/.cache/huggingface/hub/models--timm--wide_resnet50_2.racm_in1k'
    ptcfg.file = ckpt_file
    logger.info(ptcfg)
    m = timm.create_model(
                backbone,
                # pretrained_cfg=ptcfg,
                pretrained=pre_trained,
                features_only=True,
                exportable=True,
                out_indices=[1, 2, 3],
            )
    logger.info(m.classifier.weight)
    # logger.info(model)
    ...
    
if __name__ == '__main__':
    load_model_with_ckpt()