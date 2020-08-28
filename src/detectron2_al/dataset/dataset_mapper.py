from detectron2.data import DatasetMapper
from detectron2.data.detection_utils import T, logging

__all__ = ['DatasetMapper']

def build_transform_gen_al(cfg, is_train):
    # Almost the same as detectron2.data.detection_utils.build_transform_gen
    # Yet there is no horizontal flip for the input image
    if is_train:
        min_size = cfg.INPUT.MIN_SIZE_TRAIN
        max_size = cfg.INPUT.MAX_SIZE_TRAIN
        sample_style = cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING
    else:
        min_size = cfg.INPUT.MIN_SIZE_TEST
        max_size = cfg.INPUT.MAX_SIZE_TEST
        sample_style = "choice"
    if sample_style == "range":
        assert len(min_size) == 2, "more than 2 ({}) min_size(s) are provided for ranges".format(
            len(min_size)
        )

    logger = logging.getLogger(__name__)
    tfm_gens = []
    tfm_gens.append(T.ResizeShortestEdge(min_size, max_size, sample_style))
    return tfm_gens


class DatasetMapperAL(DatasetMapper):

    def __init__(self, cfg, is_train=True):
        super(DatasetMapperAL, self).__init__(cfg, is_train)
        # Only substitute the tfm_gens with the al version, 
        # where there should not be any horizontal flips
        self.tfm_gens = build_transform_gen_al(cfg, is_train)
        self.crop_gen = None # Enforce no crop_generator 