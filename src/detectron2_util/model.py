from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from detectron2.modeling.meta_arch.rcnn import ProposalNetwork, GeneralizedRCNN


@META_ARCH_REGISTRY.register()
class ActiveLearningRCNN(GeneralizedRCNN):

    def forward_al(self, batched_inputs, do_postprocess=True):

        """
        Run inference on the given inputs with active learning.

        Returns:
            same as in :meth:`forward`.
        """
        assert not self.training

        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)

        if self.proposal_generator:
            proposals, _ = self.proposal_generator(images, features, None)
        else:
            assert "proposals" in batched_inputs[0]
            proposals = [x["proposals"].to(self.device) for x in batched_inputs]

        detection_results, detection_scores = self.roi_heads._forward_al(features, proposals)
    
        if do_postprocess:
            detection_results = GeneralizedRCNN._postprocess(detection_results, batched_inputs, images.image_sizes) 
        
        return detection_results, detection_scores
