# Ultralytics YOLO 🚀, AGPL-3.0 license

from ultralytics.engine.results import Results
from ultralytics.models.yolo.detect.predict import DetectionPredictor
from ultralytics.utils import DEFAULT_CFG, ops


class SegmentationPredictor(DetectionPredictor):
    """
    A class extending the DetectionPredictor class for prediction based on a segmentation model.

    Example:
        ```python
        from ultralytics.utils import ASSETS
        from ultralytics.models.yolo.segment import SegmentationPredictor

        args = dict(model='yolov8n-seg.pt', source=ASSETS)
        predictor = SegmentationPredictor(overrides=args)
        predictor.predict_cli()
        ```
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """Initializes the SegmentationPredictor with the provided configuration, overrides, and callbacks."""
        super().__init__(cfg, overrides, _callbacks)
        self.args.task = 'segment'

    def postprocess(self, preds, img, orig_imgs):
        """Applies non-max suppression and processes detections for each image in an input batch."""
        self.args.conf = 0.35
        # import cv2
        # import numpy as np
        # img_ = preds[0][:, 4, :3840].reshape(48, 80).numpy()
        # img_ = cv2.resize(img_, (1920, 1080))
        # cv2.imshow("rider", img_[..., None])
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        p = ops.non_max_suppression(preds[0],
                                    self.args.conf,
                                    self.args.iou,
                                    agnostic=self.args.agnostic_nms,
                                    max_det=self.args.max_det,
                                    nc=len(self.model.names),
                                    classes=self.args.classes)

        if not isinstance(orig_imgs, list):  # input images are a torch.Tensor, not a list
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)

        results = []
        
        # print(len(p))

        proto_cross = preds[1][-1] if len(preds[1]) == 4 else preds[1]  # second output is len 4 if pt, but only 1 if exported
        proto = preds[1][-2] if len(preds[1]) == 4 else preds[1]  # second output is len 4 if pt, but only 1 if exported

        for i, pred in enumerate(p):
            orig_img = orig_imgs[i]
            img_path = self.batch[0][i]
            
            # print(pred[:, 0:6])
            # print(img_path)
            boxes = pred[:, :6]
            if not len(pred):  # save empty boxes
                masks_cross = None
                masks = None
            elif self.args.retina_masks:
                pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
                masks_cross = ops.process_mask_native(proto_cross[i], pred[:, 6:], pred[:, :4], orig_img.shape[:2])  # HWC
                masks = ops.process_mask_native(proto[i], pred[:, 6:], pred[:, :4], orig_img.shape[:2])  # HWC
            else:
                masks_cross = ops.process_mask(proto_cross[i], pred[:, 6:], pred[:, :4], img.shape[2:], upsample=True)  # HWC
                masks = ops.process_mask(proto[i], pred[:, 6:], pred[:, :4], img.shape[2:], upsample=True)  # HWC
                pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)

            # import cv2
            # import numpy as np
            # if(len(pred)):
            #     riders = pred[:, 5] == 0
            # pred = pred[riders]
            # masks = masks[riders].cpu()
            #     # masks_cross = masks_cross[riders]
            #     # masks = masks[riders]
            #     # boxes = boxes[riders]
            # for i, mask in enumerate(masks):
            #     # to_show = (mask*255).permute(0, 2, 3, 1).numpy().astype(np.uint8)[0]
            #     # print(to_show.shape)
            #     cv2.imwrite(f"mask{i}.png", (mask*255).numpy().astype(np.uint8))
            #     # cv2.waitKey(0)
            # exit(0)

            results.append(Results(orig_img, path=img_path, names=self.model.names, boxes=boxes, masks=masks, masks_cross=masks_cross))
        return results
