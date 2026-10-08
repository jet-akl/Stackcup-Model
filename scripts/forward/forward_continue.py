from model.runner.baxter_stackcup_runner_pipeline import BaxterStackcupRunnerPipeline
from model.baxter.base_move_baxter import BaseMoveBaxter
import cv2

# l: [0.6559424809083204, 0.0174089634278832, 0.04551136320976937]
# r: [0.6610522728819612, -0.3083797811252317, 0.29751583357333866]

# test
if __name__ == '__main__':
    # _baxter = BaseMoveBaxter()
    # _baxter.init_camera('right')

    # while cv2.waitKey(int(1000/30))!=ord('e'):
    #     image = _baxter.latest_image
    #     l_position,_ = _baxter.get_pose('left')
    #     r_position,_ = _baxter.get_pose('right')
    #     if image is not None:
    #         cv2.imshow('right',image)
    #     print('l_position: ',l_position)
    #     print('r_position: ',r_position)

    # cfg = './stackcup_mask_transformer.yaml'
    cfg = './stackcup_mask.yaml'
    # cfg = './stackcup_maskdepth.yaml'
    baxter_stackcup_runner = BaxterStackcupRunnerPipeline(cfg)
    baxter_stackcup_runner.pipline()