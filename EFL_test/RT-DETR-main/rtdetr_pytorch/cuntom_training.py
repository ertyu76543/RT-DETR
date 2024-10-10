from tools.train import main
import argparse

def run_main():
    # 필요한 인자를 설정하여 args 객체 생성
    args = argparse.Namespace(
        config='configs/rtdetr/rtdetr_r50vd_6x_coco.yml', # config file
        resume='/EFL_test/RT-DETR-main/rtdetr_pytorch/output/rtdetr_r50vd_6x_coco_BCE_warm/checkpoint0035.pth', # checkpoint training
        lossfunc='bce', # Re-weighting
        irfs=False, # Re-sampling
        # rebalanc=False, # Re-balancing
        tuning=None,
        test_only=False, # evaluation 만
        amp=False,
        seed=False
    )
    main(args)

if __name__ == '__main__':
    run_main()