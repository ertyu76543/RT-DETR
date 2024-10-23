from tools.train import main
import argparse

def run_main():
    args = argparse.Namespace(
        config='/EFL_test/RT-DETR-main/rtdetr_pytorch/configs/rtdetr/rtdetr_r50vd_6x_coco.yml', # config file
        resume='/EFL_test/RT-DETR-main/rtdetr_pytorch/output/rtdetr_r50vd_6x_coco_BCE_warm/checkpoint0034.pth', # checkpoint training
        reweight=False, # Re-weighting
        resample=False, # Re-sampling
        t=0.001,
        tuning=None,
        test_only=True, # evaluation 
        amp=False,
        seed=False
    )
    main(args)

if __name__ == '__main__':
    run_main()