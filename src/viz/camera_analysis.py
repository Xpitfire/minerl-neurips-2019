import numpy as np
import json


json_file = '/publicdata/minerl/dataset_20190712/MineRLTreechop-v0/v1_key_nectarine_spirit-2_3251-4975/metadata.json'
npz_file = '/publicdata/minerl/dataset_20190712/MineRLTreechop-v0/v1_juvenile_apple_angel-7_131635-133166/rendered.npz'


def run():
    with open(json_file, 'r') as f:
        meta = json.load(f)
    print(meta)
    rend = np.load(npz_file)
    print(np.shape(rend))
    for i in rend.items():
        print(i)
    cam0, cam1 = rend['action_camera'][:, 0], rend['action_camera'][:, 1]

    print(cam0[:20])

    bins = 20
    fact = np.log(180)/bins
    cam0_p = np.zeros_like(cam0)
    cam0_n = np.zeros_like(cam0)
    for i in range(len(cam0)):
        v = cam0[i]
        if v > 0:
            cam0_p[i] = np.log(v + 1)//fact
        elif v < 0:
            cam0_n[i] = np.log(abs(v) + 1)//fact

    print(cam0_p[:20])
    print(cam0_n[:20])

    inv_cam0 = (np.exp(fact*cam0_p)-1) + (np.exp(fact*cam0_n)-1)

    print(inv_cam0[:20])


if __name__ == '__main__':
    run()
