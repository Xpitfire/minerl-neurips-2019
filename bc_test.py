import time
import datetime
import os
import json
import pathlib
import traceback
from src.common.config import Config
from src.bc.utils import get_device
from src.bc.runner import run


if __name__ == "__main__":
    try:
        args = Config()
        st = datetime.datetime.fromtimestamp(time.time()).strftime('%Y%m%d-%H%M%S')
        args.__dict__['save_dir'] = os.path.join(args.save_dir, st)
        pathlib.Path(os.path.join(args.save_dir, st)).mkdir(parents=True, exist_ok=True)
        args.__dict__['rec_save_dir'] = os.path.join(args.rec_save_dir, st)
        pathlib.Path(os.path.join(args.rec_save_dir, st)).mkdir(parents=True, exist_ok=True)
        args_json = json.dumps(args.__dict__)
        with open(os.path.join(args.save_dir, "args.json"), "w") as f:
            f.write(args_json)
        device = get_device(**args.__dict__.copy())
        args.__dict__['device'] = device
        run(**args.__dict__.copy())
    except:
        traceback.print_exc()
        exit(-1)
