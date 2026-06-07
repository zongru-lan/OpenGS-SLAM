import os
import glob
#os.environ['MPLBACKEND'] = 'Agg'
#import matplotlib
#matplotlib.use('Agg')  
import sys
import time
import numpy as np
from argparse import ArgumentParser
from datetime import datetime

import torch
import torch.multiprocessing as mp
import yaml
from munch import munchify
import time

import wandb
from gaussian_splatting.scene.gaussian_model import GaussianModel
from gaussian_splatting.utils.system_utils import mkdir_p
from gui import gui_utils, slam_gui
from utils.config_utils import load_config
from utils.dataset import load_dataset
from utils.eval_utils import eval_ate, eval_rendering, save_gaussians
from utils.logging_utils import Log
from utils.multiprocessing_utils import FakeQueue
from utils.railway_export import export_railway_results
from utils.slam_backend import BackEnd
from utils.slam_frontend import FrontEnd

from dust3r.model import AsymmetricCroCo3DStereo


class SLAM:
    def __init__(self, config, d3r_model, save_dir=None):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()

        self.config = config
        self.save_dir = save_dir
        model_params = munchify(config["model_params"])     
        opt_params = munchify(config["opt_params"])
        pipeline_params = munchify(config["pipeline_params"])
        self.model_params, self.opt_params, self.pipeline_params = (
            model_params,
            opt_params,
            pipeline_params,
        )

        self.live_mode = self.config["Dataset"]["type"] == "realsense"
        self.monocular = self.config["Dataset"]["sensor_type"] == "monocular"
        self.use_spherical_harmonics = self.config["Training"]["spherical_harmonics"]
        self.use_gui = self.config["Results"]["use_gui"]
        if self.live_mode:
            self.use_gui = True
        self.eval_rendering = self.config["Results"]["eval_rendering"]     
        self.color_refinement = self.config["Results"]["color_refinement"]  
        
        model_params.sh_degree = 3 if self.use_spherical_harmonics else 0

        self.gaussians = GaussianModel(model_params.sh_degree, config=self.config)
        self.gaussians.init_lr(6.0)
        self.dataset = load_dataset(
            model_params, model_params.source_path, config=config
        )

        self.gaussians.training_setup(opt_params)
        bg_color = [0, 0, 0]                                               
        self.background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        frontend_queue = mp.Queue()                                         
        backend_queue = mp.Queue()

        q_main2vis = mp.Queue() if self.use_gui else FakeQueue()          
        q_vis2main = mp.Queue() if self.use_gui else FakeQueue()     

        self.config["Results"]["save_dir"] = save_dir
        self.config["Training"]["monocular"] = self.monocular

        self.frontend = FrontEnd(self.config,d3r_model)
        self.backend = BackEnd(self.config, self.save_dir)
        self.frontend.dataset = self.dataset
        self.frontend.background = self.background
        self.frontend.pipeline_params = self.pipeline_params
        self.frontend.frontend_queue = frontend_queue
        self.frontend.backend_queue = backend_queue
        self.frontend.q_main2vis = q_main2vis
        self.frontend.q_vis2main = q_vis2main
        self.frontend.set_hyperparams()

        self.backend.gaussians = self.gaussians
        self.backend.background = self.background
        self.backend.cameras_extent = 6.0          
        self.backend.pipeline_params = self.pipeline_params
        self.backend.opt_params = self.opt_params
        self.backend.frontend_queue = frontend_queue
        self.backend.backend_queue = backend_queue
        self.backend.live_mode = self.live_mode

        self.backend.set_hyperparams()

        self.params_gui = gui_utils.ParamsGUI(
            pipe=self.pipeline_params,
            background=self.background,
            gaussians=self.gaussians,
            q_main2vis=q_main2vis,
            q_vis2main=q_vis2main,
        )

        backend_process = mp.Process(target=self.backend.run) 
        if self.use_gui:
            gui_process = mp.Process(target=slam_gui.run, args=(self.params_gui,))
            gui_process.start()
            time.sleep(5)

        backend_process.start()        
        self.frontend.run()
        backend_queue.put(["pause"])    

        end.record()
        torch.cuda.synchronize()
        # empty the frontend queue
        N_frames = len(self.frontend.cameras)
        FPS = N_frames / (start.elapsed_time(end) * 0.001)
        Log("Total time", start.elapsed_time(end) * 0.001, tag="Eval")
        Log("Total FPS", N_frames / (start.elapsed_time(end) * 0.001), tag="Eval")

        if self.eval_rendering:       
            self.gaussians = self.frontend.gaussians
            kf_indices = self.frontend.kf_indices     
            ATE = eval_ate(
                self.frontend.cameras,
                self.frontend.kf_indices,
                self.save_dir,
                0,
                final=True,
                monocular=self.monocular,
            )

            start_time = time.time()
            rendering_result = eval_rendering(
                self.frontend.cameras,
                self.gaussians,
                self.dataset,
                self.save_dir,
                self.pipeline_params,
                self.background,
                kf_indices=kf_indices,
                iteration="before_opt",
            )
            columns = ["tag", "psnr", "ssim", "lpips", "RMSE ATE", "FPS"]
            metrics_table = wandb.Table(columns=columns)
            metrics_table.add_data(
                "Before",
                rendering_result["mean_psnr"],
                rendering_result["mean_ssim"],
                rendering_result["mean_lpips"],
                ATE,
                FPS,
            )
            end_time = time.time()
            render_time = end_time - start_time
            ave_time = render_time / N_frames
            #print("average render time is:", ave_time)
            
            if self.color_refinement:
                # re-used the frontend queue to retrive the gaussians from the backend.
                while not frontend_queue.empty():     
                    frontend_queue.get()
                backend_queue.put(["color_refinement"])
                while True:
                    if frontend_queue.empty():
                        time.sleep(0.01)
                        continue
                    data = frontend_queue.get()             
                    if data[0] == "sync_backend" and frontend_queue.empty():
                        gaussians = data[1]
                        self.gaussians = gaussians
                        break

                rendering_result = eval_rendering(
                    self.frontend.cameras,
                    self.gaussians,
                    self.dataset,
                    self.save_dir,
                    self.pipeline_params,
                    self.background,
                    kf_indices=kf_indices,
                    iteration="after_opt",
                )
                metrics_table.add_data(
                    "After_opt",
                    rendering_result["mean_psnr"],
                    rendering_result["mean_ssim"],
                    rendering_result["mean_lpips"],
                    ATE,
                    FPS,
                )
            wandb.log({"Metrics": metrics_table})
            save_gaussians(self.gaussians, self.save_dir, "final_after_opt", final=True)

        if (
            self.config["Dataset"].get("type") == "railway"
            and self.config["Results"].get("clean_export", False)
        ):
            if not self.eval_rendering:
                self.gaussians = self.frontend.gaussians
            export_railway_results(
                self.frontend.cameras,
                self.frontend.kf_indices,
                self.gaussians,
                self.dataset,
                self.save_dir,
                self.pipeline_params,
                self.background,
                self.config,
            )

        backend_queue.put(["stop"])
        backend_process.join()       
        Log("Backend stopped and joined the main thread")
        if self.use_gui:               
            q_main2vis.put(gui_utils.GaussianPacket(finish=True))
            gui_process.join()
            Log("GUI Stopped and joined the main thread")

    def run(self):
        pass


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument("--config", type=str)
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--color", action="store_true")

    args = parser.parse_args(sys.argv[1:])

    mp.set_start_method("spawn")      

    with open(args.config, "r") as yml:
        config = yaml.safe_load(yml)

    config = load_config(args.config)
    save_dir = None

    if args.eval:
        Log("Running OpenGS-SLAM in Evaluation Mode")
        Log("Following config will be overriden")
        Log("\tsave_results=True")
        config["Results"]["save_results"] = True
        Log("\tuse_gui=False")
        config["Results"]["use_gui"] = False
        Log("\teval_rendering=True")
        config["Results"]["eval_rendering"] = True
        Log("\tuse_wandb=True")
        config["Results"]["use_wandb"] = True
        
    if args.color:
        config["Results"]["color_refinement"] = True

    if config["Results"]["save_results"]:
        mkdir_p(config["Results"]["save_dir"])
        current_datetime = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")  
        tmp = args.config
        tmp = tmp.split(".")[0]
        if config["Dataset"].get("type") == "railway":
            scene = config["Dataset"].get("scene") or os.path.basename(
                os.path.normpath(config["Dataset"]["dataset_path"])
            )
            sequence_save_dir = os.path.join(config["Results"]["save_dir"], scene)
            internal_runs_dir = os.path.join(sequence_save_dir, "internal_runs")
            mkdir_p(sequence_save_dir)
            mkdir_p(internal_runs_dir)
            save_dir = os.path.join(internal_runs_dir, current_datetime)
            config["Results"]["sequence_save_dir"] = sequence_save_dir
        else:
            dataset_path_parts = config["Dataset"]["dataset_path"].split("/")
            save_dir = os.path.join(
                config["Results"]["save_dir"],
                dataset_path_parts[-3] + "_" + dataset_path_parts[-2],
                current_datetime,
            )
        config["Results"]["save_dir"] = save_dir
        mkdir_p(save_dir)
        with open(os.path.join(save_dir, "config.yml"), "w") as file:     
            documents = yaml.dump(config, file)
        Log("saving results in " + save_dir)
        run = wandb.init(
            project="OpenGS-SLAM",
            name=f"{tmp}_{current_datetime}",
            config=config,
            mode=None if config["Results"]["use_wandb"] else "disabled",
        )
        wandb.define_metric("frame_idx")
        wandb.define_metric("ate*", step_metric="frame_idx")
        
    model_name = (
        config.get("Model", {}).get("checkpoint")
        or config.get("model_params", {}).get("model_path")
        or "checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"
    )
    if not os.path.isfile(model_name):
        raise FileNotFoundError(
            f"Missing DUSt3R checkpoint: {model_name}. "
            "Place DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth under checkpoints/ "
            "or set Model.checkpoint in the config."
        )
    d3r_model = AsymmetricCroCo3DStereo.from_pretrained(model_name).to("cuda")

    slam = SLAM(config, save_dir=save_dir,d3r_model=d3r_model)
    
    slam.run()
    device = "cuda"
    wandb.finish()

    # All done
    Log("Done.")
