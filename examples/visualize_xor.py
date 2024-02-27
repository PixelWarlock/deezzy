import matplotlib.pyplot as plt
import numpy as np
import torch
import os
from deezzy.memberships.gaussian import Gaussian
from tqdm import tqdm
from matplotlib.animation import FuncAnimation
from matplotlib import animation

def compute_univariate_gaussian(x, u, s):
    return np.exp(-0.5*((x-u)/s)**2)


#fig_fgp, axes_fgp = plt.subplots(figsize=(15,9), nrows=2, ncols=1)
#fig_cmfp, axes_cmfp  = plt.subplots(figsize=(15,9), nrows=2, ncols=1)

fig_all, axes_all = plt.subplots(figsize=(15,9), nrows=3, ncols=2)

class Animator:

    def __init__(self, directory):

        self.fgp_directory = os.path.join(directory, 'fgp')
        self.fgp_filepaths = [os.path.join(self.fgp_directory, filename) for filename in os.listdir(self.fgp_directory)]
        self.fgp_filepaths.sort(key=lambda f: int(os.path.basename(f).split('_')[-1].split('.')[0]))

        self.cmfp_directory = os.path.join(directory, 'cmfp')
        self.cmfp_filepaths = [os.path.join(self.cmfp_directory, filename) for filename in os.listdir(self.cmfp_directory)]
        self.cmfp_filepaths.sort(key=lambda f: int(os.path.basename(f).split('_')[-1].split('.')[0]))
        
        self.combinations = None

    def compute_fgp(self, filepath_index):
        
        gaussian_membership = Gaussian(univariate=True)
        
        #for fgp_filepath in tqdm(self.fgp_filepaths):
        fgp_filepath = self.fgp_filepaths[filepath_index]

        point = -2.
        step = 0.01
        output_dict= {
            "f0g0":[],
            "f0g1":[],
            "f1g0":[],
            "f1g1":[],
            "domain":[]
        }
        fgp = torch.load(fgp_filepath).detach().cpu().unsqueeze(dim=0) #[0,...].view(1, 2,2,2)
        while point < 3.:
            
            inputs = torch.Tensor([point, point]).unsqueeze(dim=0)
            output = gaussian_membership(inputs, fgp).squeeze(dim=0).numpy()
            
            num_features=output.shape[0]
            granularity = output.shape[1] 

            for f in range(num_features):
                for g in range(granularity):

                    key = f"f{f}g{g}"
                    out = output[f,g]
                    output_dict[key].append(out)
            
            output_dict['domain'].append(point)
            point += step

        return output_dict

    def calculate_combinations(self, domain):

        combo = list()
        for y in domain:
            for x in domain:
                
                input_vector = np.array([x,y])
                combo.append(input_vector)
                
        combo = np.array(combo)
        return combo

    def compute_cmfp(self, filepath_index):
        gaussian_membership = Gaussian(univariate=False)

        cmfp_filepath = self.cmfp_filepaths[filepath_index]
        output_dict= {
            "c0m0":[],
            "c0m1":[],
            "c1m0":[],
            "c1m1":[],
            "domain":[]
        }

        cmfp = torch.load(cmfp_filepath).detach().cpu().unsqueeze(dim=0) #[0,...].view(1,2,2,2,2)

        for combo in self.combinations:
            
            inputs = torch.Tensor(combo).unsqueeze(dim=0)
            output = gaussian_membership(inputs, cmfp).squeeze(dim=0).numpy()

            for c,c_vector in enumerate(output):

                for gm, gm_value in enumerate(c_vector):

                    key = f"c{c}m{gm}"

                    output_dict[key].append(gm_value)
        return output_dict
    

    def animate_all(self, i):
        axes_all[0,0].clear()
        axes_all[1,0].clear()
        axes_all[0,1].clear()
        axes_all[1,1].clear()

        axes_all[2,0].clear()
        axes_all[2,1].clear()

        if self.combinations is None:
            self.combinations = self.calculate_combinations(domain=np.linspace(0,1,100))

        fgp_output = self.compute_fgp(i)
        cmfp_output = self.compute_cmfp(i)

        # features
        axes_all[0,0].grid(True)
        axes_all[0,0].plot(fgp_output['domain'],fgp_output['f0g0'], 'r')
        axes_all[0,0].plot(fgp_output['domain'],fgp_output['f0g1'], 'b')
        axes_all[0,0].set_title("Feature 0")

        axes_all[0,1].grid(True)
        axes_all[0,1].plot(fgp_output['domain'],fgp_output['f1g0'], 'r')
        axes_all[0,1].plot(fgp_output['domain'],fgp_output['f1g1'], 'b')
        axes_all[0,1].set_title("Feature 1")

        # classes

        #axes[0,1].tricontourf(self.combinations[:,0], self.combinations[:,1],membership_c0)
        #axes[1,1].tricontourf(self.combinations[:,0], self.combinations[:,1],membership_c1)

        axes_all[1,0].tricontourf(self.combinations[:,0], self.combinations[:,1],cmfp_output['c0m0'])
        axes_all[1,0].set_title("Class: 0 | Gaussian: 0")
        
        axes_all[1,1].tricontourf(self.combinations[:,0], self.combinations[:,1],cmfp_output['c0m1'])
        axes_all[1,1].set_title("Class: 0 | Gaussian: 1")

        axes_all[2,0].tricontourf(self.combinations[:,0], self.combinations[:,1],cmfp_output['c1m0'])
        axes_all[2,0].set_title("Class: 1 | Gaussian: 0")
        
        axes_all[2,1].tricontourf(self.combinations[:,0], self.combinations[:,1],cmfp_output['c1m1'])
        axes_all[2,1].set_title("Class: 1 | Gaussian: 1")

        plt.tight_layout()
        print(f"Finished frame: {i}")

        #plt.show()


    def run_animate_all(self):
        ani = FuncAnimation(fig_all, self.animate_all, frames=len(self.cmfp_filepaths), interval=10, repeat=False)
        FFwriter=animation.FFMpegWriter(fps=10)
        ani.save("./all.mp4", dpi=300, writer=FFwriter) 
        plt.close()

    def get_figure(self, index):
        axes_all[0,0].clear()
        axes_all[1,0].clear()
        axes_all[0,1].clear()
        axes_all[1,1].clear()

        axes_all[2,0].clear()
        axes_all[2,1].clear()

        if self.combinations is None:
            self.combinations = self.calculate_combinations(domain=np.linspace(0,1,100))

        fgp_output = self.compute_fgp(index)
        cmfp_output = self.compute_cmfp(index)

        # features
        axes_all[0,0].grid(True)
        axes_all[0,0].plot(fgp_output['domain'],fgp_output['f0g0'], 'r')
        axes_all[0,0].plot(fgp_output['domain'],fgp_output['f0g1'], 'b')
        axes_all[0,0].set_title("Feature 0")

        axes_all[0,1].grid(True)
        axes_all[0,1].plot(fgp_output['domain'],fgp_output['f1g0'], 'r')
        axes_all[0,1].plot(fgp_output['domain'],fgp_output['f1g1'], 'b')
        axes_all[0,1].set_title("Feature 1")

        # classes

        #axes[0,1].tricontourf(self.combinations[:,0], self.combinations[:,1],membership_c0)
        #axes[1,1].tricontourf(self.combinations[:,0], self.combinations[:,1],membership_c1)

        axes_all[1,0].tricontourf(self.combinations[:,0], self.combinations[:,1],cmfp_output['c0m0'])
        axes_all[1,0].set_title("Class: 0 | Gaussian: 0")
        
        axes_all[1,1].tricontourf(self.combinations[:,0], self.combinations[:,1],cmfp_output['c0m1'])
        axes_all[1,1].set_title("Class: 0 | Gaussian: 1")

        axes_all[2,0].tricontourf(self.combinations[:,0], self.combinations[:,1],cmfp_output['c1m0'])
        axes_all[2,0].set_title("Class: 1 | Gaussian: 0")
        
        axes_all[2,1].tricontourf(self.combinations[:,0], self.combinations[:,1],cmfp_output['c1m1'])
        axes_all[2,1].set_title("Class: 1 | Gaussian: 1")

        plt.tight_layout()

        

    def animate_and_save_as_images(self, output_dir):
        if os.path.exists(output_dir) is False:
            os.mkdir(output_dir)
        for i in tqdm(range(len(self.cmfp_filepaths))):
            self.get_figure(index=i)
            plt.savefig(os.path.join(output_dir, f"fig_{i}.png"))

    def preview_epoch(self, epoch=None):
        
        if self.combinations is None:
            self.combinations = self.calculate_combinations(domain=np.linspace(0,1,100)) #00

        
        if epoch is None:
            print("Epoch:", len(self.fgp_filepaths))
            fgp_output = self.compute_fgp(len(self.fgp_filepaths)-1)
            cmfp_output = self.compute_cmfp(len(self.cmfp_filepaths)-1)
        else:
            print("Epoch:", epoch)
            fgp_output = self.compute_fgp(epoch)
            cmfp_output = self.compute_cmfp(epoch)

        

        #membership_c0 = np.array(cmfp_output['c0m0']) + np.array(cmfp_output['c0m1'])
        #membership_c1 = np.array(cmfp_output['c1m0']) + np.array(cmfp_output['c1m1'])

        
        fig, axes = plt.subplots(figsize=(15,9), nrows=3, ncols=2)

        # features
        axes[0,0].grid(True)
        axes[0,0].plot(fgp_output['domain'],fgp_output['f0g0'], 'b')
        axes[0,0].plot(fgp_output['domain'],fgp_output['f0g1'], 'r')
        axes[0,0].set_title("Feature 0")
        axes[0,0].set_xlabel("Input value")
        axes[0,0].set_ylabel("Membership degree")

        axes[0,0].set_xlim([-2,3])

        axes[0,1].grid(True)
        axes[0,1].plot(fgp_output['domain'],fgp_output['f1g0'], 'b')
        axes[0,1].plot(fgp_output['domain'],fgp_output['f1g1'], 'r')
        axes[0,1].set_title("Feature 1")
        axes[0,1].set_xlim([-2,3])
        axes[0,1].set_xlabel("Input value")
        axes[0,1].set_ylabel("Membership degree")

        # classes

        #axes[0,1].tricontourf(self.combinations[:,0], self.combinations[:,1],membership_c0)
        #axes[1,1].tricontourf(self.combinations[:,0], self.combinations[:,1],membership_c1)

        axes[1,0].tricontourf(self.combinations[:,0], self.combinations[:,1],cmfp_output['c0m0'])
        axes[1,0].set_title("Class: 0 | Gaussian: 0")
        axes[1,0].set_xlabel("Adjective value for first feature")
        axes[1,0].set_ylabel("Adjective value for second feature")
        
        axes[1,1].tricontourf(self.combinations[:,0], self.combinations[:,1],cmfp_output['c0m1'])
        axes[1,1].set_title("Class: 0 | Gaussian: 1")
        axes[1,1].set_xlabel("Adjective value for first feature")
        axes[1,1].set_ylabel("Adjective value for second feature")
        

        axes[2,0].tricontourf(self.combinations[:,0], self.combinations[:,1],cmfp_output['c1m0'])
        axes[2,0].set_title("Class: 1 | Gaussian: 0")
        axes[2,0].set_xlabel("Adjective value for first feature")
        axes[2,0].set_ylabel("Adjective value for second feature")
        
        axes[2,1].tricontourf(self.combinations[:,0], self.combinations[:,1],cmfp_output['c1m1'])
        axes[2,1].set_title("Class: 1 | Gaussian: 1")
        axes[2,1].set_xlabel("Adjective value for first feature")
        axes[2,1].set_ylabel("Adjective value for second feature")

        plt.tight_layout()

        plt.suptitle(f"Epoch: {epoch}")
        plt.show()

if __name__ == '__main__':
    animator = Animator(os.path.join(os.getcwd(), "outputs/xor_representations")) #.run_animate_fgp()
    #animator.animate_and_save_as_images(output_dir=os.path.join(os.path.join(os.getcwd(), "outputs/results"))) 
    animator.preview_epoch(epoch=499)