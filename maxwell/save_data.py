
# %%

%matplotlib inline
import os
import numpy as np
import matplotlib.pyplot as plt
from extra_data import open_run, DataCollection


# This script reads all trains from a given run and saves the mean JF3 image

home_dir = "/home/vozdavoj/"
prop_no = 2838
prop_dir = os.path.join(home_dir, f"p{prop_no}")
run_no = 135

# data can be `proc`, `raw` or `all`
run = open_run(proposal=prop_no, run=run_no, data="proc")
trainN = len(run.train_ids)
print(f"Run {run_no:} contains {trainN} train IDs.")

meanJF3 = np.zeros([512,1024])
meanBkg = np.zeros([512,1024])
trainC = 0
bkgC = 0
for train_no in range(trainN):
    print(f"Reading train {train_no:-4}/{trainN} ",end='',flush=True)

    sel = run.select_trains(train_no)
    JF3 = sel['HED_IA1_JF500K3/DET/JNGFR03:daqOutput', 'data.adc'].xarray().squeeze().data
    print(f"sum = {np.sum(JF3):-7.0f} ",end='')
    if np.sum(JF3) > 1e6:
        meanJF3 += JF3
        trainC += 1
        print('-> used')
    else:
        meanBkg += JF3
        bkgC += 1
        print('-> skipped')
    
    # if train_no == 20:
    #     print(np.min(JF3),np.max(JF3),np.mean(JF3))
    #     # plt.imshow(JF3,vmin=-1,vmax=0)
    #     plt.imshow(JF3,vmin=-1,vmax=30)
    #     plt.show()
        # break
    # if train_no>100:
    #     break
    
print(f"Used {trainC:-4}/{trainN} trains for signal and {bkgC:-4} for background")

meanJF3 = meanJF3*trainC
meanBkg = meanBkg*bkgC

# %%

diff = meanJF3 - meanBkg
diff = np.where(diff>0,diff,0)

print(f"meanJF3: min={np.min(meanJF3):.0f} max={np.max(meanJF3):.0f} mean={np.mean(meanJF3):.2f}")
plt.imshow(meanJF3, vmin=0, vmax=8)
plt.colorbar()
plt.show()

print(f"meanBkg: min={np.min(meanBkg):.0f} max={np.max(meanBkg):.0f} mean={np.mean(meanBkg):.2f}")
plt.imshow(meanBkg, vmin=0, vmax=8)
plt.colorbar()
plt.show()

print(f"diff: min={np.min(diff):.0f} max={np.max(diff):.0f} mean={np.mean(diff):.2f}")
plt.imshow(diff, vmin=0, vmax=8)
plt.colorbar()
plt.show()

np.save(os.path.join(prop_dir,f"JF3_run[{run_no:03}]_train[all]_bkg.npy"),meanJF3)

# %%

