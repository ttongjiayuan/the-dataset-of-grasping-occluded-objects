import numpy as np
a=np.asarray([[618.62, 0, 320], [0, 618.62, 240]])
a=np.round(a * 10000).astype(np.uint16)
print(a)