from hdf_dataset import HDFDataset


ds = HDFDataset(
    path=r"C:\Users\MALAR507\Documents\GitHub\ProstateCancerPrognosisAI\examples\local_data\patients_dataset.h5"
)

raph = []
max = []
ian = []
for idx, data in enumerate(ds):
    sha = data[0][:, :, 120].shape

    if sha == (333, 333):
        raph.append(sha)
    elif sha == (467, 467):
        print(idx)
        max.append(sha)
    else:
        ian.append(sha)
        print(sha)

print(f"RAPH {len(raph)}, MAX {len(max)}, IAN {len(ian)}")
