import gdown

url = "https://drive.usercontent.google.com/download?id=11M_tooE6DVHq1-FcIu8GPFVi6il7biOP&export=download&authuser=0&confirm=t&uuid=c8580991-2be1-44bc-9829-d50d02e2914c"
output = "model_best_hints.pth"
gdown.download(url, output, quiet=False)
