# GPUPhyllotaxis
# Language: CUDA
# Input: TXT
# Output: TXT
# Tested with: PluMA 1.0, CUDA 10

Run phyllotaxis equations on the GPU, produce arrangement of florets

Original authors: Arian Lopez, Eduardo Dally, Jeremiah Mirander, Mauro Merconchini

The plugin accepts as input a tab-delimited file of keyword-value pairs, as parameters
for the model:
outputPosition (int)  
spiralAngle (real)
spiralPosition  (int)
spiralScale     (real)
floretsPosition (int)
totalFlorets    (int)
showSpiral      (int)

The output TXT file will contain (x, y) coordinates for each floret.
