## The CTV-SPCP code

## structure of our code
    - code: the code of our paper
    - data: sample data 
    - Simulation: the simulation and result of Section 4.1
    - utils: some support codes of our paper
    - Demo_of_MSI_denoising.m: the demo of multispectral image denoising
    - Demo_of_Video_Extraction: the demo of Video_Extraction
 
## The result of run "Demo_of_MSI_denoising.m"
    The denoising result (PSNR/SSIM/Time) is:
        Data_size    noise type       RPCA              SPCP           CTV-RPCA           CTV-SPCP 
        100*100*160  g=0.2        26.69/0.933/2.0  26.72/0.933/1.5  29.08/0.943/21.3  29.77/0.954/16.3
        100*100*160  g=0.1,S=0.1  30.43/0.969/2.0  30.45/0.970/2.0  32.63/0.974/19.6  32.88/0.977/15.7
    The visual restoration images are obtained by running code.

## The result of run "Demo_of_Video_Extraction.m"
    The foreground extraction result(AUC/Time) is:
        Data_size   noise type       RPCA          SPCP         CTV-RPCA       CTV-SPCP
        144*176*20    g=0.05     0.8066/0.536  0.8091/0.886   0.8572/4.94   0.8621/4.90
        144*176*20    g=0.2      0.6141/0.571  0.6156/0.939   0.6741/5.39   0.6783/5.14
    The visual restoration images are obtained by running code. 

## The simulation result is placed in Simulation folder.


