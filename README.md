# ENNreg
(An R package of ENNreg is publicly available on the CRAN website (https://cran.r-project.org/web/packages/evreg/index.html).)

The Python code implements ENNreg, a neural network model for regression in which prediction uncertainty is quantified by Gaussian random fuzzy numbers (GRFNs), a newly introduced family of random fuzzy subsets of the real line that generalizes both Gaussian random variables and Gaussian possibility distributions. The output GRFN is constructed by combining GRFNs induced by prototypes using a combination operator that generalizes Dempster's rule of Evidence Theory. The three output units indicate the most plausible  value of the response variable, variability around this value, and epistemic uncertainty. The network is trained by minimizing a loss function that generalizes the negative log-likelihood. 

Readers are invited to read the papers mentioned below to explore the main concepts underlying epistemic random fuzzy sets and evidential regression.


If you find this code useful and use it in your own research, please cite the following papers:

######### Citing this paper ########
```bash
@article{denoeux2023quantifying,
  title={Quantifying prediction uncertainty in regression using random fuzzy sets: the ENNreg model},
  author={Den{\oe}ux, Thierry},
  journal={IEEE Transactions on Fuzzy Systems},
  volume={31},
  number={10},
  pages={3690--3699},
  year={2023},
  publisher={IEEE}
}
@article{denoeux2023parametric,
  title={Parametric families of continuous belief functions based on generalized Gaussian random fuzzy numbers},
  author={Den{\oe}ux, Thierry},
  journal={Fuzzy Sets and Systems},
  volume={471},
  pages={108679},
  year={2023},
  publisher={Elsevier}
}
@article{huang2025evidential,
  title={Evidential time-to-event prediction with calibrated uncertainty quantification},
  author={Huang, Ling and Xing, Yucheng and Mishra, Swapnil and Den{\oe}ux, Thierry and Feng, Mengling},
  journal={International Journal of Approximate Reasoning},
  pages={109403},
  year={2025},
  publisher={Elsevier}
}
```
