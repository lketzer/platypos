# PLATYPOS - PLAneTarY PhOtoevaporation Simulator
Tool to estimate the atmospheric mass loss of planets induced by the stellar X-ray and extreme UV irradiance. 


## Our Model Assumptions
We do not make use of full-blown hydrodynamical simulations, but instead couple existing parametrizations of planetary mass-radius relations with an energy-limited hydrodynamic escape model to estimate the mass-loss rate over time.

### Mass-loss description: <br> 
-------------------------
<a href="https://www.codecogs.com/eqnedit.php?latex=\small&space;\dot{M}&space;=&space;\epsilon&space;\frac{(\pi&space;R_{XUV}^2)&space;F_{\mathrm{XUV}}}{K&space;G&space;M_{pl}/R_{pl}&space;}&space;=&space;\epsilon&space;\frac{3&space;\beta^2&space;F_{\mathrm{XUV}}}{4&space;G&space;K&space;\rho_{pl}}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\small&space;\dot{M}&space;=&space;\epsilon&space;\frac{(\pi&space;R_{XUV}^2)&space;F_{\mathrm{XUV}}}{K&space;G&space;M_{pl}/R_{pl}&space;}&space;=&space;\epsilon&space;\frac{3&space;\beta^2&space;F_{\mathrm{XUV}}}{4&space;G&space;K&space;\rho_{pl}}" title="\small \dot{M} = \epsilon \frac{(\pi R_{XUV}^2) F_{\mathrm{XUV}}}{K G M_{pl}/R_{pl} } = \epsilon \frac{3 \beta^2 F_{\mathrm{XUV}}}{4 G K \rho_{pl}}" /></a>

where 
<a href="https://www.codecogs.com/eqnedit.php?latex=\small&space;F_{\mathrm{XUV}}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\small&space;F_{\mathrm{XUV}}" title="\small F_{\mathrm{XUV}}" /></a>
is the flux impinging on the planet, 
<a href="https://www.codecogs.com/eqnedit.php?latex=\small&space;R_{pl}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\small&space;R_{pl}" title="\small R_{pl}" /></a>
and 
<a href="https://www.codecogs.com/eqnedit.php?latex=\small&space;R_{XUV}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\small&space;R_{XUV}" title="\small R_{XUV}" /></a>
are the planetary radii at optical and XUV wavelengths, respectively; 
$\beta = R_{XUV}/R_{pl}$. 
$M_{pl}$ 
is the mass and 
$\rho_{pl}$ 
the density of the planet, 
$\epsilon$ 
is the efficiency of the atmospheric escape with a value between 0 and 1, and $K$ is a factor representing the impact of Roche lobe overflow (Erkaev et al., 2007), which can take on values of 1 for no Roche lobe influence and <1 for planets filling significant fractions of their Roche lobes.


### Planet Model description: <br>
-------------------------
At the moment, the user can choose between two planet models.

1. Rocky core with H/He envelope on top <br>
If this is also the case for the V1298 Tau planets, their current masses could be much lower than estimated by a mass-radius relationship valid for older planets. We approximate this scenario by using models of planets with a hydrogen/helium envelope on top of a 5 and 10 M$_\oplus$ core, using the tabulated models of \citet{LopezFortney2014}. They calculate radii for low-mass planets with hydrogen-helium envelopes on top of Earth-like rocky cores, taking into account the cooling and thermal contraction of the atmospheres of such planets over time. Their simulations extend to young planetary ages, at which planets are expected to still be warm and possibly inflated. \citet{LopezFortney2014} provide simple analytical fits to their simulation results, which we use to trace the thermal and photoevaporative evolution of the planetary radius over time. We refer to this as the ''fluffy planet scenario'' in the following.\\

1. Planet which follows the empirical mass-radius relationships observed for planets around older stars <br>
(e.g. Otegi et al. (2020), also Chen & Kipping (2017)). These relationships show two regimes, one for small rocky planets up to radii of about $2R_\oplus$ and one for larger planets with volatile-rich envelopes. The scatter is low in the rocky planet regime and larger in the gaseous planet regime: as core vs.\ envelope fractions may vary, there is a broader range of observed masses at a given planetary radius for those larger planets. It is noteworthy that the young planet K2-100b, which has an age of $\approx 700$ Myr based on the cluster membership of its host star \citep{Mann2017}, falls into the volatile envelope regime and follows the mass-radius relationship seen for older planets.



## Repository Structure:

**platypos_package**: contains the planet classes & all the necessary funtions to construct a planet and make it evolve <br>
                      (LoF014 planet with rocky core & gaseous envelope OR planet based on mass-radius relation for mature planets (Ot20))

**supplementary_files**: contains some extra files for plotting

**example_V1298Tau**: evolve the four young V1298 Tau planets as shown in "X-ray irradiation and evaporation of the four young planets around V1298 Tau" (Poppenhaeger et al. 2020)

**population_evolution**: evolve a whole population of planets (implemented in the future)
