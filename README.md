# PLATYPOS - PLAneTarY PhOtoevaporation Simulator
Tool to estimate the atmospheric mass loss of planets induced by stellar X-ray and extreme UV irradiation. 

![](./supplementary_files/platypos3_2_best.png)

**NOTE: I am currently updating the code, so please be patient :)**

## Installation

**NOTE: 'pip install platypos' installs an old version. Better to clone and use the *version090623* branch for now!** <br> 
Also, if you want to use it and something doesn't work or you have questions, simply write me an e-mail (lketzer@aip.de) <br>
Add-on: git clone multitrack and look at the example on how to easily evolve a set of planets in Test_platypos_and_multi_track.ipynb

Create a virtual environment:

```bash
cd /Path/
python3 -m venv venv/
source venv/bin/activate
```

Install ```platypos``` through pip:

```bash
pip install platypos
```

## Our Model Assumptions
We do not make use of full-blown hydrodynamical simulations, but instead couple existing parametrizations of planetary mass-radius relations with an energy-limited hydrodynamic escape model to estimate the mass-loss rate over time. 

### Mass-loss description: <br> 
We have three mass-loss rate calculations built in. Energy-limited, radiation-recombination-limited and a hydro-based approximation. Check Ketzer & Poppenhaeger 2022 for detailed describtions. <br>

<a href="https://www.codecogs.com/eqnedit.php?latex=\small&space;\dot{M}&space;=&space;\epsilon&space;\frac{(\pi&space;R_{XUV}^2)&space;F_{\mathrm{XUV}}}{K&space;G&space;M_{pl}/R_{pl}&space;}&space;=&space;\epsilon&space;\frac{3&space;\beta^2&space;F_{\mathrm{XUV}}}{4&space;G&space;K&space;\rho_{pl}}\,," target="_blank"><img src="https://latex.codecogs.com/gif.latex?\small&space;\dot{M}&space;=&space;\epsilon&space;\frac{(\pi&space;R_{XUV}^2)&space;F_{\mathrm{XUV}}}{K&space;G&space;M_{pl}/R_{pl}&space;}&space;=&space;\epsilon&space;\frac{3&space;\beta^2&space;F_{\mathrm{XUV}}}{4&space;G&space;K&space;\rho_{pl}}\,," title="\small \dot{M} = \epsilon \frac{(\pi R_{XUV}^2) F_{\mathrm{XUV}}}{K G M_{pl}/R_{pl} } = \epsilon \frac{3 \beta^2 F_{\mathrm{XUV}}}{4 G K \rho_{pl}}\,," /></a>

where 
<a href="https://www.codecogs.com/eqnedit.php?latex=\small&space;F_{\mathrm{XUV}}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\small&space;F_{\mathrm{XUV}}" title="\small F_{\mathrm{XUV}}" /></a>
is the flux impinging on the planet, 
<a href="https://www.codecogs.com/eqnedit.php?latex=\small&space;R_{pl}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\small&space;R_{pl}" title="\small R_{pl}" /></a>
and 
<a href="https://www.codecogs.com/eqnedit.php?latex=\small&space;R_{XUV}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\small&space;R_{XUV}" title="\small R_{XUV}" /></a>
are the planetary radii at optical and XUV wavelengths, respectively (
<a href="https://www.codecogs.com/eqnedit.php?latex=\small&space;\beta&space;=&space;R_{XUV}/R_{pl}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\small&space;\beta&space;=&space;R_{XUV}/R_{pl}" title="\small \beta = R_{XUV}/R_{pl}" /></a>).
<a href="https://www.codecogs.com/eqnedit.php?latex=\small&space;M_{pl}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\small&space;M_{pl}" title="\small M_{pl}" /></a>
is the mass and 
<a href="https://www.codecogs.com/eqnedit.php?latex=\small&space;\rho_{pl}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\small&space;\rho_{pl}" title="\small \rho_{pl}" /></a>
the density of the planet, 
<a href="https://www.codecogs.com/eqnedit.php?latex=\small&space;\epsilon" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\small&space;\epsilon" title="\small \epsilon" /></a>
is the efficiency of the atmospheric escape with a value between 0 and 1, and K is a factor representing the impact of Roche lobe overflow (Erkaev et al. 2007)<sup>[1](#Erkaev-et-al-07)</sup>, which can take on values of 1 for no Roche lobe influence and <1 for planets filling significant fractions of their Roche lobes.


**Stellar high-energy evolution** <br>
Most previous studies of exoplanet evaporation approximate the stellar XUV evolution by using the average activity level of stars in a specific mass bin for well-studied clusters of different ages, and approximating it with a broken power-law with a 100 Myr-long saturation regime. Observations and theoretical studies show, however, that stars spin down at a wider range of ages (see Barnes 2003<sup>[2](#Barnes-03)</sup>, Matt et al. 2012<sup>[3](#Matt-et-al-12)</sup>, Tu et al. 2015<sup>[4](#Tu-et-al-15)</sup>, Garaffo et al. 2018<sup>[5](#Garaffo-et-al-2018)</sup>). In the context of exoplanet irradiation, this was explored in simulations by Tu et al. (2015)<sup>[4](#Tu-et-al-15)</sup> and Johnstone et al. (2015)<sup>[6](#Johnstone-et-al-2015)</sup>. Their studies show that the saturation timescales can range from ~10 to 300 Myr for solar-mass stars. Hence, a star that spins down quickly will follow a low-activity track, while a star that can maintain its rapid rotation will follow a high-activity track. This translates into significantly different irradiation levels for exoplanets, and thus the amount and strength of evaporation. Based on the findings by Tu et al. (2015), we generate a more realistic stellar activity evolution of the host star by adopting a broken power-law model with varying saturation and spin-down time scales to approximate a low-, medium- and high-activity scenario for the host star.


### Planet Model description: <br>
At the moment, the user can choose between three planet models.

1. *Planet with a rocky core and H/He envelope atop* <br>
We use the tabulated models of Lopez & Fortney (2014)<sup>[7](#Lopez-Fortney-14)</sup>, who calculate radii for low-mass planets with hydrogen-helium envelopes on top of Earth-like rocky cores, taking into account the cooling and thermal contraction of the atmospheres of such planets over time. Their simulations extend to young planetary ages, at which planets are expected to still be warm and possibly inflated. Simple analytical fits to their simulation results are provided, which we use to trace the thermal and photoevaporative evolution of the planetary radius over time. In addition, the MESA-based models by Chen & Rogers (2016) can be selected by the user too (see Ketzer & Poppenhaeger 2022 for a comparison of the two).

1. *Planet which follows the empirical mass-radius relationships observed for planets around older stars* <br> 
(see Otegi et al. (2020)<sup>[8](#Otegi-et-al-2020)</sup>, also Chen & Kipping (2017)<sup>[9](#Chen-Kipping-2017)</sup>) <br>
These "mature" relationships show two regimes, one for small rocky planets up to radii of about 2 Earth radii and one for larger planets with volatile-rich envelopes. The scatter is low in the rocky planet regime and larger in the gaseous planet regime: as core vs. envelope fractions may vary, there is a broader range of observed masses at a given planetary radius for those larger planets. 

If you want to know more about the siumulation inputs, check the code (partially documented) or Ketzer & Poppenhaeger 2022.

## Quickstart

Open a jupyter notebook:
```bash
python3 -m notebook
```
and import `platypos`: 
```python
from platypos import Planet_LoFo14
from platypos import Planet_Ot20
```

To create a planet object you need to specify several things: <br>
1) Specify host star parameters <br>
```python
host_star_params = {'star_id': 'star1', 'mass': mass_star, 'radius': radius_star,
		    'age': age_star, 'L_bol': L_bol, 'Lx_age': Lx_at_star_age}
```

2) Specify the stellar evolutionary track: <br> 
```python
stellar_evolutionary_track = {'t_start': 20. (Myr), 't_sat': 100., 't_curr': 1000.,
			      't_5Gyr': 5000., 'Lx_max': Lx_saturation,
			      'Lx_curr': Lx_1Gyr, 'Lx_5Gyr': Lx_5Gyr, 'dt_drop': 0., 'Lx_drop_factor': 0.}
```

(e.g for solar-mass star, based on Tu et al. 2015, Lx_1Gyr and Lx_5Gyr should be set to 
<img src="https://render.githubusercontent.com/render/math?math=$2.10*10^28$"> and <img src="https://render.githubusercontent.com/render/math?math=$1.65*10^27$"> [erg/s]) <br>

3) Specify the planet parameters <br>
```python
planet_params1 = {'radius': 5.59, 'distance': 0.0825, 'core_mass': 5.0}
planet_params2 = {'radius': 5.59, 'distance': 0.0825}
```

4) Create the planet object <br>
```python
pl = Planet_LoFo14(host_star_params, planet_params1)
pl = Planet_ChRo16(host_star_params, planet_params1)
pl = Planet_Ot20(host_star_params, planet_params2)
```

5) Specify additional parameters for the platypos run <br>
 	- end time of simulation: t_final
	- initial step size: init_dt (0.1 Myr)
	- evaporation efficiency: epsilon (~10%)
	- K on (or off)? `"yes"` or `"no"`
	- how to estimate your imporant EUVS ('Johnstone', 'SanzForcada', or 'Linsky')
	- the mass-loss rate estimation method ('Elim', 'Elim_and_RRlim','HBA')
	- beta_settings (beta=1, Salz or Lopez beta; check documentation)
	- track to evolve star-planet system along: stellar_evolutionary_track
	- path to save results: path_save
	- folder in path_save to save results in: folder_id

4) Evolve the planet along defined track: <br>
```python
pl.evolve_forward_and_create_full_output(t_final, init_dt, epsilon, 
					 K_on="yes", relation_EUV="Johnstone", 
					 mass_loss_calc='Elim',
                                         beta_settings={"beta_calc": "off},
					 stellar_evolutionary_track, 
					 path_save, folder_id)
```

5) Look at Results (mass & radius evolution with time): <br>
```python
df_pl = pl.read_results(path_save)
```

### Additional Info:

* For more details go into the source code. Lots of comments there.

* **supplementary_files**: Contains some extra files for plotting; <br>
                           Tu et al. (2015)<sup>[4](#Tu-et-al-15)</sup> model tracks for the X-ray luminosity evolution,  <br>
                           Jackson et al. (2012)<sup>[10](#Jackson-et-al-12)</sup> sample of X-ray measurements in young clusters)

* **examples**: Best to look at the example notebooks which show how to use `playpos`.  <br>
		*evolve_one_planet.ipynb* or *evolve_one_planet_latertypestar.ipynb* <br>
		Evolve the four young V1298 Tau planets as shown in *X-ray irradiation and evaporation of the four young planets around V1298 Tau* (Poppenhaeger, 		  Ketzer, Mallon 2020)<sup>[11](#Poppenhaeger-et-al-20)</sup> <br>
		NOTE: for the V1298Tau notebook, you also need the package `multitrack`. 


## References:
<a name="Kubyshkina-et-al-18">1</a>: [Kubyshkina et al. 2018](https://arxiv.org/pdf/1810.06920.pdf) <br>
<a name="Erkaev-et-al-07">1</a>: [Erkaev et al. 2007](https://arxiv.org/abs/astro-ph/0612729) <br>
<a name="Barnes-03">2</a>: [Barnes 2003](https://arxiv.org/abs/astro-ph/0303631) <br>
<a name="Matt-et-al-12">3</a>: [Matt et al. 2012](https://arxiv.org/abs/1206.2354) <br>
<a name="Tu-et-al-15">4</a>: [Tu et al. 2015](https://arxiv.org/abs/2005.10240) <br>
<a name="Garaffo-et-al-2018">5</a>: [Garaffo et al. 2018](https://arxiv.org/abs/1804.01986) <br>
<a name="Johnstone-et-al-2015">6</a>: [Johnstone et al. 2015](https://arxiv.org/abs/1503.07494) <br>
<a name="Lopez-Fortney-14">7</a>: [Lopez & Fortney 2014](https://arxiv.org/abs/1311.0329) <br>
<a name="Otegi-et-al-2020">8</a>: [Otegi et al. 2020](https://arxiv.org/abs/1911.04745) <br>
<a name="Chen-Kipping-2017">9</a>: [Chen & Kipping 2017](https://arxiv.org/abs/1603.08614) <br>
<a name="Jackson-et-al-12">10</a>: [Jackson et al. 2012](https://arxiv.org/abs/1111.0031) <br>
<a name="Poppenhaeger-et-al-20">11</a>: [Poppenhaeger, Ketzer, Mallon 2020](https://arxiv.org/abs/2005.10240) <br>
