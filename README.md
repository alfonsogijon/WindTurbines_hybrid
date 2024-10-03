![DOI](https://img.shields.io/badge/DOI-10.1007%2F978--3--031--63775--9__21-blue)

# 📄 WindTurbines hybrid

🔗 https://doi.org/10.1007/978-3-031-63775-9_21

Hybrid model proposal for wind turbine power prediction.

## 👥 Authors

Alfonso Gijón, Simone Eiraudo, Antonio Manjavacas, Lorenzo Bottaccioli, Andrea Lanzini, Miguel Molina-Solana, Juan Gómez-Romero

## 📖 Abstract

The ever-growing sector of wind energy underscores the importance of optimizing turbine operations and ensuring their maintenance with early fault detection mechanisms. Existing empirical and physics-based models provide approximate predictions of the generated power as a function of the wind speed, but face limitations in capturing the non-linear and complex relationships between input variables and output power. Data-driven methods present new avenues for enhancing wind turbine modeling using large datasets, thereby improving accuracy and efficiency. In this study, we use a hybrid semi-parametric model to leverage the strengths of two distinct approaches in a dataset with four turbines of a wind farm. Our model comprises a physics-inspired submodel, which offers a reliable approximation of the power, combined with a non-parametric submodel to predict the residual component. This non-parametric submodel is fed with a broader set of variables, aiming to capture phenomena not addressed by the physics-based part. For explainability purposes, the influence of input features on the output of the residual submodel is analyzed using SHAP values. The proposed hybrid model finally yields a 35–40 % accuracy improvement in the prediction of power generation with respect to the physics-based model. At the same time, the explainability analysis, along with the physics grounding from the parametric submodel, ensure deep understanding of the analyzed problem. In the end, this investigation paves the way for assessing the impact, and thus the potential optimization, of several unmodeled independent variables on the power generated by wind turbines.

## 📝 Citation

```
@inproceedings{gijon2024explainable,
  title={Explainable Hybrid Semi-parametric Model for Prediction of Power Generated by Wind Turbines},
  author={Gij{\'o}n, Alfonso and Eiraudo, Simone and Manjavacas, Antonio and Bottaccioli, Lorenzo and Lanzini, Andrea and Molina-Solana, Miguel and G{\'o}mez-Romero, Juan},
  booktitle={International Conference on Computational Science},
  pages={299--306},
  year={2024},
  organization={Springer},
  doi={10.1007/978-3-031-63775-9_21}
}
```
