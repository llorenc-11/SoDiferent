# How to guide

The fucntion used for solving systems of ODEs is called RungeKutta with signature 

```
RungeKutta(function, t_min, t_max, initial_y, tolerance=1e-5, min_step=1e-4, extra_parameters=None,compile_to_C = False)
```

For more informations about arguments see the section reference.