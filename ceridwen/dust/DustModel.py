import jax.numpy as jnp
from jax import vmap, lax
from collections import defaultdict
from sedpy_jax.attenuation_dust import ATTENUATION_LAWS


class Dust:
    """
    JAX-compatible modular dust model that supports multiple attenuation laws per bin.

    Use this to compute attenuation curves given a binning in stellar age
    and a choice of fixed attenuation law per bin. Parameter fitting
    is only applied to the `fit_params` dictionary passed at runtime.

    Call `Dust.describe_attenuation_laws()` to list all available models.
    """

    def __init__(self, bin_edges, laws, diffuse_law="kriek_conroy"):
        """
        Parameters:
            bin_edges (list of tuple): Age bins in Myr.
            laws (list of str): Attenuation law names (e.g., 'smc', 'kriek_conroy') for each bin.
            diffuse_law (str): Law to apply multiplicatively as diffuse component.
        """
        assert len(bin_edges) == len(laws), "Must have one dust law per bin"
        self.bin_edges = jnp.array(bin_edges)
        self.num_bins = len(bin_edges)
        self.laws = laws
        self.diffuse_law = diffuse_law

        self.law_names_resolved = []
        law_name_counter = defaultdict(int)
        law_occurrences = {name: laws.count(name) for name in set(laws)}

        self.law_funcs = []
        self.law_params = []

        for name in laws:
            count = law_name_counter[name]
            law_name_counter[name] += 1

            law_entry = ATTENUATION_LAWS[name]
            base_func = law_entry["func"]
            defaults = law_entry.get("defaults", {})
            params = law_entry.get("params", {})
            doc = law_entry.get("doc", "")

            if law_occurrences[name] == 1:
                # Only used once → keep original name and function
                resolved_name = name
                func = base_func
                param_dict = self._get_law_params(name)
            else:
                # Used multiple times → rename function and parameters
                number = count + 1
                resolved_name = f"{name}{number}"
                func = modify_function(base_func, number, defaults)

                # Register modified version
                ATTENUATION_LAWS[resolved_name] = {
                    "func": func,
                    "defaults": {f"{k}{number}": v for k, v in defaults.items()},
                    "params": {f"{k}{number}": d for k, d in params.items()},
                    "doc": doc
                }

                # Notify user of renaming
                renamed_keys = [f"{k}{number}" for k in defaults.keys()]
                ignored_keys = list(defaults.keys())
                print(f"[dust setup] Law '{name}' is used multiple times. Registered as '{resolved_name}'.")
                print(f"             Use parameters: {', '.join(renamed_keys)}")
                print(f"             Parameters like {', '.join(ignored_keys)} will be ignored, unless in the diffuse_params dict.")

                # Rename parameters
                param_dict = self._get_law_params(name)
                param_dict = {f"{k}{number}": v for k, v in param_dict.items()}

            self.law_names_resolved.append(resolved_name)
            self.law_funcs.append(func)
            self.law_params.append(param_dict)

        if diffuse_law is not None:
            self.diffuse_params = self._get_law_params(diffuse_law)
            self.diffuse_fn = self._get_law_fn(diffuse_law)
            self.compute_attenuation = self._compute_with_diffuse
        else:
            self.diffuse_params = None
            self.diffuse_fn = None
            self.compute_attenuation = self._compute_no_diffuse

    def _get_law_fn(self, name):
        try:
            return ATTENUATION_LAWS[name]["func"]
        except KeyError:
            raise ValueError(f"Unknown attenuation law: '{name}'. Please add '{name}' to the sedpy_jax.attenuation module, as a function and as an entry in the ATTENUATION_LAWS dictionary.")
    
    def _get_law_params(self, name):
        try:
            return ATTENUATION_LAWS[name]["params"]
        except KeyError:
            raise ValueError(f"Unknown attenuation law: '{name}'. Please add '{name}' to the sedpy_jax.attenuation module, as a function and as an entry in the ATTENUATION_LAWS dictionary.")
        
    def __repr__(self):
        def format_array(arr):
            return "\n    " + "\n    ".join(map(str, arr))

        info = [
            "Dust Model Configuration",
            "=" * 60,
            f"Number of bins          : {self.num_bins}",
            f"Bin edges (Myr)         : {format_array(self.bin_edges)}",
            f"Dust laws               : {', '.join(self.laws)}",
            f"Dust parameters        : {', '.join(map(str, self.law_params))}",
            f"Diffuse dust law        : {self.diffuse_law}",
            f"Diffuse parameters      : {self.diffuse_params}",
            "-" * 60,
        ]

        for i, lawname in enumerate(self.laws):
            lawinfo = ATTENUATION_LAWS.get(lawname, {})
            doc = lawinfo.get("doc", "No description.")
            defaults = lawinfo.get("defaults", {})
            params = lawinfo.get("params", {})
            info.append(f"Bin {i} → {lawname}")
            info.append(f"  Description: {doc}")
            for p, d in params.items():
                info.append(f"    {p:10s}: {d}")
            info.append(f"  Defaults: {defaults}")
            info.append("-" * 60)

        return "\n".join(info)
    
    def _compute_no_diffuse(self, wave, fit_params):
        """
        Compute bin-wise attenuation curves using the provided parameters.

        Parameters:
            wave (jnp.ndarray): Wavelength array in Angstroms.
            fit_params (dict): Parameters for the attenuation laws.
        Returns:
            jnp.ndarray: shape (num_bins, len(wave)) attenuation per bin
        """
        def curve_fn(i, wave, fit_params):
            def wrapped_law_fn(idx):
                return self.law_funcs[idx](wave, **fit_params)
            return lax.switch(i, [lambda: f(wave, **fit_params) for f in self.law_funcs])

        curves = vmap(curve_fn, in_axes=(0, None, None))(jnp.arange(self.num_bins), wave, fit_params)

        return curves
    
    def _compute_with_diffuse(self, wave, fit_params):
        """
        Compute bin-wise attenuation curves using the provided parameters.

        Parameters:
            wave (jnp.ndarray): Wavelength array in Angstroms.
            fit_params (dict): Parameters for the attenuation laws. Must include a subdictionary "diffuse_params" for the diffuse law.
        Returns:
            jnp.ndarray: shape (num_bins, len(wave)) attenuation per bin
        """
        def curve_fn(i, wave, fit_params):
            def wrapped_law_fn(idx):
                return self.law_funcs[idx](wave, **fit_params)
            return lax.switch(i, [lambda: f(wave, **fit_params) for f in self.law_funcs])

        curves = vmap(curve_fn, in_axes=(0, None, None))(jnp.arange(self.num_bins), wave, fit_params)

        diffuse_curve = self.diffuse_fn(
                wave, **fit_params["diffuse_params"])
        
        curves *= diffuse_curve

        return curves
    
    @staticmethod
    def describe_attenuation_laws():
        """
        Print all available dust laws with descriptions and parameter metadata.
        """
        print("=" * 70)
        print("Available Dust Attenuation Laws in sedpy_jax:\n")
        for name, info in ATTENUATION_LAWS.items():
            print(f"• {name}")
            print(f"  Description: {info.get('doc', 'No description.')}")
            print("  Parameters:")
            for param, desc in info.get("params", {}).items():
                print(f"    {param:12s}: {desc}")
            print("-" * 70)

    def get_default_fit_params(self):
        """
        Return a dictionary of default fit parameters based on chosen dust laws.
        Returns:
            dict: {
                'tau': jnp.array([...]),
                'index': jnp.array([...]),
                'diffuse_tau': float,
                'diffuse_index': float
            }
        """

        defaults = {}

        for law in self.law_names_resolved:
            for v, p in ATTENUATION_LAWS[law].get("defaults", {}).items():
                defaults[v] = p
            

        diffuse_defaults = ATTENUATION_LAWS[self.diffuse_law].get("defaults", {})
        defaults['diffuse_params'] = diffuse_defaults

        return defaults
    






# In case the same dust law is applied multiple times, this function takes care of the renaming.

def modify_function(func, number, defaults_dict=None):
    import inspect

    func_name = func.__name__
    sig = inspect.signature(func)

    new_func_name = f"{func_name}{number}"

    new_params = []
    call_params = []
    original_names = []
    for name, param in sig.parameters.items():
        if param.kind == param.VAR_KEYWORD:
            continue
        new_name = f"{name}{number}" if name != "wave" else f"wave"
        original_names.append(name)

        if param.default is not inspect.Parameter.empty:
            default_val = repr(param.default)
        elif defaults_dict and name in defaults_dict:
            default_val = repr(defaults_dict[name])
        else:
            default_val = None

        if default_val is not None:
            new_params.append(f"{new_name}={default_val}")
        else:
            new_params.append(new_name)

        call_params.append(f"{name}={new_name}")

    arg_str = ", ".join(new_params + ["**kwargs"])
    call_str = f"{func_name}({', '.join(call_params)}, **filtered_kwargs)"

    func_def = f"""
def {new_func_name}({arg_str}):
    filtered_kwargs = {{k: v for k, v in kwargs.items() if k not in {original_names!r}}}
    return {call_str}
"""

    local_ns = {func_name: func}
    exec(func_def, local_ns)
    return local_ns[new_func_name]