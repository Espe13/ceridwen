import jax.numpy as jnp
from jax import vmap, lax
from functools import partial
from collections import defaultdict
from sedpy_jax.attenuation_dust import ATTENUATION_LAWS


import inspect
from typing import Callable, Sequence


class Dust:
    """
    JAX-compatible modular dust model that supports multiple attenuation laws per bin.

    Use this to compute attenuation curves given a binning in stellar age
    and a choice of fixed attenuation law per bin. Parameter fitting
    is only applied to the `fit_params` dictionary passed at runtime.

    Call `Dust.describe_attenuation_laws()` to list all available models.
    """

    def __init__(self, bin_edges = [(-jnp.inf, -1.97)], laws = ['powerlaw']):
        """
        Parameters:
            bin_edges (list of tuple): Age bins in Gyr.
            laws (list of str): Attenuation law names (e.g., 'smc', 'kriek_conroy') for each bin.
        """
        assert len(bin_edges) == len(laws), "Must have one dust law per bin"
        self.bin_edges = jnp.array(bin_edges)
        self.num_bins = len(bin_edges)
        self.laws = laws

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
                print(f"             Parameters like {', '.join(ignored_keys)} will be ignored.")

                # Rename parameters
                param_dict = self._get_law_params(name)
                param_dict = {f"{k}{number}": v for k, v in param_dict.items()}

            sig = inspect.signature(func)

            # Keep parameters after 'wave' and ignore *args/**kwargs
            ordered_param_names = [
                p.name for p in sig.parameters.values()
                if p.name != "wave"
                and p.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
            ]

            ordered_param_names = [n for n in ordered_param_names if n in param_dict]

            # Fallback: if signature-based detection fails, use param_dict keys (stably sorted)
            if not ordered_param_names:
                ordered_param_names = sorted(param_dict.keys())

            missing = [n for n in ordered_param_names if n not in param_dict]
            if missing:
                raise ValueError(f"Parameters {missing} expected for '{resolved_name}' but not found in param_dict")

            # Create the wrapped function (this is done in Python, not inside jit)
            wrapped_func = make_law_wrapper(func, ordered_param_names)
        

            self.law_names_resolved.append(resolved_name)
            self.law_funcs.append(wrapped_func)
            self.law_params.append(param_dict)

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
            f"Bin edges (Gyr)         : {format_array(self.bin_edges)}",
            f"Dust laws               : {', '.join(self.laws)}",
            f"Dust parameters        : {', '.join(map(str, self.law_params))}",
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


    def compute_attenuation(self, wave, fit_params):
        """
        Compute bin-wise attenuation curves using the provided parameters.

        Parameters:
            wave (jnp.ndarray): Wavelength array in Angstroms.
            fit_params (NamedTuple): Parameters for the attenuation laws.
        Returns:
            jnp.ndarray: shape (num_bins, len(wave)) attenuation per bin
        """

        def curve_fn(i, wave):
            return lax.switch(i, self.law_funcs, wave, fit_params)

        curves = vmap(curve_fn, in_axes=(0, None))(jnp.arange(self.num_bins), wave)
        return curves

    def display(self, fit_params=None):
        """
        Display the attenuation curves for each bin using matplotlib.
        Returns:
            fig, ax: Matplotlib figure and axis objects.
        """
        import matplotlib.pyplot as plt
        wave = jnp.linspace(0, 10000, 1000)

        if fit_params is None:
            fit_params = self.get_default_fit_params()

        curves = self.compute_attenuation(wave, fit_params=fit_params)
        fig, ax = plt.subplots()

        for i in range(len(curves)):
            law = self.law_names_resolved[i]
            t_start, t_end = self.bin_edges[i]
            ax.plot(wave, curves[i], label=f"Bin {i+1}: {law} ({t_start:.0f}–{t_end:.0f} Myr)")

        ax.set_xlabel("Wavelength (Angstroms)")
        ax.set_ylabel("Attenuation")
        ax.set_yscale("log")
        ax.legend()
        plt.show()
        return fig, ax
    
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
        from typing import NamedTuple, Any
        """
        Return a NamedTuple of default fit parameters based on chosen dust laws.
        Returns:
            NamedTuple: e.g. FitParams(tau_smc1=..., tau_smc2=..., dust_index=...)
        """
        defaults = {}

        for law in self.law_names_resolved:
            for v, p in ATTENUATION_LAWS[law].get("defaults", {}).items():
                defaults[v] = p

        # Dynamically create a NamedTuple class with fields matching defaults
        FitParams = NamedTuple("FitParams", [(k, Any) for k in defaults.keys()])
        return FitParams(**defaults)
    
    def get_param_names(self):
        """
        Return a list of all parameter names used in the dust laws.
        Returns:
            list of str: Parameter names.
        """
        param_names = []
        for law in self.law_names_resolved:
            param_names.extend(ATTENUATION_LAWS[law].get("params", {}).keys())
        return param_names


def make_law_wrapper(f, param_names):
    """Return a JAX‑traceable wrapper that extracts given params from a NamedTuple."""
    def wrapped(wave, fit_params):
        args = tuple(getattr(fit_params, name) for name in param_names)
        return f(wave, *args)
    return wrapped


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


class DiffuseDust(Dust):
    def __init__(self, law="kriek_conroy"):
        """
        A simplified dust model that applies one dust law to all ages (bin spans all time).
        
        Parameters:
            law (str): Name of the attenuation law (must be in ATTENUATION_LAWS).
        """
        # Initialize parent class with one bin spanning all time
        super().__init__(bin_edges=[(-jnp.inf, jnp.inf)], laws=[law])

        # Rename all parameters in the single bin to use prefix 'diffuse_'
        old_name = self.law_names_resolved[0]
        param_dict = self.law_params[0]

        # Create remapped version
        self.diffuse_param_map = {}
        renamed_param_dict = {}
        for k, v in param_dict.items():
            new_k = f"diffuse_{k}"
            renamed_param_dict[new_k] = v
            self.diffuse_param_map[new_k] = k

        # Replace the first bin’s param dict with the renamed version
        self.law_params[0] = renamed_param_dict

        # Also update the wrapped function to use the renamed keys
        func = ATTENUATION_LAWS[old_name]["func"]
        sig = inspect.signature(func)

        param_names = [
            f"diffuse_{p.name}" for p in sig.parameters.values()
            if p.name != "wave"
            and p.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
        ]

        wrapped_func = make_law_wrapper(func, param_names)
        self.law_funcs[0] = wrapped_func

        # Save param name list for reference
        self.dust_param_names = param_names

    def get_default_params(self):
        """
        Return a NamedTuple of default fit parameters, with names prefixed by 'diffuse_'.
        """
        from typing import NamedTuple, Any

        defaults = {}
        law = self.law_names_resolved[0]
        for k, v in ATTENUATION_LAWS[law].get("defaults", {}).items():
            defaults[f"diffuse_{k}"] = v

        FitParams = NamedTuple("DiffuseParams", [(k, Any) for k in defaults.keys()])
        return FitParams(**defaults)
    

    def compute_attenuation(self, wave, fit_params):
        """
        Compute the diffuse attenuation curve (single bin).

        Parameters:
            wave (jnp.ndarray): Wavelength array in Angstroms.
            fit_params (NamedTuple): Parameters for the attenuation law.

        Returns:
            jnp.ndarray: shape (len(wave),) attenuation curve
        """
        # Always use the single law function (index 0)
        return self.law_funcs[0](wave, fit_params)
    
    def get_param_names(self):
        return list(self.law_params[0].keys())

    def __repr__(self):
        base_repr = super().__repr__()
        return base_repr.replace("Dust Model Configuration", "Diffuse Dust Model Configuration")