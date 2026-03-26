import jax.numpy as jnp
from jax import vmap, lax
from functools import partial
from collections import defaultdict
from sedpy_jax.attenuation_dust import ATTENUATION_LAWS

import inspect
from typing import Callable, Sequence


# ---------------------------------------------------------------------------
# make_law_wrapper — dict edition
#
# The original version used getattr(fit_params, name) which required a
# NamedTuple.  The dict version uses fit_params[name], which works with any
# plain Python/JAX dict and is fully traceable inside jax.lax.switch because
# the string key is a static Python value resolved at trace time.
# ---------------------------------------------------------------------------

def make_law_wrapper(f, param_names):
    """Return a JAX-traceable wrapper that extracts named params from a dict."""
    def wrapped(wave, fit_params):
        args = tuple(fit_params[name] for name in param_names)
        return f(wave, *args)
    return wrapped


# In case the same dust law is applied multiple times, this function renames
# the parameters so they do not collide in a shared theta dict.

def modify_function(func, number, defaults_dict=None):
    func_name = func.__name__
    sig = inspect.signature(func)

    new_func_name = f"{func_name}{number}"

    new_params = []
    call_params = []
    original_names = []
    for name, param in sig.parameters.items():
        if param.kind == param.VAR_KEYWORD:
            continue
        new_name = f"{name}{number}" if name != "wave" else "wave"
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


class Dust:
    """
    JAX-compatible modular dust model that supports multiple attenuation laws per bin.

    Parameters are now passed as plain dicts (``dict[str, Array]``) rather than
    NamedTuples.  This is the only interface change relative to the original
    DustModel.py; all computation is identical.

    Call ``Dust.describe_attenuation_laws()`` to list all available models.
    """

    def __init__(self, bin_edges=[(-jnp.inf, -1.97)], laws=['powerlaw']):
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
                resolved_name = name
                func = base_func
                param_dict = self._get_law_params(name)
            else:
                number = count + 1
                resolved_name = f"{name}{number}"
                func = modify_function(base_func, number, defaults)

                ATTENUATION_LAWS[resolved_name] = {
                    "func": func,
                    "defaults": {f"{k}{number}": v for k, v in defaults.items()},
                    "params": {f"{k}{number}": d for k, d in params.items()},
                    "doc": doc
                }

                renamed_keys = [f"{k}{number}" for k in defaults.keys()]
                ignored_keys = list(defaults.keys())
                print(f"[dust setup] Law '{name}' used multiple times → registered as '{resolved_name}'.")
                print(f"             Use parameters: {', '.join(renamed_keys)}")
                print(f"             Parameters like {', '.join(ignored_keys)} will be ignored.")

                param_dict = self._get_law_params(name)
                param_dict = {f"{k}{number}": v for k, v in param_dict.items()}

            sig = inspect.signature(func)
            ordered_param_names = [
                p.name for p in sig.parameters.values()
                if p.name != "wave"
                and p.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
            ]
            ordered_param_names = [n for n in ordered_param_names if n in param_dict]

            if not ordered_param_names:
                ordered_param_names = sorted(param_dict.keys())

            missing = [n for n in ordered_param_names if n not in param_dict]
            if missing:
                raise ValueError(
                    f"Parameters {missing} expected for '{resolved_name}' but not found in param_dict"
                )

            wrapped_func = make_law_wrapper(func, ordered_param_names)

            self.law_names_resolved.append(resolved_name)
            self.law_funcs.append(wrapped_func)
            self.law_params.append(param_dict)

    def _get_law_fn(self, name):
        try:
            return ATTENUATION_LAWS[name]["func"]
        except KeyError:
            raise ValueError(f"Unknown attenuation law: '{name}'.")

    def _get_law_params(self, name):
        try:
            return ATTENUATION_LAWS[name]["params"]
        except KeyError:
            raise ValueError(f"Unknown attenuation law: '{name}'.")

    def __repr__(self):
        def format_array(arr):
            return "\n    " + "\n    ".join(map(str, arr))

        info = [
            "Dust Model Configuration",
            "=" * 60,
            f"Number of bins          : {self.num_bins}",
            f"Bin edges (Gyr)         : {format_array(self.bin_edges)}",
            f"Dust laws               : {', '.join(self.laws)}",
            f"Dust parameters         : {', '.join(map(str, self.law_params))}",
            "-" * 60,
        ]
        for i, lawname in enumerate(self.laws):
            lawinfo = ATTENUATION_LAWS.get(lawname, {})
            doc = lawinfo.get("doc", "No description.")
            params = lawinfo.get("params", {})
            info.append(f"Bin {i} → {lawname}")
            info.append(f"  Description: {doc}")
            for p, d in params.items():
                info.append(f"    {p:10s}: {d}")
            info.append("-" * 60)
        return "\n".join(info)

    def compute_attenuation(self, wave, fit_params):
        """
        Compute bin-wise attenuation curves.

        Parameters
        ----------
        wave : jnp.ndarray
            Wavelength array in Angstroms.
        fit_params : dict[str, Array]
            Parameter dict.  Each law wrapper extracts only the keys it needs.

        Returns
        -------
        jnp.ndarray, shape (num_bins, len(wave))
        """
        def curve_fn(i, wave):
            return lax.switch(i, self.law_funcs, wave, fit_params)

        return vmap(curve_fn, in_axes=(0, None))(jnp.arange(self.num_bins), wave)

    def display(self, fit_params=None):
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
        Return a plain dict of default fit parameters.

        Keys are the parameter names used in the active dust laws; values are
        JAX scalars.  Previously returned a NamedTuple; now returns a dict so
        that it can be merged directly into the global theta dict.
        """
        defaults = {}
        for law in self.law_names_resolved:
            for k, v in ATTENUATION_LAWS[law].get("defaults", {}).items():
                defaults[k] = jnp.asarray(v)
        return defaults

    def get_param_names(self):
        param_names = []
        for law in self.law_names_resolved:
            param_names.extend(ATTENUATION_LAWS[law].get("params", {}).keys())
        return param_names


class DiffuseDust(Dust):
    """
    Single-bin dust model (one law covering all ages) with ``diffuse_`` prefixed
    parameter names to avoid collisions with birth-cloud parameters in a shared
    theta dict.
    """

    def __init__(self, law="kriek_conroy"):
        super().__init__(bin_edges=[(-jnp.inf, jnp.inf)], laws=[law])

        old_name = self.law_names_resolved[0]
        param_dict = self.law_params[0]

        self.diffuse_param_map = {}
        renamed_param_dict = {}
        for k, v in param_dict.items():
            new_k = f"diffuse_{k}"
            renamed_param_dict[new_k] = v
            self.diffuse_param_map[new_k] = k

        self.law_params[0] = renamed_param_dict

        func = ATTENUATION_LAWS[old_name]["func"]
        sig = inspect.signature(func)
        param_names = [
            f"diffuse_{p.name}" for p in sig.parameters.values()
            if p.name != "wave"
            and p.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
        ]
        wrapped_func = make_law_wrapper(func, param_names)
        self.law_funcs[0] = wrapped_func
        self.dust_param_names = param_names

    def get_default_params(self):
        """
        Return a plain dict of default diffuse-dust parameters (``diffuse_*`` keys).
        """
        defaults = {}
        law = self.law_names_resolved[0]
        for k, v in ATTENUATION_LAWS[law].get("defaults", {}).items():
            defaults[f"diffuse_{k}"] = jnp.asarray(v)
        return defaults

    def compute_attenuation(self, wave, fit_params):
        """
        Compute the diffuse attenuation curve (single bin).

        Parameters
        ----------
        wave : jnp.ndarray
        fit_params : dict[str, Array]

        Returns
        -------
        jnp.ndarray, shape (len(wave),)
        """
        return self.law_funcs[0](wave, fit_params)

    def get_param_names(self):
        return list(self.law_params[0].keys())

    def __repr__(self):
        return super().__repr__().replace(
            "Dust Model Configuration", "Diffuse Dust Model Configuration"
        )
