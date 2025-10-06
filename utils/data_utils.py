import numpy as np
import pandas as pd

def _scale_to_minus1_1(x: pd.Series):
    """Map observed x to z in [-1, 1] for Legendre polynomials."""
    xmin, xmax = x.min(), x.max()
    if np.isclose(xmax, xmin):
        return pd.Series(np.zeros_like(x, dtype=float), index=x.index)
    return 2.0 * (x - xmin) / (xmax - xmin) - 1.0

def add_centered_polynomial(df: pd.DataFrame, cols: list, *, drop_original=False, keep_linear=True):
    """
    For each column in cols, add centered linear and centered squared terms:
      x_c = x - mean(x)
      x_c2 = x_c ** 2
    """
    df = df.copy()
    for c in cols:
        if c not in df.columns:
            continue
        xc = df[c] - df[c].mean()
        df[f"{c}_c"]  = xc
        df[f"{c}_c2"] = xc ** 2
        if not keep_linear:
            df.drop(columns=[f"{c}_c"], inplace=True)
        if drop_original:
            df.drop(columns=[c], inplace=True, errors="ignore")
    return df

def add_legendre_orthogonal(df: pd.DataFrame, cols: list, *, drop_original=False, keep_L1=True):
    """
    For each column in cols, add Legendre-orthogonal polynomials on [-1,1]:
      z = scaled x in [-1,1]
      L1 = P1(z) = z
      L2 = P2(z) = 0.5*(3z^2 - 1)
    Creates columns: c_L1, c_L2.
    """
    df = df.copy()
    for c in cols:
        if c not in df.columns:
            continue
        z = _scale_to_minus1_1(df[c].astype(float))
        L1 = z
        L2 = 0.5 * (3.0 * (z**2) - 1.0)
        df[f"{c}_L1"] = L1
        df[f"{c}_L2"] = L2
        if not keep_L1:
            df.drop(columns=[f"{c}_L1"], inplace=True)
        if drop_original:
            df.drop(columns=[c], inplace=True, errors="ignore")
    return df

def postprocess_orthogonalize(df: pd.DataFrame,
                              *, 
                              center_poly_cols: list = None,
                              legendre_cols: list = None,
                              drop_original_center_inputs: bool = False,
                              drop_original_legendre_inputs: bool = False,
                              keep_center_linear: bool = True,
                              keep_legendre_L1: bool = True):
    """
    One-stop postprocess:
      - build centered polynomials for center_poly_cols (x_c, x_c2)
      - build Legendre-orthogonal polynomials for legendre_cols (x_L1, x_L2)
    """
    df_out = df.copy()
    center_poly_cols = center_poly_cols or []
    legendre_cols = legendre_cols or []

    if center_poly_cols:
        df_out = add_centered_polynomial(
            df_out,
            center_poly_cols,
            drop_original=drop_original_center_inputs,
            keep_linear=keep_center_linear,
        )
    if legendre_cols:
        df_out = add_legendre_orthogonal(
            df_out,
            legendre_cols,
            drop_original=drop_original_legendre_inputs,
            keep_L1=keep_legendre_L1,
        )
    return df_out

