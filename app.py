import math
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from scipy.stats import norm


# ------------------- Black–Scholes (with dividend yield q) -------------------

def bs_price(S: float, K: float, T: float, r: float, q: float, sigma: float):
    """
    Black–Scholes European option prices with continuous dividend yield q.
    Returns: call, put, d1, d2
    """
    if S <= 0 or K <= 0:
        raise ValueError("S and K must be positive.")
    if T <= 0:
        raise ValueError("T must be positive (in years).")
    if sigma <= 0:
        raise ValueError("sigma must be positive.")

    sqrtT = math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT

    disc_q = math.exp(-q * T)
    disc_r = math.exp(-r * T)

    call = S * disc_q * norm.cdf(d1) - K * disc_r * norm.cdf(d2)
    put = K * disc_r * norm.cdf(-d2) - S * disc_q * norm.cdf(-d1)
    return call, put, d1, d2


def bs_greeks(S: float, K: float, T: float, r: float, q: float, sigma: float):
    """
    Returns Greeks for European options with dividend yield q:
    Delta (call, put), Gamma, Vega, Theta (call, put), Rho (call, put)
    Notes:
      - Vega is per 1.00 change in sigma (so multiply by 0.01 for per 1% vol)
      - Theta is per year (divide by 365 for per day)
      - Rho is per 1.00 change in r (multiply by 0.01 for per 1% rate)
    """
    sqrtT = math.sqrt(T)
    disc_q = math.exp(-q * T)
    disc_r = math.exp(-r * T)

    d1 = (math.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT

    pdf_d1 = norm.pdf(d1)

    delta_call = disc_q * norm.cdf(d1)
    delta_put = disc_q * (norm.cdf(d1) - 1.0)

    gamma = (disc_q * pdf_d1) / (S * sigma * sqrtT)
    vega = S * disc_q * pdf_d1 * sqrtT

    theta_call = (
        -(S * disc_q * pdf_d1 * sigma) / (2 * sqrtT)
        - r * K * disc_r * norm.cdf(d2)
        + q * S * disc_q * norm.cdf(d1)
    )
    theta_put = (
        -(S * disc_q * pdf_d1 * sigma) / (2 * sqrtT)
        + r * K * disc_r * norm.cdf(-d2)
        - q * S * disc_q * norm.cdf(-d1)
    )

    rho_call = K * T * disc_r * norm.cdf(d2)
    rho_put = -K * T * disc_r * norm.cdf(-d2)

    return {
        "delta_call": delta_call,
        "delta_put": delta_put,
        "gamma": gamma,
        "vega": vega,
        "theta_call": theta_call,
        "theta_put": theta_put,
        "rho_call": rho_call,
        "rho_put": rho_put,
    }


def implied_vol_bisect(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
    option_type: str = "call",
    sigma_low: float = 1e-6,
    sigma_high: float = 5.0,
    tol: float = 1e-6,
    max_iter: int = 200,
):
    """
    Implied volatility via bisection for BS with dividend yield q.
    """
    if market_price <= 0:
        raise ValueError("Market price must be positive.")

    option_type = option_type.lower().strip()
    if option_type not in {"call", "put"}:
        raise ValueError("option_type must be 'call' or 'put'.")

    def price_given_sigma(sig: float) -> float:
        c, p, _, _ = bs_price(S, K, T, r, q, sig)
        return c if option_type == "call" else p

    # Lower bound check against discounted intrinsic
    disc_q = math.exp(-q * T)
    disc_r = math.exp(-r * T)
    intrinsic = max(0.0, S * disc_q - K * disc_r) if option_type == "call" else max(0.0, K * disc_r - S * disc_q)

    if market_price < intrinsic - 1e-12:
        raise ValueError(f"Market price is below intrinsic value ({intrinsic:.6f}).")

    p_low = price_given_sigma(sigma_low)
    p_high = price_given_sigma(sigma_high)

    if not (p_low <= market_price <= p_high):
        raise ValueError("Market price outside model price range. Increase sigma_high or check inputs.")

    lo, hi = sigma_low, sigma_high
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        p_mid = price_given_sigma(mid)
        if abs(p_mid - market_price) < tol:
            return mid
        if p_mid < market_price:
            lo = mid
        else:
            hi = mid

    return 0.5 * (lo + hi)


# ------------------- Streamlit UI -------------------

st.set_page_config(page_title="Black–Scholes Option Pricing Calculator", layout="centered")

st.title("Black–Scholes Option Pricing Calculator")
st.caption("European call and put options using Black–Scholes. Includes Greeks, implied volatility, and sensitivity plots.")

with st.expander("Model overview", expanded=False):
    st.markdown(
        """
**Inputs (user-friendly)**
- Spot price (S)
- Strike price (K)
- Time to maturity (T in years)
- Risk-free rate (r, %)
- Volatility (σ, %)
- Dividend yield (q, %)

**Model**
- European options
- Continuous dividend yield
- Constant r and σ
"""
    )

st.subheader("Inputs")

col1, col2 = st.columns(2)
with col1:
    S = st.number_input("Spot price (S)", min_value=0.0001, value=45.0, step=1.0)
    K = st.number_input("Strike price (K)", min_value=0.0001, value=40.0, step=1.0)
    T = st.number_input("Time to maturity (years)", min_value=0.0001, value=2.0, step=0.25)

with col2:
    r_pct = st.number_input("Risk-free rate (%)", value=10.0, step=0.25, format="%.4f")
    sigma_pct = st.number_input("Volatility (%)", min_value=0.0001, value=10.0, step=0.25, format="%.4f")
    q_pct = st.number_input("Dividend yield (%)", min_value=0.0, value=0.0, step=0.25, format="%.4f")

# Convert to decimals internally
r = r_pct / 100.0
sigma = sigma_pct / 100.0
q = q_pct / 100.0

st.subheader("Results")

try:
    call, put, d1, d2 = bs_price(S, K, T, r, q, sigma)
    greeks = bs_greeks(S, K, T, r, q, sigma)

    r1, r2 = st.columns(2)
    r1.metric("Call price", f"{call:.4f}")
    r2.metric("Put price", f"{put:.4f}")

    with st.expander("Greeks", expanded=True):
        g1, g2, g3, g4 = st.columns(4)
        g1.metric("Call Delta", f"{greeks['delta_call']:.4f}")
        g2.metric("Put Delta", f"{greeks['delta_put']:.4f}")
        g3.metric("Gamma", f"{greeks['gamma']:.6f}")
        g4.metric("Vega", f"{greeks['vega']:.4f}")

        g5, g6, g7, g8 = st.columns(4)
        g5.metric("Call Theta (per year)", f"{greeks['theta_call']:.4f}")
        g6.metric("Put Theta (per year)", f"{greeks['theta_put']:.4f}")
        g7.metric("Call Rho", f"{greeks['rho_call']:.4f}")
        g8.metric("Put Rho", f"{greeks['rho_put']:.4f}")

        st.caption("Tip: Vega and Rho are per 1.00 change. Multiply by 0.01 for per 1% change. Theta shown per year.")

    with st.expander("Model details (d1, d2, parity)", expanded=False):
        st.write(f"d1: {d1:.6f}")
        st.write(f"d2: {d2:.6f}")

        st.markdown("**Put–Call Parity (with dividend yield q)**")
        st.write("C − P = S·e^(−qT) − K·e^(−rT)")
        lhs = call - put
        rhs = S * math.exp(-q * T) - K * math.exp(-r * T)
        st.write(f"LHS (C − P): {lhs:.6f}")
        st.write(f"RHS: {rhs:.6f}")
        st.write(f"Absolute difference: {abs(lhs - rhs):.6e}")

    st.markdown("### Implied Volatility")
    iv_col1, iv_col2 = st.columns(2)
    with iv_col1:
        option_type = st.selectbox("Option type", ["Call", "Put"], index=0)
    default_market_price = call if option_type == "Call" else put
    with iv_col2:
        market_price = st.number_input("Market option price", min_value=0.0001, value=float(default_market_price), step=0.1)

    try:
        iv = implied_vol_bisect(
            market_price=market_price,
            S=S, K=K, T=T,
            r=r, q=q,
            option_type=option_type.lower(),
        )
        st.success(f"Implied volatility (σ): {iv:.4f} ({iv * 100:.2f}%)")
    except Exception as e:
        st.warning(f"IV solver: {e}")

    st.markdown("### Sensitivity Analysis")
    mode = st.selectbox(
        "Choose sensitivity plot",
        ["Option value vs Spot (S)", "Option value vs Volatility (σ)", "Option value vs Time (T)"],
        index=0,
    )

    if mode == "Option value vs Spot (S)":
        S_grid = np.linspace(max(0.01, 0.5 * S), 1.5 * S, 40)
        call_vals = []
        put_vals = []
        for s_ in S_grid:
            c_, p_, _, _ = bs_price(float(s_), K, T, r, q, sigma)
            call_vals.append(c_)
            put_vals.append(p_)
        x = S_grid
        xlabel = "Spot price (S)"

    elif mode == "Option value vs Volatility (σ)":
        sig_grid = np.linspace(max(0.0001, 0.5 * sigma), 1.5 * sigma, 40)
        call_vals = []
        put_vals = []
        for sig_ in sig_grid:
            c_, p_, _, _ = bs_price(S, K, T, r, q, float(sig_))
            call_vals.append(c_)
            put_vals.append(p_)
        x = sig_grid * 100.0
        xlabel = "Volatility (%)"

    else:
        T_grid = np.linspace(max(0.0001, 0.05 * T), 1.5 * T, 40)
        call_vals = []
        put_vals = []
        for t_ in T_grid:
            c_, p_, _, _ = bs_price(S, K, float(t_), r, q, sigma)
            call_vals.append(c_)
            put_vals.append(p_)
        x = T_grid
        xlabel = "Time to maturity (years)"

    fig = plt.figure()
    plt.plot(x, call_vals, label="Call")
    plt.plot(x, put_vals, label="Put")
    plt.xlabel(xlabel)
    plt.ylabel("Option price")
    plt.legend()
    plt.grid(True, alpha=0.25)
    st.pyplot(fig)

except Exception as e:
    st.error(f"Input error: {e}")

st.divider()
st.caption("Limitations: European options, Black–Scholes assumptions, constant volatility and rates.")