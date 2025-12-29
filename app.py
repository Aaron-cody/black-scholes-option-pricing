import math
import streamlit as st
from scipy.stats import norm


def black_scholes_price(S: float, K: float, T: float, r: float, q: float, sigma: float):
    """
    Black–Scholes prices for European call and put options with continuous dividend yield q.
    Returns: (call, put, d1, d2)
    """
    if S <= 0 or K <= 0:
        raise ValueError("S and K must be positive.")
    if T <= 0:
        raise ValueError("T must be positive (in years).")
    if sigma <= 0:
        raise ValueError("sigma must be positive.")

    d1 = (math.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)

    disc_q = math.exp(-q * T)
    disc_r = math.exp(-r * T)

    call = S * disc_q * norm.cdf(d1) - K * disc_r * norm.cdf(d2)
    put = K * disc_r * norm.cdf(-d2) - S * disc_q * norm.cdf(-d1)
    return call, put, d1, d2


def black_scholes_greeks(S: float, K: float, T: float, r: float, q: float, sigma: float):
    """
    European Greeks with continuous dividend yield q:
    Call Delta, Put Delta, Gamma, Vega (vega per 1.00 change in sigma)
    """
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    pdf_d1 = norm.pdf(d1)

    disc_q = math.exp(-q * T)

    delta_call = disc_q * norm.cdf(d1)
    delta_put = disc_q * (norm.cdf(d1) - 1.0)
    gamma = (disc_q * pdf_d1) / (S * sigma * math.sqrt(T))
    vega = S * disc_q * pdf_d1 * math.sqrt(T)

    return delta_call, delta_put, gamma, vega


def implied_volatility_bisect(
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
    Implied volatility via bisection, consistent with Black–Scholes + dividend yield q.
    option_type: "call" or "put"
    Returns implied sigma.
    """
    if market_price <= 0:
        raise ValueError("Market price must be positive.")

    option_type = option_type.lower().strip()
    if option_type not in {"call", "put"}:
        raise ValueError("option_type must be 'call' or 'put'.")

    def price_given_sigma(sig: float) -> float:
        c, p, _, _ = black_scholes_price(S=S, K=K, T=T, r=r, q=q, sigma=sig)
        return c if option_type == "call" else p

    # Intrinsic value lower bound under dividends, using forward-style discounting
    disc_q = math.exp(-q * T)
    disc_r = math.exp(-r * T)
    intrinsic = max(0.0, S * disc_q - K * disc_r) if option_type == "call" else max(0.0, K * disc_r - S * disc_q)

    if market_price < intrinsic - 1e-12:
        raise ValueError(f"Market price is below intrinsic value ({intrinsic:.6f}).")

    p_low = price_given_sigma(sigma_low)
    p_high = price_given_sigma(sigma_high)

    if not (p_low <= market_price <= p_high):
        raise ValueError(
            "Market price is outside the model price range for the chosen sigma bounds. "
            "Try a larger sigma_high or check inputs."
        )

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
st.caption("European call and put options with continuous dividend yield (q). Built in Python with Streamlit.")

with st.expander("Model overview", expanded=False):
    st.markdown(
        """
**Inputs**
- S: underlying price  
- K: strike price  
- T: time to maturity (years)  
- r: risk-free rate  
- q: dividend yield (continuous)  
- σ: volatility  

**Outputs**
- European call and put prices
- Greeks (Call Delta, Put Delta, Gamma, Vega)
- Implied volatility via bisection

**Notes**
- European options only
- Constant volatility and rates
"""
    )

st.subheader("Inputs")

col1, col2 = st.columns(2)
with col1:
    S = st.number_input("Underlying price (S)", min_value=0.0001, value=45.0, step=1.0)
    K = st.number_input("Strike price (K)", min_value=0.0001, value=40.0, step=1.0)
    T = st.number_input("Time to maturity (T, years)", min_value=0.0001, value=2.0, step=0.25)

with col2:
    r = st.number_input("Risk-free rate (r)", value=0.10, step=0.01, format="%.4f")
    q = st.number_input("Dividend yield (q)", value=0.00, step=0.01, format="%.4f")
    sigma = st.number_input("Volatility (σ)", min_value=0.0001, value=0.10, step=0.01, format="%.4f")

st.subheader("Results")

try:
    call, put, d1, d2 = black_scholes_price(S=S, K=K, T=T, r=r, q=q, sigma=sigma)
    delta_call, delta_put, gamma, vega = black_scholes_greeks(S=S, K=K, T=T, r=r, q=q, sigma=sigma)

    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Call price", f"{call:.4f}")
    r2.metric("Put price", f"{put:.4f}")
    r3.metric(
        "d1",
        f"{d1:.4f}",
        help="Risk-adjusted moneyness term in Black–Scholes (with dividend yield q)."
    )
    r4.metric(
        "d2",
        f"{d2:.4f}",
        help="d1 minus volatility scaled by the square root of time."
    )

    st.markdown("### Greeks")
    g1, g2, g3, g4 = st.columns(4)
    g1.metric("Call Delta", f"{delta_call:.4f}")
    g2.metric("Put Delta", f"{delta_put:.4f}")
    g3.metric("Gamma", f"{gamma:.6f}")
    g4.metric("Vega", f"{vega:.4f}")

    st.markdown("### Implied Volatility")
    st.caption("Implied volatility is the σ that makes the model price match the market price for the selected option type.")

    iv_col1, iv_col2 = st.columns(2)
    with iv_col1:
        option_type_ui = st.selectbox("Option type", ["Call", "Put"])
    option_type = option_type_ui.lower()

    default_market_price = call if option_type_ui == "Call" else put

    with iv_col2:
        market_price = st.number_input(
            "Market option price",
            min_value=0.0001,
            value=float(default_market_price),
            step=0.1
        )

    try:
        iv = implied_volatility_bisect(
            market_price=market_price,
            S=S, K=K, T=T, r=r, q=q,
            option_type=option_type,
            sigma_low=1e-6,
            sigma_high=5.0
        )
        st.success(f"Implied volatility (σ): {iv:.4f} ({iv*100:.2f}%)")
    except Exception as e:
        st.warning(f"IV solver: {e}")

    st.markdown("### Quick checks")
    st.write("Put-Call Parity (with dividends):  C − P = S·e^(−qT) − K·e^(−rT)")
    lhs = call - put
    rhs = S * math.exp(-q * T) - K * math.exp(-r * T)
    st.write(f"LHS (C − P): {lhs:.6f}")
    st.write(f"RHS (S·e^(−qT) − K·e^(−rT)): {rhs:.6f}")
    st.write(f"Absolute difference: {abs(lhs - rhs):.6e}")

except Exception as e:
    st.error(f"Input error: {e}")

st.divider()
st.caption("Limitations: European options only, constant volatility and constant interest and dividend yields.")
st.caption("Portfolio note: extend with more Greeks, implied vol smile/surface, and American option pricing (binomial tree).")
