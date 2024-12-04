import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import norm
import warnings
from streamlit_extras.colored_header import colored_header

# Header
colored_header(
    label="The New Desmos",
    description="Welcome to The New Desmos, a free online graphing calculator!",
    color_name="orange-70",
)
st.write("Welcome to my graphing calculator.")

# Styling
st.markdown(
    """
    <style>
        [data-testid="stAppViewContainer"] {
            background-image: url("https://img.freepik.com/free-vector/abstract-paper-style-background_23-2150760175.jpg?t=st=1733193065~exp=1733196665~hmac=490d5cf82e040e13c4912e13bca9e637ddb2c6fc3eef710255a15b6ae37b0e8b&w=996");
            background-size: cover;  
            background-position: center;
        }
        [data-testid="stHeader"] {
            background-color: transparent !important;
        }
        [data-testid="stSidebar"] {
            background-color: #000ba8a;
        }
        [data-testid="stFileUploaderDropzone"] {
            background-color: transparent !important;
        }
        [data-testid="stBaseButton-headerNoPadding"] {
            background-color: #030303;
        }
        [data-testid="stElementContainer"]{
            background-color: #030bfcS;
        }
        [role="slider"] {
            background-color: black; /* Change the background color */
        }
        
    </style>
    """,
    unsafe_allow_html=True,
)

# Global degree slider for polynomial functions
n = st.slider("Choose the degree of x (n)", min_value=1, max_value=10)

x_val, y_val = None, None

# Sidebar for function and data selection
st.sidebar.header("Data Entry")
st.sidebar.write("Enter your data below:")
func = st.sidebar.radio(
    "Choose your function:", ("Linear", "Quadratic", "Sine", "Cosine", "Tangent", "In", "Histogram")
)

# Function definitions
def Linear(x, m, b):
    return m * x + b

def Quad(x, a, b, c):
    return a * x**n + b * x + c

def sin(x, a, b, c):
    return a * np.sin(b * x**n + c)

def cos(x, a, b, c):
    return a * np.cos(b * x**n + c)

def tan(x, a, b, c):
    return a * np.tan(b * x + c)

def In(x, a, b, c):
    return a * np.arcsinh(b * x**n + c)

def gaussian(x, mu, sigma, scale):
    return scale * norm.pdf(x, mu, sigma)

def Histogram():
    if x_val is not None and y_val is not None:
        params, _ = curve_fit(gaussian, x_val, y_val, p0=[np.mean(x_val), np.std(x_val), max(y_val)])
        mu, sigma, scale = params
        st.write(f"Gaussian Fit: μ = {mu:.4f}, σ = {sigma:.4f}, scale = {scale:.4f}")
        x_fit = np.linspace(min(x_val), max(x_val), 1000)
        y_fit = gaussian(x_fit, mu, sigma, scale)
        fig, ax = plt.subplots()
        ax.bar(x_val, y_val, width=(x_val[1] - x_val[0]) if len(x_val) > 1 else 1, color='blue', edgecolor='black')
        ax.plot(x_fit, y_fit, color='orange', label='Fitted Gaussian')
        ax.set_title('Histogram with Gaussian Fit')
        ax.set_xlabel('X-values')
        ax.set_ylabel('Frequency')
        st.pyplot(fig)
    else:
        st.info("Awaiting data input...")

# Function dictionary
func_dict = {
    "Linear": Linear,
    "Quadratic": Quad,
    "Sine": sin,
    "Cosine": cos,
    "Tangent": tan,
    "In": In,
    "Histogram": Histogram,
}

# Data entry option: CSV or manual
Choice = st.sidebar.radio("Choose your data source:", ("CSV", "Manual Input"))

if Choice == "CSV":
    uploaded_file = st.sidebar.file_uploader("Upload your CSV file with x and y columns", type=["csv"])
    try:
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            if "x" in df.columns and "y" in df.columns:
                x_val = df["x"].values
                y_val = df["y"].values
                if len(x_val) != len(y_val):
                    st.error("Number of x values does not match the number of y values.")
                    x_val, y_val = None, None
            else:
                st.error("The CSV file must contain 'x' and 'y' columns.")
                x_val, y_val = None, None
        else:
            st.info("Please upload a CSV file.")
    except Exception as e:
        st.error(f"Error reading CSV file: {e}")

elif Choice == "Manual Input":
    data_inputx = st.sidebar.text_area("Enter x data:")
    data_inputy = st.sidebar.text_area("Enter y data:")
    if data_inputx and data_inputy:
        try:
            x_val = np.array(list(map(float, data_inputx.strip().split(","))))
            y_val = np.array(list(map(float, data_inputy.strip().split(","))))
            if len(x_val) != len(y_val):
                st.error("The number of x and y values must be the same.")
                x_val, y_val = None, None
        except Exception as e:
            st.error("Data format error. Make sure the numbers are separated by a comma.")
            x_val, y_val = None, None
    else:
        x_val, y_val = None, None

# Display the data and perform curve fitting
if x_val is not None and y_val is not None:
    st.write("Entered Data")
    st.write(pd.DataFrame({"x": x_val, "y": y_val}))

    try:
        if func == "Linear":
            params, _ = curve_fit(Linear, x_val, y_val)
            a, b = params
            st.write(f"Linear Fit: y = {a:.4f}x + {b:.4f}")
            fitted_func = Linear
        elif func == "Quadratic":
            fitted_func = lambda x, a, b, c: Quad(x, a, b, c)
            params, _ = curve_fit(fitted_func, x_val, y_val)
        elif func == "Histogram":
            Histogram()
            fitted_func = None
        else:
            fitted_func = func_dict[func]
            params, _ = curve_fit(fitted_func, x_val, y_val)
            a, b, c = params
            st.write(f"{func} Fit: a = {a:.4f}, b = {b:.4f}, c = {c:.4f}")

        if fitted_func:
            y_fitted = fitted_func(x_val, *params)
            absolute_errors = np.abs(y_val - y_fitted)
            percent_error = (np.sum(absolute_errors) / np.sum(np.abs(y_val))) * 100
            st.write(f"Percent Error: {percent_error:.4f}%")

            x_fit = np.linspace(min(x_val), max(x_val), 1000)
            y_fit = fitted_func(x_fit, *params)
            fig, ax = plt.subplots()
            ax.scatter(x_val, y_val, label="Data", color="blue")
            ax.plot(x_fit, y_fit, label="Fitted Curve", color="orange")
            ax.set_title(f"{func} Fit")
            ax.set_xlabel("x-axis")
            ax.set_ylabel("y-axis")
            ax.legend()
            st.pyplot(fig)
    except Exception as e:
        st.error(f"Curve fitting has failed: {e}")

else:
    st.info("Awaiting data input...")