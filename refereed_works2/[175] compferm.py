
import sys
import numpy as np
from scipy.integrate import solve_ivp

# ============================================================
# Bioreactor Compartment Model
# ============================================================

class VerticalBioreactor:

    def __init__(self, N, params):
        self.N = N
        self.p = params

    def mu(self, S, O):
        """Growth kinetics"""
        p = self.p
        return (p["mu_max"]
                * S / (p["K_S"] + S)
                * O / (p["K_O"] + O))

    def rhs(self, t, y):
        """ODE system"""
        N = self.N
        p = self.p

        dydt = np.zeros_like(y)

        # Unpack states
        X = y[0::3]
        S = y[1::3]
        O = y[2::3]

        # Loop over compartments
        for i in range(N):

            # --- Local kinetics ---
            mu_i = self.mu(S[i], O[i])
            rX = mu_i * X[i]
            rS = -(1 / p["Yxs"]) * mu_i * X[i]
            rO = -(1 / p["Yxo"]) * mu_i * X[i]

            # --- Axial advection exchange ---
            adv_X = 0.0
            adv_S = 0.0
            adv_O = 0.0
            if i > 0:
                adv_X += p["Q_up"]/p["V"] * (X[i-1] - X[i])
                adv_S += p["Q_up"]/p["V"] * (S[i-1] - S[i])
                adv_O += p["Q_up"]/p["V"] * (O[i-1] - O[i])
            if i < N-1:
                adv_X += p["Q_down"]/p["V"] * (X[i+1] - X[i])
                adv_S += p["Q_down"]/p["V"] * (S[i+1] - S[i])
                adv_O += p["Q_down"]/p["V"] * (O[i+1] - O[i])

            # --- Axial dispersion (central difference) ---
            disp_X = 0.0
            disp_S = 0.0
            disp_O = 0.0
            if 0 < i < N-1:
                disp_X = p["D_ax"]/p["V"] * (X[i+1] - 2*X[i] + X[i-1])
                disp_S = p["D_ax"]/p["V"] * (S[i+1] - 2*S[i] + S[i-1])
                disp_O = p["D_ax"]/p["V"] * (O[i+1] - 2*O[i] + O[i-1])

            # --- Oxygen mass transfer ---
            kla_i = p["kla_profile"][i]
            transfer_O = kla_i * (p["O_star"] - O[i])

            # --- Substrate feed (top compartment only) ---
            feed_S = 0.0
            if i == N-1:
                feed_S = (p["F_S"]/p["V"]) * (p["S_in"] - S[i])

            # --- Assemble ODEs ---
            dXdt = rX + adv_X + disp_X
            dSdt = rS + adv_S + disp_S + feed_S
            dOdt = rO + adv_O + disp_O + transfer_O
            dydt[3*i]   = dXdt
            dydt[3*i+1] = dSdt
            dydt[3*i+2] = dOdt
        return dydt


def simulate(N, params):
    model = VerticalBioreactor(N, params)

    # Initial conditions
    X0 = np.ones(N) * 0.1
    S0 = np.ones(N) * 5.0
    O0 = np.ones(N) * 0.006

    y0 = np.zeros(3*N)
    y0[0::3] = X0
    y0[1::3] = S0
    y0[2::3] = O0

    t_span = (0, simulation_time)
    t_eval = np.linspace(t_span[0], t_span[1], number_of_steps)
    sol = solve_ivp(model.rhs, t_span, y0,
                    method="BDF",  # stiff solver
                    t_eval=t_eval,
                    atol=1e-8,
                    rtol=1e-6)

    print("Simulation complete.")
    return (model, sol)

def save_data(model, filename):
    import pandas as pd
    data = {}
    # Time column
    data["time"] = t
    # Add compartment columns
    for i in range(model.N):
        data[f"X_{i+1}"] = X[i, :]
        data[f"S_{i+1}"] = S[i, :]
        data[f"O_{i+1}"] = O[i, :]
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    print(f"Results saved to {filename}")

def plot(sol):
    import matplotlib.pyplot as plt
    t = sol.t
    X = sol.y[0::3]
    S = sol.y[1::3]
    O = sol.y[2::3]
    plt.figure(figsize=(12,5))
    # Biomass
    plt.subplot(1,3,1)
    for i in range(model.N): plt.plot(t, X[i], label=f'Comp {i+1}')
    plt.xlabel('Time (h)')
    plt.ylabel('Biomass (g/L)')
    plt.legend()
    plt.title('Biomass')
    # Substrate
    plt.subplot(1,3,2)
    for i in range(model.N): plt.plot(t, S[i], label=f'Comp {i+1}')
    plt.xlabel('Time (h)')
    plt.ylabel('Substrate (g/L)')
    plt.legend()
    plt.title('Substrate')
    # Oxygen
    plt.subplot(1,3,3)
    for i in range(model.N): plt.plot(t, O[i], label=f'Comp {i+1}')
    plt.xlabel('Time (h)')
    plt.ylabel('DO (g/L)')
    plt.legend()
    plt.title('Dissolved Oxygen')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    N = 5  # number of vertical compartments

    simulation_time = 20
    number_of_steps = 500

    filename = "result.csv"

    params = {
        "mu_max": 0.5,      # 1/h; Maximum specific growth rate (E. coli: 0.6–1.2/h; Yeast: 0.3–0.5/h; Mammalian cells: 0.02–0.05/h)
        "K_S": 0.2,         # g/L; Monod half-saturation constant for substrate; Low K_S → high affinity for substrate. High K_S → organism requires higher substrate concentration to grow efficiently.
        "K_O": 0.001,       # g/L; Half-saturation constant for dissolved oxygen; Oxygen saturation (O_star) is usually around 0.007–0.009 g/L at 30°C in water.
        "Yxs": 0.5,         # gX/gS; Gram of biomass yield per gram of substrate
        "Yxo": 1.0,         # gX/gO2; Gram of biomass yield per gram of oxygen
        "Q_up": 2.0,        # L/h; ; Gram of biomass yield per gram of substrate; Large Q_up → strong mixing. Small Q_up → strong vertical gradients
        "Q_down": 2.0,      # L/h; Downward liquid circulation rate
        "D_ax": 0.5,        # L/h (axial dispersion); Axial dispersion coefficient (L/h equivalent) representing turbulent diffusion / back-mixing between adjacent compartments. Higher D_ax → smoother gradients. Lower D_ax → sharper gradients
        "V": 1.0,           # L per compartment; Volume per compartment
        "F_S": 0.2,         # L/h substrate feed; Substrate feed flow rate (L/h); Liquid feed entering the top compartment. If this is zero → batch reactor. If positive → fed-batch or continuous mode
        "S_in": 20.0,       # g/L; Substrate concentration in feed stream
        "O_star": 0.008,    # g/L saturation DO; Oxygen saturation concentration
    }

    # ============================================================
    # Command-line override parsing
    # Usage example: python compferm.py N=7 simulation_time=40 Q_up=3.0
    # ============================================================

    for arg in sys.argv[1:]:
        key, value = arg.split("=")

        # Convert to correct type
        try:
            value = int(value)
        except ValueError:
            try:
                value = float(value)
            except ValueError:
                pass  # keep as string

        if key == "N":
            N = int(value)
        elif key == "simulation_time":
            simulation_time = float(value)
        elif key == "number_of_steps":
            number_of_steps = int(value)
        elif key == "filename":
            filename = value
        elif key in params:
            params[key] = value
        else:
            print(f"Warning: Unknown parameter '{key}'")

    # Generate kla profile AFTER N is finalized
    params["kla_profile"] = np.linspace(200, 50, N)

    (model, sol) = simulate(N, params)


    print("Final compartment states:")
    for i in range(model.N):
        print(f"Compartment {i+1}: X={sol.y[3*i,-1]:.3f}, S={sol.y[3*i+1,-1]:.3f}, O={sol.y[3*i+2,-1]:.5f}")

    # ============================================================
    # Extract compartment-wise concentrations over time
    # ============================================================

    t = sol.t                      # time points
    Y = sol.y                      # state matrix (3N x len(t))

    # Reshape into structured arrays
    X = np.zeros((model.N, len(t)))
    S = np.zeros((model.N, len(t)))
    O = np.zeros((model.N, len(t)))

    for i in range(model.N):
        X[i, :] = Y[3*i, :]
        S[i, :] = Y[3*i + 1, :]
        O[i, :] = Y[3*i + 2, :]

    print("Concentration arrays created:")
    print("X shape:", X.shape)
    print("S shape:", S.shape)
    print("O shape:", O.shape)

    save_data(model, filename)
    plot(sol)



