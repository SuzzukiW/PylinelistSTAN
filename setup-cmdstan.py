from cmdstanpy import install_cmdstan

def setup():
    print("Installing CmdStan...")
    install_cmdstan()
    print("CmdStan installation complete.")

if __name__ == "__main__":
    setup()